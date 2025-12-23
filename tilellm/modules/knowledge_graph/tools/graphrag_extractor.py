"""
Simplified GraphRAG extractor for entity and relationship extraction from text.
Adapted from original GraphRAG implementation but without external dependencies.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

GRAPH_FIELD_SEP = "<SEP>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]

# Simplified GraphRAG extraction prompt (adapted from original)
GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized, in language of 'Text'
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities in language of 'Text'
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other in language of 'Text'
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
######################

-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes and control characters."""
    if not isinstance(input, str):
        return str(input)
    
    import html
    result = html.unescape(input.strip())
    # Remove control characters
    return re.sub(r'[\"\x00-\x1f\x7f-\x9f]', '', result)


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    
    pattern = "|".join(re.escape(marker) for marker in markers)
    results = re.split(pattern, content)
    return [r.strip() for r in results if r.strip()]


def handle_single_entity_extraction(record_attributes: list[str], chunk_key: str):
    """Extract entity from a parsed record."""
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    
    return {
        "entity_name": entity_name.upper(),
        "entity_type": entity_type.upper(),
        "description": entity_description,
        "source_id": chunk_key,
    }


def handle_single_relationship_extraction(record_attributes: list[str], chunk_key: str):
    """Extract relationship from a parsed record."""
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_keywords = clean_str(record_attributes[4]) if len(record_attributes) > 4 else ""
    
    # Try to extract weight (last element might be numeric)
    weight = 1.0
    if len(record_attributes) > 5:
        try:
            weight = float(record_attributes[-1])
        except ValueError:
            pass
    
    pair = sorted([source.upper(), target.upper()])
    return {
        "src_id": pair[0],
        "tgt_id": pair[1],
        "weight": weight,
        "description": edge_description,
        "keywords": edge_keywords,
        "source_id": chunk_key,
    }


def flat_uniq_list(arr, key):
    """Flatten and deduplicate a list of values from dicts."""
    res = []
    for a in arr:
        val = a.get(key)
        if isinstance(val, list):
            res.extend(val)
        elif val is not None:
            res.append(val)
    return list(set(res))


class GraphRAGExtractor:
    """
    Simplified GraphRAG extractor for entity and relationship extraction.
    Uses LLM to parse text and extract structured knowledge.
    """
    
    def __init__(self, llm_invoker, language: str = "English", entity_types: Optional[List[str]] = None):
        """
        Initialize extractor.
        
        Args:
            llm_invoker: LLM instance with invoke/chat method (LangChain ChatModel)
            language: Output language for descriptions
            entity_types: List of entity types to extract
        """
        self.llm = llm_invoker
        self.language = language
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
    
    async def extract_chunk(self, chunk_key: str, chunk_text: str) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        """
        Extract entities and relationships from a single text chunk.
        
        Returns:
            Tuple of (entities_dict, relationships_dict, token_count)
        """
        # Prepare prompt variables
        prompt_vars = {
            "entity_types": ", ".join(self.entity_types),
            "input_text": chunk_text,
            "tuple_delimiter": "\",\"",
            "record_delimiter": "\n",
            "completion_delimiter": "[COMPLETED]"
        }
        
        prompt = GRAPH_EXTRACTION_PROMPT.format(**prompt_vars)
        
        try:
            # Call LLM - assume it has invoke method (LangChain style)
            if hasattr(self.llm, 'invoke'):
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content="You are a helpful assistant that extracts entities and relationships from text."),
                    HumanMessage(content=prompt)
                ]
                response = await self.llm.ainvoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
            elif hasattr(self.llm, 'chat'):
                # Assume chat method takes system and messages
                response_text = await self.llm.chat(
                    system="You are a helpful assistant that extracts entities and relationships from text.",
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                # Assume it's a callable that returns text
                response_text = await self.llm(prompt)
            
            # Parse response
            lines = response_text.strip().split('\n')
            maybe_nodes = defaultdict(list)
            maybe_edges = defaultdict(list)
            
            for line in lines:
                line = line.strip()
                if not line or '[COMPLETED]' in line:
                    continue
                
                # Remove surrounding parentheses if present
                if line.startswith('(') and line.endswith(')'):
                    line = line[1:-1]
                
                record_attributes = split_string_by_multi_markers(line, ['","'])
                
                # Try to parse as entity
                entity = handle_single_entity_extraction(record_attributes, chunk_key)
                if entity:
                    maybe_nodes[entity["entity_name"]].append(entity)
                    continue
                
                # Try to parse as relationship
                rel = handle_single_relationship_extraction(record_attributes, chunk_key)
                if rel:
                    maybe_edges[(rel["src_id"], rel["tgt_id"])].append(rel)
            
            return dict(maybe_nodes), dict(maybe_edges), len(response_text.split())
            
        except Exception as e:
            logger.error(f"Error extracting from chunk {chunk_key}: {e}")
            return {}, {}, 0
    
    async def extract(self, doc_id: str, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities and relationships from multiple chunks.
        
        Args:
            doc_id: Document identifier
            chunks: List of dictionaries containing 'id' and 'text'
            
        Returns:
            Tuple of (entities_list, relationships_list)
        """
        all_nodes = defaultdict(list)
        all_edges = defaultdict(list)
        total_tokens = 0
        
        # Process chunks sequentially (can be parallelized later)
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"{doc_id}_{i}")
            chunk_text = chunk.get("text", "")
            
            if not chunk_text:
                continue
                
            logger.info(f"Processing chunk {i+1}/{len(chunks)} (ID: {chunk_id})")
            
            nodes, edges, tokens = await self.extract_chunk(chunk_id, chunk_text)
            total_tokens += tokens
            
            # Merge results
            for entity_name, entity_list in nodes.items():
                all_nodes[entity_name].extend(entity_list)
            
            for edge_key, edge_list in edges.items():
                all_edges[edge_key].extend(edge_list)
        
        logger.info(f"Extracted {len(all_nodes)} unique entities and {len(all_edges)} unique relationships from {len(chunks)} chunks")
        
        # Merge duplicate entities
        merged_entities = []
        for entity_name, entity_list in all_nodes.items():
            if not entity_list:
                continue
            
            # Determine most common entity type
            type_counts = {}
            for entity in entity_list:
                entity_type = entity.get("entity_type", "UNKNOWN")
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "UNKNOWN"
            
            # Merge descriptions
            descriptions = [e.get("description", "") for e in entity_list]
            unique_descriptions = list(set(desc for desc in descriptions if desc.strip()))
            merged_description = GRAPH_FIELD_SEP.join(sorted(unique_descriptions))
            
            # Merge source IDs
            source_ids = flat_uniq_list(entity_list, "source_id")
            
            merged_entities.append({
                "entity_name": entity_name,
                "entity_type": most_common_type,
                "description": merged_description,
                "source_id": source_ids
            })
        
        # Merge duplicate relationships
        merged_relationships = []
        for (src, tgt), rel_list in all_edges.items():
            if not rel_list:
                continue
            
            # Merge weights, descriptions, and keywords
            total_weight = sum(rel.get("weight", 1.0) for rel in rel_list)
            descriptions = [rel.get("description", "") for rel in rel_list]
            unique_descriptions = list(set(desc for desc in descriptions if desc.strip()))
            merged_description = GRAPH_FIELD_SEP.join(sorted(unique_descriptions))
            
            keywords = flat_uniq_list(rel_list, "keywords")
            source_ids = flat_uniq_list(rel_list, "source_id")
            
            merged_relationships.append({
                "src_id": src,
                "tgt_id": tgt,
                "weight": total_weight,
                "description": merged_description,
                "keywords": keywords,
                "source_id": source_ids
            })
        
        return merged_entities, merged_relationships