import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from tilellm.modules.knowledge_graph.tools.graphrag_extractor import GRAPH_FIELD_SEP
# Import extraction configurations
from tilellm.modules.knowledge_graph_falkor.tools.extraction_prompts import (
    get_extraction_config,
    list_available_domains,
    ExtractionConfig
)

logger = logging.getLogger(__name__)

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
    # Keep empty strings to preserve field positions
    return [r.strip() for r in results]


def handle_single_entity_extraction(record_attributes: list[str], chunk_key: str):
    """Extract entity from a parsed record."""
    if len(record_attributes) < 4:
        return None

    # Check if first field (after cleaning) is "entity"
    record_type = clean_str(record_attributes[0])
    if record_type != 'entity':
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
    """
    Extract relationship from a parsed record.

    REQUIRED FORMAT (relationship_type is MANDATORY):
    ("relationship", source, target, relationship_type, description, strength, date)

    Minimum 4 fields required. If relationship_type is missing, the record is rejected.
    """
    # Need at least: relationship, source, target, type
    if len(record_attributes) < 4:
        logger.debug(f"Skipping relationship - insufficient fields (need 4+, got {len(record_attributes)}): {record_attributes[:3] if len(record_attributes) >= 3 else record_attributes}")
        return None

    # Check if first field (after cleaning) is "relationship"
    record_type = clean_str(record_attributes[0])
    if record_type != 'relationship':
        return None

    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())

    if not source.strip() or not target.strip():
        logger.debug(f"Skipping relationship - empty source or target")
        return None

    # Field 3 MUST be the relationship type
    relationship_type = clean_str(record_attributes[3].upper())

    # Validate relationship type is not a description
    if not relationship_type or len(relationship_type.split()) > 3:
        logger.debug(f"Skipping relationship - field 3 looks like description, not type: {record_attributes[3][:50]}")
        return None
    
    # Detect old format (relationship, source, target, description, weight)
    # If we have exactly 5 fields and the 5th field looks like a number, it's likely old format
    if len(record_attributes) == 5:
        try:
            float(clean_str(record_attributes[4]))
            # 5th field is numeric, meaning field 3 is description, not type
            logger.debug(f"Skipping relationship - detected old format (description in field 3): {record_attributes[3][:50]}")
            return None
        except ValueError:
            pass  # Not numeric, likely new format with missing date/weight

    # Extract remaining fields
    edge_description = clean_str(record_attributes[4]) if len(record_attributes) > 4 else ""

    weight = 1.0
    if len(record_attributes) > 5:
        try:
            weight = float(clean_str(record_attributes[5]))
        except ValueError:
            logger.debug(f"Invalid weight value '{record_attributes[5]}', using default 1.0")
            weight = 1.0

    edge_date = None
    if len(record_attributes) > 6:
        edge_date = clean_str(record_attributes[6])

    # Log for debugging
    logger.info(f"Extracted relationship: {source} -[{relationship_type}]-> {target} (weight: {weight})")

    return {
        "src_id": source.upper(),
        "tgt_id": target.upper(),
        "relationship_type": relationship_type,
        "weight": weight,
        "description": edge_description,
        "keywords": relationship_type,
        "date": edge_date,
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


def should_skip_line(line: str) -> bool:
    """
    Check if a line should be skipped during parsing.
    
    Lines containing only markdown formatting, separators, or
    clearly non-data content should be ignored.
    """
    # Already handled in caller: empty lines or [COMPLETED]
    line_lower = line.lower()
    
    # Skip markdown formatting lines
    if line.startswith(('```', '***', '---', '===', '###', '## ', '# ', '**', '* ', '- ', '_ ', '> ', '|', '//', '/*', '<!--', '-->')):
        return True
    
    # Skip lines that are just asterisks or other separators
    if line.strip() in ('**', '*', '-', '_', '---', '***', '____', '----------', '=========='):
        return True
    
    # Skip numbered list items (e.g., "1.", "2. ")
    import re
    if re.match(r'^\d+\.\s', line):
        return True
    
    # Skip lines that are common LLM disclaimers or instructions
    skip_keywords = [
        'output:', 'example:', 'note:', 'important:', 'warning:', 
        'format:', 'entity_types:', 'text:', 'goal:', 'steps:',
        'here is', 'following', 'please note', 'remember that',
        'make sure', 'ensure that', 'do not', 'don\'t',
        'real data:', 'examples:', 'example 1:', 'example 2:',
        'entity_types:', 'text:', 'output:', 'completed',
        '```plaintext', '```json', '```python', '```txt'
    ]
    if any(keyword in line_lower for keyword in skip_keywords):
        return True
    
    # Skip lines that don't contain quotes (data lines should have quotes)
    # But be careful - some LLMs might output without quotes in some cases
    # So we'll be conservative: if line has no quotes AND looks like plain text (has spaces)
    # then skip it
    if '"' not in line and "'" not in line and len(line.split()) > 2:
        # Might be a plain text description line
        return True
    
    return False


class GraphRAGExtractor:
    """
    Simplified GraphRAG extractor for entity and relationship extraction.
    Uses LLM to parse text and extract structured knowledge.

    Supports multiple domains via creation_prompt parameter.
    """

    def __init__(
        self,
        llm_invoker,
        language: str = "English",
        entity_types: Optional[List[str]] = None,
        creation_prompt: Optional[str] = None
    ):
        """
        Initialize extractor.

        Args:
            llm_invoker: LLM instance with invoke/chat method (LangChain ChatModel)
            language: Output language for descriptions
            entity_types: List of entity types to extract (overrides creation_prompt config)
            creation_prompt: Domain identifier (e.g., "debt_recovery", "generic").
                           If None, uses "generic" domain. If entity_types is provided,
                           it overrides the config's entity types.
        """
        self.llm = llm_invoker
        self.language = language

        # Get extraction configuration based on creation_prompt
        self.config = get_extraction_config(creation_prompt)

        # Allow entity_types override, otherwise use config
        self.entity_types = entity_types or self.config.entity_types
        self.relationship_types = self.config.relationship_types
        self.extraction_prompt_template = self.config.extraction_prompt

        logger.info(f"Initialized GraphRAGExtractor with domain: {self.config.domain}, "
                   f"entity_types: {len(self.entity_types)}, "
                   f"relationship_types: {len(self.relationship_types)}")
    
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

        # Use the extraction prompt from config
        prompt = self.extraction_prompt_template.format(**prompt_vars)
        
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

            # DEBUG: Log first 500 chars of LLM response
            logger.info(f"LLM extraction response (first 500 chars): {response_text[:500]}")

            # Parse response
            lines = response_text.strip().split('\n')
            maybe_nodes = defaultdict(list)
            maybe_edges = defaultdict(list)

            for line in lines:
                line = line.strip()
                if not line or '[COMPLETED]' in line:
                    continue
                
                # Skip lines that are clearly markdown formatting or non-data
                if should_skip_line(line):
                    logger.debug(f"Skipping non-data line: {line[:100]}")
                    continue
                
                # Remove surrounding parentheses if present
                if line.startswith('(') and line.endswith(')'):
                    line = line[1:-1]

                # Try to split with different delimiters (LLM may use different quote patterns)
                # First try with double quotes: "entity"",""name"",""type
                # Then try with single quotes: "entity","name","type
                record_attributes = split_string_by_multi_markers(line, ['"",""', '","'])
                logger.debug(f"Parsed attributes ({len(record_attributes)}): {record_attributes}")
                
                # Skip lines that produce very few fields (likely not data)
                if len(record_attributes) < 2:
                    logger.debug(f"Skipping line with insufficient parsed fields ({len(record_attributes)}): {line[:100]}")
                    continue
                
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
            logger.exception(f"Error extracting from chunk {chunk_key}")
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
            
            # Determine most common relationship type
            type_counts = {}
            for rel in rel_list:
                rel_type = rel.get("relationship_type")
                if rel_type:
                    type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "RELATED_TO"
            
            merged_relationships.append({
                "src_id": src,
                "tgt_id": tgt,
                "relationship_type": most_common_type,
                "weight": total_weight,
                "description": merged_description,
                "keywords": keywords,
                "source_id": source_ids
            })
        
        return merged_entities, merged_relationships
        
        
        
        
        
    @staticmethod
    async def extract_entities(text: str, llm: Any, entity_types: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
            Standalone function to extract entities and relationships from a text chunk.
            Compatible with PDFEntityExtractor.
        

            Args:
                text: Input text
                llm: LLM invoker
                entity_types: List of entity types
        
            Returns:
                Tuple of (entities_list, relationships_list)
            """
        
        extractor = GraphRAGExtractor(llm_invoker=llm, entity_types=entity_types)
        
        # We use a dummy ID as we are processing a single text block
        nodes, edges, _ = await extractor.extract_chunk("temp_id", text)

        # Convert nodes dict to list
        entities_list = []
        
        for entity_name, entity_list in nodes.items():
            if not entity_list:
                continue
        
            # Use the first occurrence for type and description
            first_entity = entity_list[0]
            entities_list.append({
                   "name": first_entity["entity_name"],
                    "type": first_entity["entity_type"],
                    "description": first_entity["description"]
                })
        
        # Convert edges dict to list
        relationships_list = []
        
        for (src, tgt), rel_list in edges.items():
        
            if not rel_list:
                continue
        
            first_rel = rel_list[0]
            relationships_list.append({
                "source": first_rel["src_id"],
                "target": first_rel["tgt_id"],
                "description": first_rel["description"],
                "strength": first_rel.get("weight", 1.0)
            })

        return entities_list, relationships_list
        
        