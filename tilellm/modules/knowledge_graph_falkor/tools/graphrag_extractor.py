import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Import extraction configurations
from tilellm.modules.knowledge_graph_falkor.tools.extraction_prompts import (
    get_extraction_config,
    list_available_domains,
    ExtractionConfig
)

logger = logging.getLogger(__name__)

GRAPH_FIELD_SEP = "<SEP>"

# Legacy exports for backward compatibility
# These are kept for existing code that imports them directly
# New code should use get_extraction_config() instead
DEFAULT_ENTITY_TYPES_GENERIC = ["ORGANIZATION", "PERSON", "GEO", "EVENT", "CATEGORY"]
DEFAULT_RELATIONSHIP_TYPES_GENERIC = ["RELATED_TO"]
DEFAULT_ENTITY_TYPES = [
    "ORGANIZATION", "PERSON", "GEO",
    "LOAN", "MORTGAGE", "GUARANTEE", "PROTEST", "DEBT",
    "CONTRACT", "PAYMENT", "DEFAULT", "LEGAL_PROCEEDING", "ASSET",
    "WRIT_OF_EXECUTION",
    "INSOLVENCY_EVENT"
]
DEFAULT_RELATIONSHIP_TYPES = [
    "HAS_LOAN", "SECURED_BY", "GUARANTEES", "HAS_PAYMENT",
    "RECEIVED", "HAS_LEGAL_ACTION", "OWNS", "OBLIGATED_UNDER",
    "TRIGGERED", "RESULTED_IN", "CONCERNS", "RELATED_TO",
    "PRECEDES", "NOTIFIED_TO", "ISSUED_BY"
]


# NOTE: Extraction prompts have been moved to extraction_prompts.py
# The prompts below are kept for reference but are no longer used.
# Use get_extraction_config() to get the appropriate prompt for your domain.

# Kept for backward compatibility - will be removed in future version
_LEGACY_GRAPH_EXTRACTION_PROMPT_GENERIC = """
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

# Kept for backward compatibility - will be removed in future version
_LEGACY_GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document relevant to debt recovery and due diligence activities, identify all entities—including financial instruments and events—and their relationships with EXPLICIT relationship types. Pay special attention to temporal information (dates, sequences) to enable timeline reconstruction.

-Steps-
1. Identify all entities. For each identified entity, extract:
- entity_name: Name/identifier of the entity, capitalized, in language of 'Text'
- entity_type: One of: [{entity_types}]
  * Financial types guide:
    - "LOAN": credit facilities, financing agreements
    - "MORTGAGE": real estate collateral agreements
    - "GUARANTEE": suretyships, fideiussioni, personal/real guarantees
    - "PROTEST": unpaid checks/bills formally protested
    - "DEBT": outstanding obligations, arrears
    - "DEFAULT": missed payments, covenant breaches
    - "LEGAL_PROCEEDING": court cases, enforcement actions
    - "PAYMENT": transactions, settlements, installments
    - "ASSET": collateral, seized properties, valuables
  * For persons who are debtors, USE "PERSON" as entity_type but include "DEBTOR" in description
  * For organizations who are debtors, USE "ORGANIZATION" as entity_type but include "DEBTOR" in description
- entity_description: Comprehensive description including amounts, dates, parties, status, and key terms in language of 'Text'
Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. Identify relationships between entities with EXPLICIT relationship types:
- Use specific relationship types from this list:
  * HAS_LOAN: creditor/debtor has a loan (e.g., "Banca ABC HAS_LOAN LOAN-789", "Mario Rossi HAS_LOAN LOAN-789")
  * SECURED_BY: loan is secured by guarantee/collateral (e.g., "LOAN-789 SECURED_BY Mortgage-123")
  * GUARANTEES: guarantor guarantees a loan (e.g., "Ditta Rossi SRL GUARANTEES LOAN-789")
  * HAS_PAYMENT: loan/debt has payment (e.g., "LOAN-789 HAS_PAYMENT Payment-2024-01")
  * RECEIVED: person/org received communication (e.g., "Mario Rossi RECEIVED Letter-2024-03-15")
  * HAS_LEGAL_ACTION: loan has legal proceeding (e.g., "LOAN-789 HAS_LEGAL_ACTION Foreclosure-FC-001")
  * OWNS: person/org owns asset (e.g., "Mario Rossi OWNS Property-Via-Roma-10")
  * OBLIGATED_UNDER: person/org obligated under contract (e.g., "Mario Rossi OBLIGATED_UNDER LOAN-789")
  * TRIGGERED: event triggered another event (e.g., "Default-2024-11 TRIGGERED Foreclosure-FC-001")
  * RESULTED_IN: action resulted in outcome (e.g., "Payment-Plan RESULTED_IN Partial-Settlement")
  * CONCERNS: document/communication concerns entity (e.g., "Letter-001 CONCERNS LOAN-789")
  * RELATED_TO: generic relationship (use ONLY when no specific type applies)

3. Atomic Extraction: 
 - Do not group multiple events into a single entity. If there are three protests, create three separate "protest" entities. This is crucial for timeline accuracy.
 - For "protest" entities, use the amount and year in the entity name to ensure uniqueness (e.g., "Protest_800_Euro_2024").

4. Constraints:
 - GUARANTEES relationship MUST connect a Guarantor to a LOAN or DEBT, NEVER directly to the Person/Debtor.

5. Role Preservation: 
- For every "person" or "organization", explicitly state their role (DEBTOR, GUARANTOR, CO-OBLIGOR) at the beginning of the entity_description.

6. Temporal Anchoring: 
- Every relationship involving a "legal_proceeding", "payment", or "protest" MUST have a relationship_date. If a specific day is missing, use the month or year mentioned in the text

For each relationship extract ALL 7 fields (relationship_type is MANDATORY - never omit it):
- source_entity: name from step 1
- target_entity: name from step 1
- relationship_type: ONE of the types above (UPPERCASE, use underscores) - REQUIRED, NEVER use generic descriptions here
- relationship_description: explanation including temporal/causal context in language of 'Text'
- relationship_strength: numeric score (1-10) reflecting evidence strength
- relationship_date: approximate date/timestamp if mentioned (format: YYYY-MM-DD or "YYYY" or "unknown")

CRITICAL: Every relationship MUST have exactly 7 fields. The 4th field MUST be a relationship_type from the list above.
Format: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>{tuple_delimiter}<relationship_date>

7. Return output as single list of entities and relationships using **{record_delimiter}** as delimiter.

8. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1 (Debt Recovery Context):

Entity_types: [person, organization, loan, guarantee, default, legal_proceeding, payment]
Text:
On 2023-03-15, Banca ABC granted a €250,000 mortgage loan (ref: LOAN-789) to Marco Rossi. The loan was guaranteed by a suretyship from Ditta Rossi SRL. After three missed payments in Q4 2024, the bank declared default on 2024-11-30 and initiated foreclosure proceedings (case #FC-2024-88) on 2025-01-10. A partial payment of €15,000 was received on 2025-02-20.

################
Output:
("entity"{tuple_delimiter}"Banca ABC"{tuple_delimiter}"organization"{tuple_delimiter}"Italian bank acting as creditor in mortgage loan LOAN-789"){record_delimiter}
("entity"{tuple_delimiter}"Marco Rossi"{tuple_delimiter}"person"{tuple_delimiter}"DEBTOR - Borrower of €250,000 mortgage loan LOAN-789"){record_delimiter}
("entity"{tuple_delimiter}"Ditta Rossi SRL"{tuple_delimiter}"organization"{tuple_delimiter}"Company providing suretyship guarantee for Marco Rossi's loan"){record_delimiter}
("entity"{tuple_delimiter}"LOAN-789"{tuple_delimiter}"loan"{tuple_delimiter}"€250,000 mortgage loan originated 2023-03-15, secured by real estate, currently in default"){record_delimiter}
("entity"{tuple_delimiter}"Suretyship for LOAN-789"{tuple_delimiter}"guarantee"{tuple_delimiter}"Personal guarantee provided by Ditta Rossi SRL for full loan amount"){record_delimiter}
("entity"{tuple_delimiter}"Default on LOAN-789"{tuple_delimiter}"default"{tuple_delimiter}"Formal default declared 2024-11-30 after three consecutive missed payments in Q4 2024"){record_delimiter}
("entity"{tuple_delimiter}"Foreclosure FC-2024-88"{tuple_delimiter}"legal_proceeding"{tuple_delimiter}"Enforcement action initiated 2025-01-10 following loan default"){record_delimiter}
("entity"{tuple_delimiter}"Partial payment €15k"{tuple_delimiter}"payment"{tuple_delimiter}"Settlement payment of €15,000 received 2025-02-20 during foreclosure proceedings"){record_delimiter}
("relationship"{tuple_delimiter}"Banca ABC"{tuple_delimiter}"LOAN-789"{tuple_delimiter}"HAS_LOAN"{tuple_delimiter}"Banca ABC originated and owns mortgage loan LOAN-789"{tuple_delimiter}9{tuple_delimiter}2023-03-15){record_delimiter}
("relationship"{tuple_delimiter}"Marco Rossi"{tuple_delimiter}"LOAN-789"{tuple_delimiter}"OBLIGATED_UNDER"{tuple_delimiter}"Marco Rossi is primary borrower obligated under loan LOAN-789"{tuple_delimiter}10{tuple_delimiter}2023-03-15){record_delimiter}
("relationship"{tuple_delimiter}"Ditta Rossi SRL"{tuple_delimiter}"LOAN-789"{tuple_delimiter}"GUARANTEES"{tuple_delimiter}"Ditta Rossi SRL executed suretyship guarantee for LOAN-789"{tuple_delimiter}9{tuple_delimiter}2023-03-15){record_delimiter}
("relationship"{tuple_delimiter}"LOAN-789"{tuple_delimiter}"Suretyship for LOAN-789"{tuple_delimiter}"SECURED_BY"{tuple_delimiter}"Loan LOAN-789 is secured by suretyship guarantee"{tuple_delimiter}10{tuple_delimiter}2023-03-15){record_delimiter}
("relationship"{tuple_delimiter}"LOAN-789"{tuple_delimiter}"Default on LOAN-789"{tuple_delimiter}"RESULTED_IN"{tuple_delimiter}"Loan LOAN-789 entered formal default status after payment failures"{tuple_delimiter}10{tuple_delimiter}2024-11-30){record_delimiter}
("relationship"{tuple_delimiter}"Default on LOAN-789"{tuple_delimiter}"Foreclosure FC-2024-88"{tuple_delimiter}"TRIGGERED"{tuple_delimiter}"Default directly triggered initiation of foreclosure proceedings"{tuple_delimiter}10{tuple_delimiter}2025-01-10){record_delimiter}
("relationship"{tuple_delimiter}"LOAN-789"{tuple_delimiter}"Foreclosure FC-2024-88"{tuple_delimiter}"HAS_LEGAL_ACTION"{tuple_delimiter}"Loan has active foreclosure proceedings"{tuple_delimiter}10{tuple_delimiter}2025-01-10){record_delimiter}
("relationship"{tuple_delimiter}"LOAN-789"{tuple_delimiter}"Partial payment €15k"{tuple_delimiter}"HAS_PAYMENT"{tuple_delimiter}"Partial payment received during active foreclosure proceedings"{tuple_delimiter}7{tuple_delimiter}2025-02-20){completion_delimiter}
######################

-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


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
        
        