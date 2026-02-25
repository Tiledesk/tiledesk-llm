"""
Extraction prompts and configurations for different knowledge graph domains.

This module contains all graph extraction prompts and their associated configurations.
Each domain has its own set of entity types, relationship types, and extraction prompt.
"""

from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from ..config_loader import ConfigManager

# ============================================================================
# GENERIC DOMAIN (DEFAULT)
# ============================================================================

DEFAULT_ENTITY_TYPES_GENERIC = [
    # Core entities
    "PERSON", "ORGANIZATION", "LOCATION",
    # Content & Knowledge
    "DOCUMENT", "CONCEPT", "TECHNOLOGY", "PRODUCT", "SERVICE",
    # Temporal & Events
    "EVENT", "DATE", "TIMEFRAME",
    # Classification & Structure
    "CATEGORY", "TOPIC", "KEYWORD",
    # Measurements & Data
    "METRIC", "QUANTITY", "CURRENCY"
]

DEFAULT_RELATIONSHIP_TYPES_GENERIC = [
    # Organizational
    "WORKS_FOR", "EMPLOYED_BY", "MANAGES", "REPORTS_TO", "COLLABORATES_WITH",
    # Ownership & Control
    "OWNS", "CONTROLS", "OPERATES", "MAINTAINS",
    # Spatial
    "LOCATED_IN", "BASED_IN", "NEAR", "PART_OF",
    # Temporal
    "HAPPENED_ON", "OCCURRED_AT", "STARTED_ON", "ENDED_ON", "DURING",
    # Participation
    "PARTICIPATED_IN", "ATTENDED", "ORGANIZED", "SPONSORED",
    # Creation & Authorship
    "CREATED_BY", "AUTHORED_BY", "DEVELOPED_BY", "DESIGNED_BY",
    # Content & Reference
    "MENTIONS", "REFERENCES", "CITES", "DESCRIBES", "DISCUSSES",
    # Hierarchical
    "CONTAINS", "CONSISTS_OF", "BELONGS_TO", "MEMBER_OF",
    # Usage & Dependencies
    "USES", "REQUIRES", "DEPENDS_ON", "PROVIDES", "SUPPORTS",
    # Association & Similarity
    "ASSOCIATED_WITH", "SIMILAR_TO", "RELATED_TO", "CONNECTED_TO",
    # Action & Impact
    "IMPACTS", "AFFECTS", "INFLUENCES", "CAUSES", "RESULTS_IN"
]

GRAPH_EXTRACTION_PROMPT_GENERIC = """
-Goal-
Given a text document, identify all relevant entities and their relationships with SPECIFIC relationship types. Pay attention to temporal information (dates, timeframes) and organizational structures.

-Steps-
1. Identify all entities. For each identified entity, extract:
- entity_name: Name/identifier of the entity, capitalized, in language of 'Text'
- entity_type: One of: [{entity_types}]
  * Entity type guide:
    - "PERSON": Individuals, professionals, historical figures
    - "ORGANIZATION": Companies, institutions, agencies, groups
    - "LOCATION": Cities, countries, regions, addresses, venues
    - "DOCUMENT": Reports, articles, books, papers, contracts
    - "CONCEPT": Ideas, theories, methodologies, frameworks
    - "TECHNOLOGY": Software, hardware, platforms, tools, systems
    - "PRODUCT": Goods, merchandise, offerings
    - "SERVICE": Services, programs, initiatives
    - "EVENT": Meetings, conferences, projects, incidents, milestones
    - "DATE": Specific dates, years, periods
    - "CATEGORY": Classifications, types, genres
    - "METRIC": KPIs, measurements, statistics
- entity_description: Comprehensive description including attributes, context, and significance in language of 'Text'
Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. Identify relationships between entities with SPECIFIC relationship types:
- Use appropriate relationship types from this list:
  * WORKS_FOR, EMPLOYED_BY: employment relationships
  * MANAGES, REPORTS_TO: organizational hierarchy
  * OWNS, CONTROLS: ownership and control
  * LOCATED_IN, BASED_IN: spatial relationships
  * HAPPENED_ON, OCCURRED_AT, DURING: temporal relationships
  * PARTICIPATED_IN, ATTENDED, ORGANIZED: event participation
  * CREATED_BY, AUTHORED_BY, DEVELOPED_BY: creation and authorship
  * MENTIONS, REFERENCES, CITES, DESCRIBES: content relationships
  * CONTAINS, PART_OF, MEMBER_OF: hierarchical relationships
  * USES, REQUIRES, PROVIDES: usage and dependencies
  * SIMILAR_TO, RELATED_TO: associations (use only when no specific type fits)

3. Temporal Anchoring:
- Include relationship dates when mentioned (format: YYYY-MM-DD, YYYY-MM, or YYYY)
- For events, always try to extract temporal information

For each relationship extract ALL 7 fields (relationship_type is MANDATORY):
- source_entity: name from step 1
- target_entity: name from step 1
- relationship_type: ONE of the types above (UPPERCASE, use underscores) - REQUIRED
- relationship_description: explanation with context in language of 'Text'
- relationship_strength: numeric score (1-10) reflecting evidence strength
- relationship_date: date/timestamp if mentioned (format: YYYY-MM-DD or "YYYY" or "unknown")

CRITICAL: Every relationship MUST have exactly 7 fields. The 4th field MUST be a relationship_type from the list above.
Format: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>{tuple_delimiter}<relationship_date>

4. Return output as single list of entities and relationships using **{record_delimiter}** as delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1 (Business & Technology Context):

Entity_types: [person, organization, technology, product, event, location, date]
Text:
On March 15, 2024, Sarah Chen, CEO of TechVision Inc., announced the launch of CloudSync Pro at the annual DevCon conference in San Francisco. The new cloud storage platform, developed by the company's engineering team led by Marcus Rodriguez, integrates with existing enterprise systems. Microsoft Corporation expressed interest in partnering on the project. The conference, attended by over 5,000 developers, featured keynote speeches on AI and cloud computing trends.

################
Output:
("entity"{tuple_delimiter}"Sarah Chen"{tuple_delimiter}"person"{tuple_delimiter}"CEO of TechVision Inc., announced CloudSync Pro launch at DevCon 2024"){record_delimiter}
("entity"{tuple_delimiter}"TechVision Inc."{tuple_delimiter}"organization"{tuple_delimiter}"Technology company that developed and launched CloudSync Pro cloud storage platform"){record_delimiter}
("entity"{tuple_delimiter}"CloudSync Pro"{tuple_delimiter}"product"{tuple_delimiter}"New cloud storage platform that integrates with enterprise systems, launched March 2024"){record_delimiter}
("entity"{tuple_delimiter}"Marcus Rodriguez"{tuple_delimiter}"person"{tuple_delimiter}"Engineering team leader at TechVision Inc., led development of CloudSync Pro"){record_delimiter}
("entity"{tuple_delimiter}"Microsoft Corporation"{tuple_delimiter}"organization"{tuple_delimiter}"Large technology company interested in partnership on CloudSync Pro project"){record_delimiter}
("entity"{tuple_delimiter}"DevCon 2024"{tuple_delimiter}"event"{tuple_delimiter}"Annual developer conference held in San Francisco, attended by 5,000+ developers, featured AI and cloud computing topics"){record_delimiter}
("entity"{tuple_delimiter}"San Francisco"{tuple_delimiter}"location"{tuple_delimiter}"City where DevCon 2024 conference was held"){record_delimiter}
("entity"{tuple_delimiter}"Cloud Computing"{tuple_delimiter}"concept"{tuple_delimiter}"Technology trend discussed at DevCon, relevant to CloudSync Pro platform"){record_delimiter}
("relationship"{tuple_delimiter}"Sarah Chen"{tuple_delimiter}"TechVision Inc."{tuple_delimiter}"WORKS_FOR"{tuple_delimiter}"Sarah Chen is CEO of TechVision Inc."{tuple_delimiter}10{tuple_delimiter}2024-03-15){record_delimiter}
("relationship"{tuple_delimiter}"Marcus Rodriguez"{tuple_delimiter}"TechVision Inc."{tuple_delimiter}"EMPLOYED_BY"{tuple_delimiter}"Marcus Rodriguez leads engineering team at TechVision Inc."{tuple_delimiter}9{tuple_delimiter}2024){record_delimiter}
("relationship"{tuple_delimiter}"TechVision Inc."{tuple_delimiter}"CloudSync Pro"{tuple_delimiter}"DEVELOPED_BY"{tuple_delimiter}"TechVision Inc. developed CloudSync Pro platform"{tuple_delimiter}10{tuple_delimiter}2024-03-15){record_delimiter}
("relationship"{tuple_delimiter}"Marcus Rodriguez"{tuple_delimiter}"CloudSync Pro"{tuple_delimiter}"DEVELOPED_BY"{tuple_delimiter}"Marcus Rodriguez led engineering team that developed CloudSync Pro"{tuple_delimiter}9{tuple_delimiter}2024){record_delimiter}
("relationship"{tuple_delimiter}"Sarah Chen"{tuple_delimiter}"DevCon 2024"{tuple_delimiter}"PARTICIPATED_IN"{tuple_delimiter}"Sarah Chen announced product launch at DevCon conference"{tuple_delimiter}10{tuple_delimiter}2024-03-15){record_delimiter}
("relationship"{tuple_delimiter}"DevCon 2024"{tuple_delimiter}"San Francisco"{tuple_delimiter}"HAPPENED_ON"{tuple_delimiter}"DevCon 2024 conference was held in San Francisco"{tuple_delimiter}10{tuple_delimiter}2024-03-15){record_delimiter}
("relationship"{tuple_delimiter}"Microsoft Corporation"{tuple_delimiter}"CloudSync Pro"{tuple_delimiter}"INTERESTED_IN"{tuple_delimiter}"Microsoft expressed interest in partnering on CloudSync Pro project"{tuple_delimiter}7{tuple_delimiter}2024-03-15){record_delimiter}
("relationship"{tuple_delimiter}"CloudSync Pro"{tuple_delimiter}"Cloud Computing"{tuple_delimiter}"USES"{tuple_delimiter}"CloudSync Pro is a cloud storage platform based on cloud computing technology"{tuple_delimiter}9{tuple_delimiter}2024){record_delimiter}
("relationship"{tuple_delimiter}"DevCon 2024"{tuple_delimiter}"Cloud Computing"{tuple_delimiter}"DISCUSSES"{tuple_delimiter}"DevCon 2024 featured keynote speeches on cloud computing trends"{tuple_delimiter}8{tuple_delimiter}2024-03-15){completion_delimiter}

######################
Example 2 (Academic & Research Context):

Entity_types: [person, organization, document, concept, event, date]
Text:
Dr. Elena Petrova from MIT published a groundbreaking paper on quantum computing algorithms in Nature journal in January 2025. Her research, which builds on previous work by the Quantum Research Group, introduces a new framework called "Entangled State Optimization." The paper has been cited by researchers at Stanford University and Google AI.

################
Output:
("entity"{tuple_delimiter}"Dr. Elena Petrova"{tuple_delimiter}"person"{tuple_delimiter}"Researcher at MIT who published groundbreaking quantum computing paper in Nature, January 2025"){record_delimiter}
("entity"{tuple_delimiter}"MIT"{tuple_delimiter}"organization"{tuple_delimiter}"Massachusetts Institute of Technology, research institution where Dr. Petrova works"){record_delimiter}
("entity"{tuple_delimiter}"Quantum Computing Paper"{tuple_delimiter}"document"{tuple_delimiter}"Research paper on quantum computing algorithms published in Nature journal, January 2025"){record_delimiter}
("entity"{tuple_delimiter}"Nature Journal"{tuple_delimiter}"document"{tuple_delimiter}"Scientific journal that published Dr. Petrova's quantum computing research"){record_delimiter}
("entity"{tuple_delimiter}"Quantum Research Group"{tuple_delimiter}"organization"{tuple_delimiter}"Research group whose previous work was foundation for Dr. Petrova's research"){record_delimiter}
("entity"{tuple_delimiter}"Entangled State Optimization"{tuple_delimiter}"concept"{tuple_delimiter}"New framework for quantum computing introduced by Dr. Petrova in January 2025 paper"){record_delimiter}
("entity"{tuple_delimiter}"Stanford University"{tuple_delimiter}"organization"{tuple_delimiter}"Academic institution whose researchers cited Dr. Petrova's work"){record_delimiter}
("entity"{tuple_delimiter}"Google AI"{tuple_delimiter}"organization"{tuple_delimiter}"Research division that cited Dr. Petrova's quantum computing paper"){record_delimiter}
("relationship"{tuple_delimiter}"Dr. Elena Petrova"{tuple_delimiter}"MIT"{tuple_delimiter}"WORKS_FOR"{tuple_delimiter}"Dr. Petrova is affiliated with MIT research institution"{tuple_delimiter}10{tuple_delimiter}2025-01){record_delimiter}
("relationship"{tuple_delimiter}"Dr. Elena Petrova"{tuple_delimiter}"Quantum Computing Paper"{tuple_delimiter}"AUTHORED_BY"{tuple_delimiter}"Dr. Petrova authored the groundbreaking quantum computing paper"{tuple_delimiter}10{tuple_delimiter}2025-01){record_delimiter}
("relationship"{tuple_delimiter}"Quantum Computing Paper"{tuple_delimiter}"Nature Journal"{tuple_delimiter}"PUBLISHED_IN"{tuple_delimiter}"Paper was published in Nature journal in January 2025"{tuple_delimiter}10{tuple_delimiter}2025-01){record_delimiter}
("relationship"{tuple_delimiter}"Quantum Computing Paper"{tuple_delimiter}"Entangled State Optimization"{tuple_delimiter}"DESCRIBES"{tuple_delimiter}"Paper introduces and describes the Entangled State Optimization framework"{tuple_delimiter}10{tuple_delimiter}2025-01){record_delimiter}
("relationship"{tuple_delimiter}"Entangled State Optimization"{tuple_delimiter}"Quantum Research Group"{tuple_delimiter}"BUILDS_ON"{tuple_delimiter}"New framework builds on previous work by Quantum Research Group"{tuple_delimiter}8{tuple_delimiter}2025){record_delimiter}
("relationship"{tuple_delimiter}"Stanford University"{tuple_delimiter}"Quantum Computing Paper"{tuple_delimiter}"CITES"{tuple_delimiter}"Stanford researchers cited Dr. Petrova's paper"{tuple_delimiter}9{tuple_delimiter}2025){record_delimiter}
("relationship"{tuple_delimiter}"Google AI"{tuple_delimiter}"Quantum Computing Paper"{tuple_delimiter}"CITES"{tuple_delimiter}"Google AI research division cited the quantum computing paper"{tuple_delimiter}9{tuple_delimiter}2025){completion_delimiter}
######################

-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""



# Alias for backward compatibility
GRAPH_EXTRACTION_PROMPT = GRAPH_EXTRACTION_PROMPT_GENERIC


# ============================================================================
# CONFIGURATION REGISTRY
# ============================================================================

import logging
from pathlib import Path

# Import the Pydantic ExtractionConfig from models
from ..models.extraction_config import ExtractionConfig as PydanticExtractionConfig

# Export for backwards compatibility
ExtractionConfig = PydanticExtractionConfig

logger = logging.getLogger(__name__)

# Global config manager instance
_CONFIG_MANAGER = None


def _get_config_manager() -> ConfigManager:
    """Initialize and return the ConfigManager singleton."""
    global _CONFIG_MANAGER
    if _CONFIG_MANAGER is None:
        # Determine config directory relative to this file
        config_dir = Path(__file__).parent.parent / "config" / "extraction"
        _CONFIG_MANAGER = ConfigManager(str(config_dir))
        _CONFIG_MANAGER.load_all()
    return _CONFIG_MANAGER


def get_extraction_config(creation_prompt: Optional[str] = None) -> PydanticExtractionConfig:
    """
    Get the extraction configuration for a given domain.

    Args:
        creation_prompt: Domain identifier (e.g., "debt_recovery", "generic").
                        If None or not found, returns generic config.

    Returns:
        ExtractionConfig for the specified domain
    """
    manager = _get_config_manager()
    
    if creation_prompt is None:
        config = manager.get_config("generic")
        if config is None:
            raise ValueError("Generic configuration not found")
        return config

    # Normalize the prompt (lowercase, remove spaces)
    normalized = creation_prompt.lower().strip().replace(" ", "_")
    
    # Return config if found, otherwise generic
    config = manager.get_config(normalized)
    if config is None:
        logger.warning(f"Config '{normalized}' not found, falling back to generic")
        config = manager.get_config("generic")
        if config is None:
            raise ValueError("Generic configuration not found")
    
    return config


def list_available_domains() -> List[str]:
    """
    List all available domain configurations.

    Returns:
        List of domain identifiers
    """
    manager = _get_config_manager()
    configs = manager.list_configs()
    return list(configs.keys())
