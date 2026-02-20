"""
Prompt templates for Knowledge Graph operations.
Uses LangChain PromptTemplate for consistent formatting.
"""
from typing import Optional

try:
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback if langchain not available
    class PromptTemplate:
        def __init__(self, input_variables, template):
            self.input_variables = input_variables
            self.template = template
        
        def format(self, **kwargs):
            result = self.template
            for key, value in kwargs.items():
                placeholder = f"{{{key}}}"
                result = result.replace(placeholder, str(value))
            return result
    
    LANGCHAIN_AVAILABLE = False

# ==================== GRAPH QA PROMPTS ====================

GRAPH_QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on knowledge graph information.
Use the provided knowledge graph context to answer the question accurately and concisely.
If the information is insufficient, say so clearly.
"""

GRAPH_QA_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question", "context", "chat_history"],
    template="""Based on the following knowledge graph information, answer the question.

{context}

Question: {question}

Chat History (for context):
{chat_history}

Provide a concise and accurate answer based on the knowledge graph. If the information is insufficient, say so."""
)

# ==================== ADVANCED GRAPH QA PROMPTS ====================

ADVANCED_GRAPH_QA_SYSTEM_PROMPT = """You are an advanced assistant that answers questions using multiple sources:
1. Global community reports (high-level summaries)
2. Document excerpts from vector search
3. Knowledge graph entities and relationships
4. Expanded graph context

Synthesize information from all relevant sources to provide comprehensive answers.
If information is contradictory, note the uncertainty.
If insufficient information exists, state what you can answer and what's missing.
"""

ADVANCED_GRAPH_QA_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question", "community_reports", "document_excerpts", "graph_context", "chat_history"],
    template="""Answer the question using the following multi-source information:

{community_reports}

{document_excerpts}

{graph_context}

Question: {question}

Chat History (for context):
{chat_history}

Provide a comprehensive answer that synthesizes information from all relevant sources. 
Acknowledge uncertainty or contradictions when they exist."""
)

# ==================== CONTEXT FORMATTING FUNCTIONS ====================

def format_community_reports(reports: list) -> str:
    """Format community reports for prompt context."""
    if not reports:
        return ""
    
    formatted = "## Global Context (Community Reports):\n"
    for i, report in enumerate(reports, 1):
        content = report.properties.get("report") or report.properties.get("summary") or report.properties.get("title", "")
        formatted += f"{i}. {content[:500]}...\n"
    return formatted

def format_document_excerpts(chunk_texts: list) -> str:
    """Format document excerpts for prompt context."""
    if not chunk_texts:
        return ""
    
    formatted = "## Relevant Document Excerpts:\n"
    for i, text in enumerate(chunk_texts[:5], 1):
        formatted += f"{i}. {text[:300]}...\n"
    return formatted

def format_graph_context(nodes: list, relationships: list) -> str:
    """Format graph nodes and relationships for prompt context."""
    if not nodes and not relationships:
        return ""
    
    formatted = "## Knowledge Graph Context:\n"
    
    # Group nodes by label
    if nodes:
        nodes_by_label = {}
        for node in nodes:
            label = node.get("label", "Unknown")
            name = node.get("properties", {}).get("name", node.get("properties", {}).get("title", "Unnamed"))
            nodes_by_label.setdefault(label, []).append(name)
        
        for label, names in nodes_by_label.items():
            formatted += f"### {label}:\n"
            for name in names[:5]:
                formatted += f"- {name}\n"
    
    # Add relationships
    if relationships:
        formatted += "\n## Relationships:\n"
        for rel in relationships[:5]:
            rel_type = rel.get("type", "RELATED")
            source_id = rel.get("source_id", "")
            target_id = rel.get("target_id", "")
            
            # Find node names (simplified - assumes nodes list contains these)
            source_node = next((n for n in nodes if n.get("id") == source_id), None)
            target_node = next((n for n in nodes if n.get("id") == target_id), None)
            source_name = source_node.get("properties", {}).get("name", "Unknown") if source_node else "Unknown"
            target_name = target_node.get("properties", {}).get("name", "Unknown") if target_node else "Unknown"
            
            formatted += f"- {source_name} --[{rel_type}]--> {target_name}\n"
    
    return formatted

def format_chat_history(chat_history_dict: dict, max_messages: Optional[int] = None) -> str:
    """Format chat history for prompt context with optional turn limit."""
    if not chat_history_dict or not isinstance(chat_history_dict, dict):
        return "No chat history available."
    
    try:
        # Sort keys as integers if possible
        sorted_keys = sorted(chat_history_dict.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    except Exception:
        sorted_keys = sorted(chat_history_dict.keys())
        
    # Apply max_messages limit (turns)
    if max_messages and max_messages > 0:
        sorted_keys = sorted_keys[-max_messages:]
        
    history_lines = []
    for key in sorted_keys:
        entry = chat_history_dict[key]
        
        q = None
        a = None
        
        # Handle dict-like entry
        if isinstance(entry, dict):
            q = entry.get("question")
            a = entry.get("answer")
        # Handle object-like entry (e.g. ChatEntry Pydantic model)
        elif hasattr(entry, "question") and hasattr(entry, "answer"):
            q = entry.question
            a = entry.answer
            
        if q is not None and a is not None:
            # Handle potential multimodal question (list of contents)
            if isinstance(q, list):
                from tilellm.models.schemas.multimodal_content import TextContent
                q_text = next((c.text for c in q if isinstance(c, TextContent)), str(q))
            else:
                q_text = str(q)
                
            history_lines.append(f"User: {q_text}")
            history_lines.append(f"Assistant: {a}")
        else:
            # Fallback for unknown structure
            history_lines.append(f"{key}: {entry}")
    
    return "\n".join(history_lines) if history_lines else "No chat history available."


# ==================== COMMUNITY REPORT PROMPT ====================

COMMUNITY_REPORT_PROMPT = """
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format(in language of 'Text' content):
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


# Example Input
-----------
Text:

-Entities-

id,entity,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

-Relationships-

id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March

Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes. [Data: Entities (5), Relationships (37, 38, 39, 40, 41,+more)]"
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community. [Data: Entities(6), Relationships (38, 43)]"
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community. [Data: Relationships (39)]"
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved. [Data: Relationships (40)]"
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:

-Entities-
{entity_df}

-Relationships-
{relation_df}

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format(in language of 'Text' content):
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Output:"""