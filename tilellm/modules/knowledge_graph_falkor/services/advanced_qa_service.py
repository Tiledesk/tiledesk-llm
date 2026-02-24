"""
Advanced QA Service for Domain-Specific Graph Queries.
Specialized for debt collection (recupero crediti) use cases.

Pipeline:
1. Intent Classifier → identifies query type (timeline, exposure, guarantees, etc.)
2. Parameter Extractor → extracts entities (debtor_name, date_range, loan_id)
3. Template Cypher Engine → generates safe parameterized queries
4. Graph Execution → runs queries on FalkorDB
5. LLM Response Enricher → transforms raw results into readable response
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from langchain_core.documents import Document
from ..utils.advanced_qa_prompt import INTENT_PROMPTS

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classifies user query intent for debt collection domain."""

    # Intent templates with keywords
    INTENT_PATTERNS = {
        "timeline": ["cronologia", "sequenza", "eventi", "storia", "timeline", "when", "quando", "completa", "tutti", "intero", "full", "complete"],
        "exposure": ["esposizione", "debito", "importo", "quanto", "how much", "exposure", "amount"],
        "guarantees": ["garanzie", "collateral", "security", "guarantee", "ipoteca", "pegno"],
        "payments": ["pagamenti", "versamenti", "payments", "transactions", "rata", "rate"],
        "contacts": ["contatti", "comunicazioni", "lettere", "email", "pec", "contacts"],
        "legal_actions": ["legale", "precetto", "ingiunzione", "tribunale", "legal", "court"],
        "debtor_info": ["debitore", "anagrafica", "chi è", "who is", "debtor", "informazioni"],
        "relationship": ["relazione", "collegamento", "legame", "relationship", "connected", "related", "garanti", "guarantors", "fideiussori", "garante"],
        "vespro": ["vespri", "vespro", "siciliani", "ferdinando", "aragona", "1282", "rivolta"]
    }

    def __init__(self, llm=None):
        self.llm = llm

    async def classify_intent(self, question: str) -> Tuple[str, float]:
        """
        Classify the intent of the user's question.

        Returns:
            Tuple of (intent_name, confidence_score)
        """
        question_lower = question.lower()

        # Rule-based classification
        intent_scores = {}
        for intent, keywords in self.INTENT_PATTERNS.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                intent_scores[intent] = score

        if intent_scores:
            # Get highest scoring intent
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[best_intent] / 3.0, 1.0)  # Normalize to 0-1
            return best_intent, confidence

        # Fallback: use LLM if available
        if self.llm:
            try:
                return await self._llm_classify(question)
            except Exception as e:
                logger.warning(f"LLM intent classification failed: {e}")

        # Default fallback
        return "general", 0.5

    async def _llm_classify(self, question: str) -> Tuple[str, float]:
        """Use LLM for intent classification."""
        prompt = f"""Classify the intent of this debt collection query into ONE of these categories:
- timeline: asking about sequence of events
- exposure: asking about debt amount
- guarantees: asking about collaterals/securities
- payments: asking about payment history
- contacts: asking about communications
- legal_actions: asking about legal proceedings
- debtor_info: asking about debtor information
- relationship: asking about relationships between entities
- general: general question

Question: "{question}"

Answer with just the category name and confidence (0-1):
Format: category|confidence"""

        if hasattr(self.llm, 'ainvoke'):
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
        else:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

        # Parse response
        parts = content.strip().split('|')
        intent = parts[0].strip() if len(parts) > 0 else "general"
        confidence = float(parts[1].strip()) if len(parts) > 1 else 0.7

        return intent, confidence


class ParameterExtractor:
    """Extracts structured parameters from natural language queries."""

    def __init__(self, llm=None):
        self.llm = llm

    async def extract_parameters(self, question: str, intent: str, chat_history_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract relevant parameters based on intent, considering chat history.

        Returns:
            Dictionary with extracted parameters
        """
        params = {}

        # Extract common entities using regex patterns
        # (omissis for brevity in thought, but keeping logic)
        
        # Debtor name (capitalized words, possibly with articles)
        debtor_pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b'
        debtor_matches = re.findall(debtor_pattern, question)
        
        # Italian stopwords to exclude from debtor name extraction
        stopwords = {
            'mostrami', 'dimmi', 'mostra', 'visualizza', 'elenca', 'lista',
            'quali', 'quante', 'quanti', 'quanto', 'qual', 'quando',
            'chi', 'cosa', 'come', 'dove', 'perché', 'perche',
            'se', 'con', 'senza', 'tra', 'fra', 'sul', 'sulla', 'sullo',
            'nel', 'nella', 'nello', 'dal', 'dalla', 'dallo',
            'al', 'alla', 'allo', 'del', 'della', 'dello',
            'un', 'una', 'uno', 'il', 'lo', 'la', 'i', 'gli', 'le',
            'di', 'a', 'da', 'in', 'su', 'per', 'con', 'tra', 'fra'
        }
        
        if debtor_matches:
            # Filter out stopwords and choose the longest match (likely the actual name)
            valid_matches = [match for match in debtor_matches 
                            if match.lower() not in stopwords and len(match) > 2]
            if valid_matches:
                # Prefer matches that appear after "di" or "per" if present
                question_lower = question.lower()
                for match in valid_matches:
                    match_lower = match.lower()
                    idx = question_lower.find(match_lower)
                    # Check if preceded by "di " or "per "
                    if idx > 2:
                        preceding = question_lower[max(0, idx-3):idx]
                        if preceding in [' di ', ' per ']:
                            params["debtor_name"] = match
                            break
                else:
                    # Otherwise take the longest valid match
                    params["debtor_name"] = max(valid_matches, key=len)
            else:
                # Fallback to first match if all are stopwords
                params["debtor_name"] = debtor_matches[0]

        # Loan ID / Practice ID
        loan_id_pattern = r'\b(?:pratica|loan|id|contratto)[:\s]+([A-Z0-9\-]+)\b'
        loan_match = re.search(loan_id_pattern, question, re.IGNORECASE)
        if loan_match:
            params["loan_id"] = loan_match.group(1)

        # Detect if it's a request for ALL/FULL results
        full_request_keywords = ["completa", "tutti", "tutto", "tutta", "intero", "interi", "full", "complete", "all",
                                 "mostra tutto", "mostra completa", "mostra tutti", "mostra tutta", "mostra intero",
                                 "visualizza tutto", "visualizza completa", "visualizza tutti", "visualizza tutta",
                                 "elenca tutto", "elenca completa", "elenca tutti", "elenca tutta"]
        params["is_full_request"] = any(kw in question.lower() for kw in full_request_keywords)
        
        # Detect relationship type if mentioned in question
        relationship_keywords = {
            "GUARANTEES": ["garanti", "guarantors", "fideiussori", "garante", "garanzie personali"],
            "HAS_LOAN": ["prestiti", "finanziamenti", "mutui", "loan", "debito"],
            "SECURED_BY": ["garanzie reali", "collateral", "ipoteca", "pegno", "garanzia reale"],
            "HAS_PAYMENT": ["pagamenti", "rate", "versamenti", "pagato"],
            "RECEIVED": ["comunicazioni", "lettere", "email", "pec", "contatti"],
            "HAS_LEGAL_ACTION": ["azioni legali", "procedimenti", "tribunale", "legale"]
        }
        
        question_lower = question.lower()
        for rel_type, keywords in relationship_keywords.items():
            if any(kw in question_lower for kw in keywords):
                params["relationship_type"] = rel_type
                break

        # Use LLM for more sophisticated extraction if available
        if self.llm:
            try:
                llm_params = await self._llm_extract(question, intent, chat_history_dict)
                # Merge: LLM takes precedence for sophisticated resolution
                params.update(llm_params)
            except Exception as e:
                logger.warning(f"LLM parameter extraction failed: {e}")

        # If LLM didn't catch is_full_request but rule-based did, keep it
        if "is_full_request" not in params:
             params["is_full_request"] = any(kw in question.lower() for kw in full_request_keywords)

        return params

    async def _llm_extract(self, question: str, intent: str, chat_history_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to extract parameters, optionally using chat history."""
        history_text = ""
        if chat_history_dict:
            from ..utils import format_chat_history
            history_text = format_chat_history(chat_history_dict)

        prompt = f"""Extract structured information from this debt collection query.
Intent: {intent}
Question: "{question}"

CONVERSATION HISTORY:
{history_text}

If the question uses pronouns or refers to entities mentioned in the history (e.g., "him", "that debtor", "the loan"), resolve them using the conversation history.

Extract the following if present:
- debtor_name: name of the debtor (resolve from history if necessary)
- loan_id: loan/practice identifier
- date_range: date or date range
- amount: monetary amount
- is_full_request: true if user is asking for the complete/full timeline or ALL events without summary

Return as JSON format. If not present, omit the field.
Example: {{"debtor_name": "Mario Rossi", "loan_id": "PR-2024-001", "is_full_request": true}}
"""

        if hasattr(self.llm, 'ainvoke'):
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
        else:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

        # Try to parse JSON
        try:
            import json
            # Extract JSON from response (might have extra text)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")

        return {}


class CypherTemplateEngine:
    """Generates safe Cypher queries using templates and parameter binding."""

    # Cypher query templates for different intents
    # NOTE: Using flexible node matching (person/organization) instead of DEBTOR label
    QUERY_TEMPLATES = {
        "timeline": """
            // Find the debtor and their relationships in a single path match
            MATCH (debtor)-[r]-(connected)
            WHERE debtor.namespace = $namespace 
            AND connected.namespace = $namespace
            AND (
                toUpper(COALESCE(debtor.name, '')) CONTAINS $debtor_name OR 
                toUpper(COALESCE(debtor.entity_name, '')) CONTAINS $debtor_name
            )
            RETURN DISTINCT
                COALESCE(debtor.name, debtor.entity_name) as debtor_name,
                COALESCE(connected.name, connected.entity_name) as connected_entity_name,
                labels(connected)[0] as connected_entity_type,
                connected.description as connected_description,
                type(r) as relationship_type,
                r.description as relationship_description,
                // Robust date extraction
                COALESCE(
                    connected.event_date,
                    connected.date, 
                    connected.timestamp, 
                    connected.creation_date,
                    r.date, 
                    connected.import_timestamp, 
                    r.import_timestamp
                ) as date_reference,
                connected.import_timestamp as entity_import_timestamp,
                r.import_timestamp as relationship_import_timestamp
            ORDER BY 
                CASE 
                    WHEN date_reference IS NOT NULL THEN 0 
                    ELSE 1 
                END,
                date_reference ASC
            LIMIT 200
        """,

        "timeline_old": """
                // First find the debtor node(s)
                MATCH (debtor)
                WHERE debtor.namespace = $namespace 
                AND (COALESCE(toUpper(debtor.name), '') CONTAINS $debtor_name OR COALESCE(debtor.entity_name, '') CONTAINS $debtor_name)
                WITH debtor
                LIMIT 5  // Limit number of debtor nodes

                // Find all relationships from debtor to other nodes (events/entities)
                MATCH (debtor)-[r]-(connected)
                WHERE connected.namespace = $namespace
                WITH debtor, connected, r
                LIMIT 200  // Limit total relationships to avoid timeout

                // Return rich contextual information for LLM to construct timeline
                // LLM will extract dates from text descriptions if present
                RETURN DISTINCT
                    COALESCE(debtor.name, debtor.entity_name) as debtor_name,
                    COALESCE(connected.name, connected.entity_name) as connected_entity_name,
                    labels(connected)[0] as connected_entity_type,
                    connected.description as connected_description,
                    type(r) as relationship_type,
                    r.description as relationship_description,
                    // Use import_timestamp as fallback date reference
                    COALESCE(connected.import_timestamp, r.import_timestamp) as date_reference,
                    connected.import_timestamp as entity_import_timestamp,
                    r.import_timestamp as relationship_import_timestamp
                ORDER BY 
                    CASE 
                        WHEN date_reference IS NOT NULL THEN 0 
                        ELSE 1 
                    END,
                    date_reference ASC
                LIMIT 100
            """,

        "exposure": """
            MATCH (debtor)-[r:HAS_LOAN|OBLIGATED_UNDER]-(loan)
            WHERE (debtor:PERSON OR debtor:ORGANIZATION)
            AND (toUpper(COALESCE(debtor.name, '')) CONTAINS $debtor_name OR toUpper(COALESCE(debtor.entity_name, '')) CONTAINS $debtor_name)
            AND (loan:LOAN)
            RETURN COALESCE(debtor.name, debtor.entity_name) as debtor,
                   COALESCE(loan.id, loan.entity_name, loan.name) as loan_id,
                   loan.principal_amount as principal,
                   loan.interest as interest,
                   loan.total_exposure as total,
                   loan.description as details
            ORDER BY total DESC
            LIMIT 200
        """,

        "guarantees": """
            MATCH (debtor)-[:HAS_LOAN|OBLIGATED_UNDER]-(loan)-[:SECURED_BY]-(guarantee)
            WHERE (debtor:PERSON OR debtor:ORGANIZATION)
            AND (toUpper(COALESCE(debtor.name, '')) CONTAINS $debtor_name OR toUpper(COALESCE(debtor.entity_name, '')) CONTAINS $debtor_name)
            AND (loan:LOAN)
            RETURN COALESCE(debtor.name, debtor.entity_name) as debtor,
                   COALESCE(loan.id, loan.entity_name) as loan_id,
                   COALESCE(guarantee.type, guarantee.entity_type) as guarantee_type,
                   guarantee.value as value,
                   guarantee.status as status,
                   guarantee.description as details
            LIMIT 200
        """,

        "payments": """
            MATCH (debtor)-[:HAS_LOAN|OBLIGATED_UNDER]-(loan)-[:HAS_PAYMENT]-(payment)
            WHERE (debtor:PERSON OR debtor:ORGANIZATION)
            AND (toUpper(COALESCE(debtor.name, '')) CONTAINS $debtor_name OR toUpper(COALESCE(debtor.entity_name, '')) CONTAINS $debtor_name)
            AND (loan:LOAN)
            AND (payment:PAYMENT)
            RETURN COALESCE(debtor.name, debtor.entity_name) as debtor,
                   COALESCE(loan.id, loan.entity_name) as loan_id,
                   COALESCE(payment.date, payment.timestamp) as date,
                   payment.amount as amount,
                   COALESCE(payment.type, payment.entity_type) as type,
                   payment.description as details
            ORDER BY date DESC
            LIMIT 300
        """,

        "contacts": """
            MATCH (debtor)-[:RECEIVED|CONCERNS]-(contact)
            WHERE (debtor:PERSON OR debtor:ORGANIZATION)
            AND (toUpper(COALESCE(debtor.name, '')) CONTAINS $debtor_name OR toUpper(COALESCE(debtor.entity_name, '')) CONTAINS $debtor_name)
            RETURN COALESCE(debtor.name, debtor.entity_name) as debtor,
                   COALESCE(contact.date, contact.timestamp) as date,
                   COALESCE(contact.type, contact.entity_type) as type,
                   contact.subject as subject,
                   contact.status as status,
                   contact.description as details
            ORDER BY date DESC
            LIMIT 20
        """,

        "legal_actions": """
            MATCH (debtor)-[:HAS_LOAN|OBLIGATED_UNDER]-(loan)-[:HAS_LEGAL_ACTION|TRIGGERED]-(action)
            WHERE (debtor:PERSON OR debtor:ORGANIZATION)
            AND (toUpper(COALESCE(debtor.name, '')) CONTAINS $debtor_name OR toUpper(COALESCE(debtor.entity_name, '')) CONTAINS $debtor_name)
            AND (loan:LOAN)
            AND (action:LEGAL_PROCEEDING OR action.entity_type = 'LEGAL_PROCEEDING')
            RETURN COALESCE(debtor.name, debtor.entity_name) as debtor,
                   COALESCE(loan.id, loan.entity_name) as loan_id,
                   COALESCE(action.type, action.entity_type) as action_type,
                   COALESCE(action.date, action.timestamp) as date,
                   action.court as court,
                   action.status as status,
                   action.description as details
            ORDER BY date DESC
            LIMIT 200
        """,

        "debtor_info": """
            MATCH (debtor)
            WHERE (debtor:PERSON OR debtor:ORGANIZATION)
            AND (toUpper(COALESCE(debtor.name, '')) CONTAINS $debtor_name OR toUpper(COALESCE(debtor.entity_name, '')) CONTAINS $debtor_name)
            OPTIONAL MATCH (debtor)-[r]-(connected)
            RETURN debtor,
                   type(r) as relationship_type,
                   labels(connected) as connected_labels,
                   COALESCE(connected.name, connected.entity_name) as connected_name,
                   connected.description as connected_description
            LIMIT 20
        """,

        "relationship": """
            MATCH path = (start)-[*1..3]-(end)
            WHERE (toUpper(start.name) CONTAINS $entity1 OR start.entity_name CONTAINS $entity1)
            AND (toUpper(end.name) CONTAINS $entity2 OR end.entity_name CONTAINS $entity2)
            WITH path, length(path) as path_length
            ORDER BY path_length ASC
            LIMIT 10
            RETURN path, path_length,
                   [node IN nodes(path) | COALESCE(node.name, node.entity_name)] as node_names,
                   [rel IN relationships(path) | type(rel)] as relationship_types
        """,
        
        "relationship_by_type": """
            // Find all nodes connected to debtor via specific relationship type(s)
            MATCH (debtor)-[r]-(connected)
            WHERE (debtor:PERSON OR debtor:ORGANIZATION)
            AND (toUpper(COALESCE(debtor.name, '')) CONTAINS $debtor_name OR toUpper(COALESCE(debtor.entity_name, '')) CONTAINS $debtor_name)
            AND ($relationship_type IS NULL OR type(r) = $relationship_type)
            RETURN DISTINCT
                COALESCE(debtor.name, debtor.entity_name) as debtor_name,
                COALESCE(connected.name, connected.entity_name) as connected_entity_name,
                labels(connected)[0] as connected_entity_type,
                type(r) as relationship_type,
                connected.description as connected_description,
                r.description as relationship_description,
                COALESCE(connected.import_timestamp, r.import_timestamp) as date_reference
            ORDER BY date_reference DESC
            LIMIT 200
        """,
        
        "guarantors": """
            // Find all guarantors for a debtor (through GUARANTEES relationship to loans)
            MATCH (debtor)-[:HAS_LOAN|OBLIGATED_UNDER]-(loan)<-[:GUARANTEES]-(guarantor)
            WHERE (debtor:PERSON OR debtor:ORGANIZATION)
            AND (guarantor:PERSON OR guarantor:ORGANIZATION)
            AND (toUpper(COALESCE(debtor.name, '')) CONTAINS $debtor_name OR toUpper(COALESCE(debtor.entity_name, '')) CONTAINS $debtor_name)
            RETURN DISTINCT
                COALESCE(debtor.name, debtor.entity_name) as debtor_name,
                COALESCE(guarantor.name, guarantor.entity_name) as guarantor_name,
                labels(guarantor)[0] as guarantor_type,
                COLLECT(DISTINCT COALESCE(loan.id, loan.entity_name, loan.name)) as loan_ids,
                guarantor.description as guarantor_description
            LIMIT 200
        """,

        "vespro": """
            // Query per trovare relazioni tra Ferdinando III d'Aragona e protagonisti dei Vespri Siciliani
            MATCH (ferdinando:PERSON)-[*1..3]-(protagonista)
            WHERE ferdinando.namespace = $namespace
            AND protagonista.namespace = $namespace
            AND (
                toUpper(ferdinando.name) CONTAINS 'FERDINANDO'
                OR toUpper(ferdinando.name) CONTAINS 'ARAGONA'
            )
            WITH ferdinando, protagonista
            WHERE (
                toUpper(COALESCE(protagonista.name, '')) CONTAINS 'VESPRI'
                OR toUpper(COALESCE(protagonista.name, '')) CONTAINS 'SICILIA'
                OR toUpper(COALESCE(protagonista.description, '')) CONTAINS 'VESPRI'
                OR toUpper(COALESCE(protagonista.description, '')) CONTAINS 'SICILIANI'
            )
            RETURN DISTINCT
                ferdinando.name as ferdinando_name,
                protagonista.name as protagonista_name,
                labels(protagonista)[0] as protagonista_type,
                COALESCE(protagonista.description, '') as protagonista_description
            LIMIT 50
        """

    }



    def generate_query(self, intent: str, parameters: Dict[str, Any], namespace: str, index_name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a safe Cypher query with parameter binding.

        Returns:
            Tuple of (cypher_query, parameters_dict)
        """
        # Get base template based on intent and parameters
        template = None
        
        # Special handling for relationship queries with specific relationship type
        if intent == "relationship":
            relationship_type = parameters.get("relationship_type")
            if relationship_type == "GUARANTEES":
                template = self.QUERY_TEMPLATES.get("guarantors")
            elif relationship_type:
                template = self.QUERY_TEMPLATES.get("relationship_by_type")
        
        # Fallback to standard template for the intent
        if not template:
            template = self.QUERY_TEMPLATES.get(intent)
            
        if not template:
            # Fallback to general query — build WHERE dynamically to avoid FalkorDB null-parameter issues
            fallback_conditions = []
            if namespace:
                fallback_conditions.append("debtor.namespace = $namespace")
            if index_name:
                fallback_conditions.append("debtor.index_name = $index_name")
            fallback_where = ("WHERE " + " AND ".join(fallback_conditions)) if fallback_conditions else ""
            template = f"""
                MATCH (debtor)
                {fallback_where}
                RETURN debtor
                LIMIT 20
            """

        # Add namespace filtering to template if not present
        if "namespace" not in template and namespace:
            # Insert WHERE clause if not present
            if "WHERE" not in template.upper():
                template = template.replace("RETURN", "WHERE debtor.namespace = $namespace RETURN")
            else:
                template = template.replace("WHERE", f"WHERE debtor.namespace = $namespace AND ")

        # Prepare parameters for binding — only include non-None values
        query_params = {}
        if namespace is not None:
            query_params["namespace"] = namespace
        if index_name is not None:
            query_params["index_name"] = index_name

        # Map extracted parameters to query parameters
        query_params["debtor_name"] = parameters.get("debtor_name", "").upper()

        if "loan_id" in parameters:
            query_params["loan_id"] = parameters["loan_id"]

        if "entity1" in parameters:
            query_params["entity1"] = parameters["entity1"].upper()

        if "entity2" in parameters:
            query_params["entity2"] = parameters["entity2"].upper()
            
        if "relationship_type" in parameters:
            query_params["relationship_type"] = parameters["relationship_type"]

        return template.strip(), query_params


class AdvancedQAService:
    """
    Advanced QA Service for domain-specific graph queries.
    Orchestrates the full pipeline from intent classification to response generation.
    """

    def __init__(self,
                 graph_service,
                 graph_rag_service,
                 community_graph_service,
                 llm=None):
        self.graph_service = graph_service
        self.graph_rag_service = graph_rag_service
        self.community_graph_service = community_graph_service
        self.llm = llm

        # Initialize components
        self.intent_classifier = IntentClassifier(llm=llm)
        self.parameter_extractor = ParameterExtractor(llm=llm)
        self.cypher_engine = CypherTemplateEngine()
        
        # High-volume handling threshold
        self.high_volume_threshold = 80

    async def process_query(
        self,
        question: str,
        namespace: str,
        engine,
        vector_store_repo,
        llm_embeddings,
        search_type: str = "hybrid",
        sparse_encoder=None,
        chat_history_dict: Optional[Dict[str, Any]] = None,
        max_community_reports: int = 3,
        top_k: int = 10,
        # New Hybrid History Flags
        contextualize_prompt: bool = False,
        include_history_in_prompt: bool = True,
        max_history_messages: int = 10,
        conversation_summary: bool = False,
        graph_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an advanced QA query using the full pipeline.

        Pipeline:
        1. Hybrid History Prep & Query Contextualization
        2. Classify intent
        3. Extract parameters
        4. Find seed nodes from vector store
        5. Get relevant community reports (max 3)
        6. Generate and execute Cypher queries
        7. Enrich response with LLM

        Args:
            question: User's natural language question
            namespace: Graph namespace
            engine: Engine configuration
            vector_store_repo: Vector store repository
            llm_embeddings: Embedding model
            search_type: Search type (hybrid/semantic)
            sparse_encoder: Sparse encoder config
            chat_history_dict: Chat history
            max_community_reports: Maximum community reports to use (default: 3)
            top_k: Number of seed nodes to retrieve
            contextualize_prompt: Enable query rewriting for retrieval
            include_history_in_prompt: Include history in final synthesis
            max_history_messages: Limit history turns
            conversation_summary: Enable history summarization

        Returns:
            Response dict compatible with GraphQAAdvancedResponse
        """
        logger.info(f"Processing advanced QA query: {question}")

        # 0. HYBRID HISTORY PREPARATION
        retrieval_query = question
        summary_text = ""
        
        if chat_history_dict and (contextualize_prompt or conversation_summary):
            from tilellm.controller.controller_utils import summarize_history, create_contextualize_query
            from tilellm.models import QuestionAnswer
            
            qa_mock = QuestionAnswer(
                question=question,
                namespace=namespace,
                engine=engine,
                chat_history_dict=chat_history_dict,
                contextualize_prompt=contextualize_prompt,
                max_history_messages=max_history_messages
            )

            if contextualize_prompt:
                retrieval_query = await create_contextualize_query(self.llm, qa_mock)
                logger.info(f"Contextualized query for retrieval: {retrieval_query}")
            
            if conversation_summary:
                sorted_keys = sorted(chat_history_dict.keys(), key=lambda x: int(x))
                if len(sorted_keys) > max_history_messages:
                    from ..utils import format_chat_history
                    old_keys = sorted_keys[:-max_history_messages]
                    old_history_dict = {k: chat_history_dict[k] for k in old_keys}
                    old_history_text = format_chat_history(old_history_dict)
                    summary_text = await summarize_history(old_history_text, self.llm)

        # Step 1: Classify Intent (using retrieval_query for better accuracy)
        intent, confidence = await self.intent_classifier.classify_intent(retrieval_query)
        logger.info(f"Intent classified: {intent} (confidence: {confidence:.2f})")

        # Step 2: Extract Parameters (Considering History)
        parameters = await self.parameter_extractor.extract_parameters(
            retrieval_query, intent, chat_history_dict=chat_history_dict
        )
        logger.info(f"Extracted parameters: {parameters}")

        # Step 3: Find Seed Nodes from Vector Store
        seed_documents = await self._find_seed_nodes(
            question=retrieval_query,
            namespace=namespace,
            engine=engine,
            vector_store_repo=vector_store_repo,
            llm_embeddings=llm_embeddings,
            search_type=search_type,
            sparse_encoder=sparse_encoder,
            top_k=top_k
        )
        logger.info(f"Found {len(seed_documents)} seed documents from vector store")

        # Step 4: Get Relevant Community Reports (max 3)
        community_reports = await self._get_community_reports(
            question=retrieval_query,
            namespace=namespace,
            engine=engine,
            vector_store_repo=vector_store_repo,
            llm_embeddings=llm_embeddings,
            max_reports=max_community_reports,
            graph_name=graph_name
        )
        logger.info(f"Retrieved {len(community_reports)} community reports")

        # Step 5: Generate and Execute Cypher Queries
        cypher_results = await self._execute_cypher_query(
            intent=intent,
            parameters=parameters,
            namespace=namespace,
            index_name=engine.index_name if hasattr(engine, 'index_name') else None,
            graph_name=graph_name
        )
        logger.info(f"Cypher query returned {len(cypher_results)} results")

        # Step 6: Enrich Response with LLM
        final_answer = await self._enrich_response(
            question=question,
            intent=intent,
            parameters=parameters,
            seed_documents=seed_documents,
            community_reports=community_reports,
            cypher_results=cypher_results,
            chat_history_dict=chat_history_dict,
            include_history=include_history_in_prompt,
            max_history=max_history_messages,
            summary_text=summary_text
        )

        # Extract entities and relationships from Cypher results
        entities, relationships = self._extract_graph_elements(cypher_results)

        # Update chat history with current turn in project standard format (one entry per turn with question/answer)
        updated_history = chat_history_dict.copy() if chat_history_dict else {}
        turn_id = str(len(updated_history))
        from tilellm.models import ChatEntry
        updated_history[turn_id] = ChatEntry(
            question=question,
            answer=final_answer
        )

        return {
            "answer": final_answer,
            "entities": entities,
            "relationships": relationships,
            "query_used": question,
            "retrieval_strategy": f"advanced_qa_{intent}",
            "scores": {
                "intent": intent,
                "confidence": confidence,
                "seed_documents": len(seed_documents),
                "community_reports": len(community_reports),
                "cypher_results": len(cypher_results)
            },
            "expanded_nodes": entities,
            "expanded_relationships": relationships,
            "chat_history_dict": updated_history,
            "query_contextualized": retrieval_query if contextualize_prompt else None
        }

    async def _find_seed_nodes(
        self,
        question: str,
        namespace: str,
        engine,
        vector_store_repo,
        llm_embeddings,
        search_type: str,
        sparse_encoder,
        top_k: int
    ) -> List[Document]:
        """Find seed nodes using vector store search."""
        from tilellm.models import QuestionAnswer
        from tilellm.controller.controller_utils import fetch_question_vectors

        # Build QuestionAnswer for search
        qa = QuestionAnswer(
            question=question,
            namespace=namespace,
            engine=engine,
            search_type=search_type,
            top_k=top_k,
            llm="openai",
            model="gpt-4"
        )

        # Initialize embeddings and index
        _, sparse_enc, index = await vector_store_repo.initialize_embeddings_and_index(
            qa, llm_embeddings
        )

        # Get vectors
        if search_type == "hybrid" and sparse_enc:
            dense_vector, sparse_vector = await fetch_question_vectors(qa, sparse_enc, llm_embeddings)
        else:
            dense_vector = await llm_embeddings.aembed_query(question)
            sparse_vector = None

        # Perform search
        results = await vector_store_repo.perform_hybrid_search(qa, index, dense_vector, sparse_vector)

        # Convert to Documents
        documents = []
        if results and results.get('matches'):
            for match in results['matches'][:top_k]:
                metadata = match.get('metadata', {})
                doc = Document(
                    page_content=metadata.get('text', ''),
                    metadata={
                        'id': match.get('id'),
                        'score': match.get('score', 0),
                        **metadata
                    }
                )
                documents.append(doc)

        return documents

    async def _get_community_reports(
        self,
        question: str,
        namespace: str,
        engine,
        vector_store_repo,
        llm_embeddings,
        max_reports: int,
        graph_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant community reports (max 3)."""
        from tilellm.models import QuestionAnswer
        from tilellm.controller.controller_utils import fetch_question_vectors

        graph_name_to_use = graph_name if graph_name else namespace
        report_namespace = f"{graph_name_to_use}-reports"

        try:
            qa = QuestionAnswer(
                question=question,
                namespace=report_namespace,
                engine=engine,
                search_type="hybrid",
                top_k=max_reports,
                llm="openai",
                model="gpt-4"
            )

            _, sparse_enc, index = await vector_store_repo.initialize_embeddings_and_index(
                qa, llm_embeddings
            )

            dense_vector, sparse_vector = await fetch_question_vectors(qa, sparse_enc, llm_embeddings)

            results = await vector_store_repo.search_community_report(qa, index, dense_vector, sparse_vector)

            reports = []
            if results and results.get('matches'):
                for match in results['matches'][:max_reports]:
                    metadata = match.get('metadata', {})
                    reports.append({
                        'title': metadata.get('title', ''),
                        'summary': metadata.get('summary', ''),
                        'full_report': metadata.get('full_report', ''),
                        'level': metadata.get('level', 0),
                        'rating': metadata.get('rating', 0.0)
                    })

            return reports
        except Exception as e:
            logger.warning(f"Failed to retrieve community reports: {e}")
            return []

    async def _execute_cypher_query(
        self,
        intent: str,
        parameters: Dict[str, Any],
        namespace: str,
        index_name: Optional[str],
        graph_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query based on intent and parameters."""
        # Generate query
        cypher_query, query_params = self.cypher_engine.generate_query(
            intent=intent,
            parameters=parameters,
            namespace=namespace,
            index_name=index_name
        )

        logger.info(f"Executing Cypher query:\n{cypher_query}")
        logger.info(f"With parameters: {query_params}")

        try:
            # Execute query using graph repository
            repo = self.graph_service._get_repository()
            
            # DEBUG: First check if we can find any debtor nodes
            debug_query = """
                MATCH (debtor)
                WHERE debtor.namespace = $namespace 
                AND (COALESCE(toUpper(debtor.name), '') CONTAINS $debtor_name OR COALESCE(debtor.entity_name, '') CONTAINS $debtor_name)
                RETURN debtor.entity_name as entity_name, debtor.name as name, labels(debtor) as labels, count(*) as count
                LIMIT 5
            """
            debug_results = await repo._execute_query(debug_query, query_params, namespace=namespace, graph_name=graph_name)
            logger.info(f"DEBUG debtor search results: {debug_results}")

            # Also check for any nodes with timestamp/date or other date fields
            timestamp_query = """
                MATCH (n)
                WHERE n.namespace = $namespace
                AND (
                    n.timestamp IS NOT NULL OR
                    n.date IS NOT NULL OR
                    n.created_at IS NOT NULL OR
                    n.updated_at IS NOT NULL OR
                    n.start_date IS NOT NULL OR
                    n.end_date IS NOT NULL OR
                    n.due_date IS NOT NULL OR
                    n.creation_date IS NOT NULL OR
                    n.event_date IS NOT NULL OR
                    n.time IS NOT NULL OR
                    n.import_timestamp IS NOT NULL
                )
                RETURN labels(n)[0] as label, count(*) as count
                LIMIT 10
            """
            timestamp_results = await repo._execute_query(timestamp_query, query_params, namespace=namespace, graph_name=graph_name)
            logger.info(f"DEBUG nodes with timestamp/date: {timestamp_results}")

            # Debug: check property keys for first few nodes
            property_query = """
                MATCH (n)
                WHERE n.namespace = $namespace
                RETURN labels(n)[0] as label, keys(n) as keys
                LIMIT 5
            """
            property_results = await repo._execute_query(property_query, query_params, namespace=namespace, graph_name=graph_name)
            logger.info(f"DEBUG node properties: {property_results}")

            # Debug: check import_timestamp values
            import_timestamp_query = """
                MATCH (n)
                WHERE n.namespace = $namespace
                AND n.import_timestamp IS NOT NULL
                RETURN labels(n)[0] as label, n.name as name, n.import_timestamp as import_timestamp
                LIMIT 5
            """
            import_timestamp_results = await repo._execute_query(import_timestamp_query, query_params, namespace=namespace, graph_name=graph_name)
            logger.info(f"DEBUG import_timestamp values: {import_timestamp_results}")

            # Debug: check relationships from debtor nodes
            relationship_query = """
                MATCH (debtor)
                WHERE debtor.namespace = $namespace
                AND (COALESCE(toUpper(debtor.name), '') CONTAINS $debtor_name OR COALESCE(debtor.entity_name, '') CONTAINS $debtor_name)
                MATCH (debtor)-[r]-(other)
                RETURN type(r) as rel_type, labels(other)[0] as other_label, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """
            relationship_results = await repo._execute_query(relationship_query, query_params, namespace=namespace, graph_name=graph_name)
            logger.info(f"DEBUG debtor relationships: {relationship_results}")

            results = await repo._execute_query(
                cypher_query,
                query_params,
                namespace=namespace,
                graph_name=graph_name
            )
            logger.info(f"Cypher query raw results (first 3): {results[:3] if results else []}")

            return results if results else []
        except Exception as e:
            logger.error(f"Cypher query execution failed: {e}")
            return []

    async def _enrich_response(
        self,
        question: str,
        intent: str,
        parameters: Dict[str, Any],
        seed_documents: List[Document],
        community_reports: List[Dict[str, Any]],
        cypher_results: List[Dict[str, Any]],
        chat_history_dict: Optional[Dict[str, Any]],
        include_history: bool = True,
        max_history: int = 10,
        summary_text: str = ""
    ) -> str:
        """Use LLM to transform raw results into readable response."""
        
        # Check for Two-Phase Response Condition (High volume of results)
        # BYPASS if user explicitly asked for "complete/full" results
        is_full_request = parameters.get("is_full_request", False)
        
        if len(cypher_results) > self.high_volume_threshold and not is_full_request:
            logger.info(f"High volume of results ({len(cypher_results)}) detected for intent '{intent}'. Using Summary Mode.")
            return await self._generate_summary(
                question, intent, parameters, cypher_results, chat_history_dict,
                include_history=include_history, max_history=max_history, summary_text=summary_text
            )

        # Build context from all sources
        context_parts = []

        # Community reports context
        if community_reports:
            context_parts.append("=== COMMUNITY REPORTS ===")
            for idx, report in enumerate(community_reports, 1):
                context_parts.append(f"\nReport {idx}: {report['title']}")
                context_parts.append(report['summary'])

        # Seed documents context
        if seed_documents:
            context_parts.append("\n=== RELEVANT DOCUMENTS ===")
            for idx, doc in enumerate(seed_documents[:10], 1):
                context_parts.append(f"\nDocument {idx}: {doc.page_content[:500]}...")

        context = "\n".join(context_parts)

        # Format chat history
        history_text = ""
        if include_history:
            from ..utils import format_chat_history
            history_text = format_chat_history(chat_history_dict, max_messages=max_history)

        # Format Graph Results specifically for the intent
        formatted_events = self._format_cypher_results(intent, cypher_results)

        # Select the specific prompt for the intent
        prompt_template = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["general"])
        
        # Construct the final prompt
        prompt = f"QUESTION: {question}\n"
        if summary_text:
            prompt += f"\n\nCONVERSATION SUMMARY:\n{summary_text}"
        if history_text and history_text != "No chat history available.":
            prompt += f"\n\nCHAT HISTORY:\n{history_text}"
            
        prompt += prompt_template.format(
            question=question,
            parameters=parameters,
            context=context,
            formatted_events=formatted_events,
            history_text=history_text # Keep for backward compatibility in template
        )

        # Get LLM response
        try:
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM enrichment failed: {e}")
            return f"Based on the available information:\n\n{context}\n\n{formatted_events}"

    async def _generate_timeline_summary(
        self,
        question: str,
        parameters: Dict[str, Any],
        cypher_results: List[Dict[str, Any]],
        chat_history_dict: Optional[Dict[str, Any]]
    ) -> str:
        """Handle high-volume timeline queries with a summary + option to expand.
        Now delegates to the generic _generate_summary method for consistency."""
        return await self._generate_summary(
            question=question,
            intent="timeline",
            parameters=parameters,
            cypher_results=cypher_results,
            chat_history_dict=chat_history_dict
        )

    async def _generate_summary(
        self,
        question: str,
        intent: str,
        parameters: Dict[str, Any],
        cypher_results: List[Dict[str, Any]],
        chat_history_dict: Optional[Dict[str, Any]],
        include_history: bool = True,
        max_history: int = 10,
        summary_text: str = ""
    ) -> str:
        """Handle high-volume queries with a summary + option to expand for all intents."""
        
        total_items = len(cypher_results)
        
        # Select subset for formatting (first 15 + last 15 for large datasets)
        subset_size = min(30, total_items)
        if total_items > 30:
            subset_results = cypher_results[:15] + cypher_results[-15:]
        else:
            subset_results = cypher_results
            
        formatted_events = self._format_cypher_results(intent, subset_results)
        
        # Determine prompt template
        if intent == "timeline":
            prompt_key = "timeline_summary"
        else:
            prompt_key = f"{intent}_summary"
            
        prompt_template = INTENT_PROMPTS.get(prompt_key, INTENT_PROMPTS["general"])
        
        # Format chat history
        history_text = ""
        if include_history:
            from ..utils import format_chat_history
            history_text = format_chat_history(chat_history_dict, max_messages=max_history)
            
        # Build prompt
        prompt = f"QUESTION: {question}\n"
        if summary_text:
            prompt += f"\n\nCONVERSATION SUMMARY:\n{summary_text}"
        if history_text and history_text != "No chat history available.":
            prompt += f"\n\nCHAT HISTORY:\n{history_text}"

        # Build prompt parameters for template
        if prompt_key == "timeline_summary":
            prompt += prompt_template.format(
                question=question,
                event_count=total_items,
                formatted_events=formatted_events,
                history_text=history_text
            )
        else:
            prompt += prompt_template.format(
                question=question,
                item_count=total_items,
                formatted_events=formatted_events,
                history_text=history_text
            )
        
        try:
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                
            # Append the interactive offer to show all results
            offer_text = self._get_summary_offer(intent, total_items)
            return f"{content}\n\n{offer_text}"
            
        except Exception as e:
            logger.error(f"LLM summary generation failed for intent '{intent}': {e}")
            return f"Ho trovato {total_items} elementi. Ecco un estratto:\n{formatted_events}"
    
    def _get_summary_offer(self, intent: str, total_items: int) -> str:
        """Generate appropriate offer text based on intent."""
        intent_names = {
            "timeline": "timeline completa",
            "exposure": "lista completa delle esposizioni",
            "guarantees": "lista completa delle garanzie", 
            "payments": "lista completa dei pagamenti",
            "contacts": "lista completa delle comunicazioni",
            "legal_actions": "lista completa delle azioni legali",
            "debtor_info": "informazioni complete sul debitore",
            "relationship": "relazioni complete"
        }
        intent_name = intent_names.get(intent, "risultati completi")
        return f"[Nota: Sono stati omessi alcuni elementi minori. Chiedi 'Mostra tutto' o '{intent_name}' per visualizzare tutti i {total_items} elementi.]"

    def _format_cypher_results(self, intent: str, results: List[Dict[str, Any]]) -> str:
        """
        Format Cypher results based on intent type.
        Now handles robust formatting of dates and amounts.
        """
        if not results:
            return "No graph results found."

        formatted_lines = []

        if intent == "timeline":
            # Sort by date reference
            sorted_results = sorted(
                results, 
                key=lambda x: (
                    x.get('date_reference') or 
                    x.get('entity_import_timestamp') or 
                    x.get('relationship_import_timestamp') or 
                    '9999-99-99'
                )
            )
            
            for idx, result in enumerate(sorted_results, 1):
                debtor = result.get('debtor_name', 'Unknown')
                entity = result.get('connected_entity_name', 'Unknown')
                rel_type = result.get('relationship_type', 'RELATION')
                date_ref = result.get('date_reference', 'N/D')
                
                # Format Description
                desc = result.get('relationship_description') or result.get('connected_description') or ""
                # Try to extract amount from description if not present in separate field
                amount_str = ""
                extracted_amount = self._extract_amount(desc)
                if extracted_amount:
                    amount_str = f" | {self._format_amount_value(extracted_amount)}"
                
                formatted_lines.append(
                    f"{idx}. [{date_ref}] {debtor} -> {rel_type} -> {entity}{amount_str} | {desc[:150]}"
                )

        elif intent == "exposure":
            for result in results:
                debtor = result.get('debtor', 'Unknown')
                loan_id = result.get('loan_id', 'N/A')
                total = result.get('total', 0)
                formatted_lines.append(f"- Pratica {loan_id} ({debtor}): Esposizione Totale {self._format_amount_value(total)}")

        elif intent == "guarantees":
             for result in results:
                g_type = result.get('guarantee_type', 'Garanzia')
                value = result.get('value')
                val_str = self._format_amount_value(value) if value else "Valore N/D"
                status = result.get('status', 'N/D')
                formatted_lines.append(f"- {g_type}: {val_str} (Stato: {status}) | {result.get('details', '')}")

        elif intent == "relationship":
            # Check for guarantors results (has guarantor_name field)
            if results and 'guarantor_name' in results[0]:
                for idx, result in enumerate(results, 1):
                    debtor = result.get('debtor_name', 'Unknown')
                    guarantor = result.get('guarantor_name', 'Unknown')
                    g_type = result.get('guarantor_type', 'Garante')
                    loan_ids = result.get('loan_ids', [])
                    loan_str = f"Pratiche: {', '.join(loan_ids)}" if loan_ids else "Nessuna pratica specificata"
                    formatted_lines.append(f"{idx}. {guarantor} ({g_type}) garante per {debtor} | {loan_str}")
            # Check for relationship_by_type results (has connected_entity_name field)
            elif results and 'connected_entity_name' in results[0]:
                for idx, result in enumerate(results, 1):
                    debtor = result.get('debtor_name', 'Unknown')
                    entity = result.get('connected_entity_name', 'Unknown')
                    rel_type = result.get('relationship_type', 'RELATED_TO')
                    date_ref = result.get('date_reference', 'N/D')
                    desc = result.get('connected_description') or result.get('relationship_description') or ""
                    formatted_lines.append(f"{idx}. [{date_ref}] {debtor} -[{rel_type}]-> {entity} | {desc[:150]}")
            elif intent == "vespro":
                formatted_lines.append("=== RELAZIONI FERDINANDO III D'ARAGONA - VESPRI SICILIANI ===\n")
                for idx, result in enumerate(results, 1):
                    ferdinando = result.get('ferdinando_name', 'Ferdinando III d\'Aragona')
                    protagonista = result.get('protagonista_name', 'Unknown')
                    p_type = result.get('protagonista_type', 'Personaggio')
                    description = result.get('protagonista_description', '')[:200]

                    formatted_lines.append(
                        f"{idx}. **{protagonista}** ({p_type})\n"
                        f"   Collegato a: {ferdinando}\n"
                        f"   Descrizione: {description}\n"
                    )
            else:
                # Generic relationship results (path-based)
                for idx, result in enumerate(results[:20], 1):
                    if 'node_names' in result and 'relationship_types' in result:
                        nodes = result.get('node_names', [])
                        rels = result.get('relationship_types', [])
                        path_str = " → ".join([f"{nodes[i]} -[{rels[i]}]->" if i < len(rels) else nodes[i] for i in range(len(nodes))])
                        formatted_lines.append(f"{idx}. {path_str}")
                    else:
                        formatted_lines.append(f"{idx}. {result}")

        else:
            # Generic fallback
            for idx, result in enumerate(results[:20], 1):
                formatted_lines.append(f"{idx}. {result}")

        return "\n".join(formatted_lines)

    def _format_amount_value(self, value: Any) -> str:
        """Format a numeric value as EUR currency string."""
        if value is None:
            return ""
        try:
            float_val = float(value)
            return f"€ {float_val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except (ValueError, TypeError):
            return str(value)

    def _extract_amount(self, text: str) -> Optional[float]:
        """Estrae importi da descrizioni testuali (supporta €, euro, formats italiani)."""
        if not text:
            return None
        # Pattern per importi italiani (es. € 10.000,00 o 10.000,00 euro)
        patterns = [
            r'€\s*([\d\.,]+)',
            r'([\d\.,]+)\s*euro',
            r'([\d\.,]+)\s*€',
            r'importo\s*[:\s]*([\d\.,]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Normalizza formato italiano (10.000,00 → 10000.00)
                    # Handle cases like 1.000,00 -> 1000.00 and 1,000.00 -> 1000.00
                    num_str = match.group(1)
                    if ',' in num_str and '.' in num_str:
                         if num_str.find(',') > num_str.find('.'): # 1.000,00
                             num_str = num_str.replace('.', '').replace(',', '.')
                         else: # 1,000.00
                             num_str = num_str.replace(',', '')
                    elif ',' in num_str: # 1000,00
                        num_str = num_str.replace(',', '.')
                    
                    return float(num_str)
                except:
                    continue
        return None


    def _extract_graph_elements(self, cypher_results: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from Cypher results (handles flat rows)."""
        entities = []
        relationships = []
        seen_entities = set()
        seen_relationships = set()

        for result in cypher_results:
            # Case 1: Result contains full node/rel objects (dict with 'id')
            for key, value in result.items():
                if isinstance(value, dict) and 'id' in value:
                    entity_id = value.get('id')
                    if entity_id and entity_id not in seen_entities:
                        entities.append({
                            'id': str(entity_id),
                            'label': value.get('label', 'Entity'),
                            'properties': {k: v for k, v in value.items() if k not in ['id', 'label']}
                        })
                        seen_entities.add(entity_id)

            # Case 2: Result contains flat rows (like our timeline/exposure queries)
            debtor_raw = result.get('debtor_name') or result.get('debtor')
            connected_raw = result.get('connected_entity_name') or result.get('connected_name') or result.get('loan_id')
            rel_type = result.get('relationship_type') or result.get('rel_type') or "RELATED_TO"

            # Extract names from potential dict objects
            debtor_name = None
            if isinstance(debtor_raw, dict):
                debtor_name = debtor_raw.get('name') or debtor_raw.get('entity_name') or str(debtor_raw.get('id', ''))
            elif debtor_raw:
                debtor_name = str(debtor_raw)
                
            connected_name = None
            if isinstance(connected_raw, dict):
                connected_name = connected_raw.get('name') or connected_raw.get('entity_name') or str(connected_raw.get('id', ''))
            elif connected_raw:
                connected_name = str(connected_raw)

            if debtor_name:
                if debtor_name not in seen_entities:
                    entities.append({
                        "id": debtor_name,
                        "label": "Debtor",
                        "properties": {"name": debtor_name}
                    })
                    seen_entities.add(debtor_name)
                
                if connected_name:
                    if connected_name not in seen_entities:
                        entities.append({
                            "id": connected_name,
                            "label": result.get('connected_entity_type', 'Entity'),
                            "properties": {
                                "name": connected_name,
                                "description": result.get('connected_description'),
                                "date": result.get('date_reference')
                            }
                        })
                        seen_entities.add(connected_name)
                    
                    # Create relationship
                    rel_id = f"{debtor_name}-{rel_type}-{connected_name}"
                    if rel_id not in seen_relationships:
                        relationships.append({
                            "id": rel_id,
                            "source": debtor_name,
                            "target": connected_name,
                            "type": rel_type,
                            "properties": {
                                "description": result.get('relationship_description'),
                                "date": result.get('date_reference')
                            }
                        })
                        seen_relationships.add(rel_id)

        return entities, relationships
