"""
Test suite for Advanced QA Service
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from tilellm.modules.knowledge_graph_falkor.services.advanced_qa_service import (
    IntentClassifier,
    ParameterExtractor,
    CypherTemplateEngine,
    AdvancedQAService
)


class TestIntentClassifier:
    """Test intent classification"""

    @pytest.mark.asyncio
    async def test_timeline_intent(self):
        classifier = IntentClassifier()
        intent, confidence = await classifier.classify_intent(
            "Qual è la cronologia degli eventi per Mario Rossi?"
        )
        assert intent == "timeline"
        assert confidence > 0.5

    @pytest.mark.asyncio
    async def test_exposure_intent(self):
        classifier = IntentClassifier()
        intent, confidence = await classifier.classify_intent(
            "Quanto deve Giovanni Bianchi?"
        )
        assert intent == "exposure"
        assert confidence > 0.5

    @pytest.mark.asyncio
    async def test_guarantees_intent(self):
        classifier = IntentClassifier()
        intent, confidence = await classifier.classify_intent(
            "Quali garanzie sono attive per la pratica LOAN-123?"
        )
        assert intent == "guarantees"
        assert confidence > 0.5

    @pytest.mark.asyncio
    async def test_payments_intent(self):
        classifier = IntentClassifier()
        intent, confidence = await classifier.classify_intent(
            "Mostrami i pagamenti di Luigi Verdi"
        )
        assert intent == "payments"
        assert confidence > 0.5

    @pytest.mark.asyncio
    async def test_legal_intent(self):
        classifier = IntentClassifier()
        intent, confidence = await classifier.classify_intent(
            "Ci sono procedimenti legali contro Mario Rossi?"
        )
        assert intent == "legal_actions"
        assert confidence > 0.5

    @pytest.mark.asyncio
    async def test_default_fallback(self):
        classifier = IntentClassifier()
        intent, confidence = await classifier.classify_intent(
            "Questa è una query generica senza keyword specifiche xyz123"
        )
        assert intent == "general"


class TestParameterExtractor:
    """Test parameter extraction"""

    @pytest.mark.asyncio
    async def test_extract_debtor_name(self):
        extractor = ParameterExtractor()
        params = await extractor.extract_parameters(
            "Qual è l'esposizione di Mario Rossi?",
            "exposure"
        )
        assert "debtor_name" in params
        assert params["debtor_name"] == "Mario Rossi"

    @pytest.mark.asyncio
    async def test_extract_loan_id(self):
        extractor = ParameterExtractor()
        params = await extractor.extract_parameters(
            "Mostrami i dettagli della pratica LOAN-789",
            "timeline"
        )
        assert "loan_id" in params
        assert params["loan_id"] == "LOAN-789"

    @pytest.mark.asyncio
    async def test_extract_dates(self):
        extractor = ParameterExtractor()
        params = await extractor.extract_parameters(
            "Pagamenti dal 15/03/2024 al 20/12/2024",
            "payments"
        )
        assert "dates" in params
        assert len(params["dates"]) >= 1

    @pytest.mark.asyncio
    async def test_extract_amounts(self):
        extractor = ParameterExtractor()
        params = await extractor.extract_parameters(
            "Debito di €250,000 per Marco Rossi",
            "exposure"
        )
        assert "amounts" in params or "debtor_name" in params


class TestCypherTemplateEngine:
    """Test Cypher query generation"""

    def test_timeline_query_generation(self):
        engine = CypherTemplateEngine()
        query, params = engine.generate_query(
            intent="timeline",
            parameters={"debtor_name": "Mario Rossi"},
            namespace="test_namespace",
            index_name="test_index"
        )
        assert "MATCH" in query
        assert "ORDER BY" in query
        assert "date_reference" in query
        assert params["debtor_name"] == "MARIO ROSSI"
        assert params["namespace"] == "test_namespace"

    def test_exposure_query_generation(self):
        engine = CypherTemplateEngine()
        query, params = engine.generate_query(
            intent="exposure",
            parameters={"debtor_name": "Giovanni Bianchi"},
            namespace="test_namespace",
            index_name="test_index"
        )
        assert "MATCH" in query
        assert "HAS_LOAN" in query
        assert "total_exposure" in query
        assert params["debtor_name"] == "GIOVANNI BIANCHI"

    def test_guarantees_query_generation(self):
        engine = CypherTemplateEngine()
        query, params = engine.generate_query(
            intent="guarantees",
            parameters={"debtor_name": "Marco Rossi"},
            namespace="test_namespace",
            index_name="test_index"
        )
        assert "MATCH" in query
        assert "SECURED_BY" in query
        assert params["debtor_name"] == "MARCO ROSSI"

    def test_unknown_intent_fallback(self):
        engine = CypherTemplateEngine()
        query, params = engine.generate_query(
            intent="unknown_intent",
            parameters={},
            namespace="test_namespace",
            index_name="test_index"
        )
        assert "MATCH" in query
        assert params["namespace"] == "test_namespace"


class TestAdvancedQAService:
    """Integration tests for Advanced QA Service"""

    @pytest.mark.asyncio
    async def test_format_cypher_results_timeline(self):
        # Mock services
        graph_service = Mock()
        graph_rag_service = Mock()
        community_service = Mock()
        llm = Mock()

        service = AdvancedQAService(
            graph_service=graph_service,
            graph_rag_service=graph_rag_service,
            community_graph_service=community_service,
            llm=llm
        )

        results = [
            {
                "debtor_name": "Mario Rossi",
                "connected_entity_name": "Loan Agreement 001",
                "connected_entity_type": "LOAN",
                "relationship_type": "HAS_LOAN",
                "date_reference": "2024-01-01",
                "connected_description": "Loan originated for business expansion"
            },
            {
                "debtor_name": "Mario Rossi",
                "connected_entity_name": "Default Notice",
                "connected_entity_type": "LEGAL_PROCEEDING",
                "relationship_type": "HAS_DEFAULT",
                "date_reference": "2024-06-01",
                "connected_description": "First payment default recorded"
            },
            {
                "debtor_name": "Mario Rossi",
                "connected_entity_name": "Court Summons",
                "connected_entity_type": "LEGAL_PROCEEDING",
                "relationship_type": "HAS_LEGAL_ACTION",
                "date_reference": "2024-09-01",
                "connected_description": "Legal action initiated for debt recovery"
            }
        ]

        formatted = service._format_cypher_results("timeline", results)
        assert "2024-01-01" in formatted
        assert "Loan Agreement 001" in formatted
        assert "HAS_LOAN" in formatted
        assert "2024-06-01" in formatted
        assert "Default Notice" in formatted

    @pytest.mark.asyncio
    async def test_format_cypher_results_exposure(self):
        graph_service = Mock()
        graph_rag_service = Mock()
        community_service = Mock()
        llm = Mock()

        service = AdvancedQAService(
            graph_service=graph_service,
            graph_rag_service=graph_rag_service,
            community_graph_service=community_service,
            llm=llm
        )

        results = [
            {"debtor": "Mario Rossi", "total": 150000},
            {"debtor": "Luigi Verdi", "total": 75000}
        ]

        formatted = service._format_cypher_results("exposure", results)
        assert "Mario Rossi" in formatted
        assert "150,000" in formatted or "150000" in formatted

    def test_extract_graph_elements(self):
        graph_service = Mock()
        graph_rag_service = Mock()
        community_service = Mock()
        llm = Mock()

        service = AdvancedQAService(
            graph_service=graph_service,
            graph_rag_service=graph_rag_service,
            community_graph_service=community_service,
            llm=llm
        )

        cypher_results = [
            {
                "debtor": {
                    "id": "node_123",
                    "label": "DEBTOR",
                    "name": "Mario Rossi"
                }
            },
            {
                "loan": {
                    "id": "node_456",
                    "label": "LOAN",
                    "amount": 250000
                }
            }
        ]

        entities, relationships = service._extract_graph_elements(cypher_results)
        assert len(entities) == 2
        assert entities[0]["id"] == "node_123"
        assert entities[1]["id"] == "node_456"


# Integration test example (requires actual setup)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline():
    """
    Full integration test - requires actual FalkorDB and vector store setup.
    Run with: pytest -m integration
    """
    # This would require actual service setup
    # Mock implementation for example
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
