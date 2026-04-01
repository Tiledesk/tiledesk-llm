"""
Unit tests for Knowledge Graph Falkor models and schemas.
Tests Pydantic model validation and serialization.
"""
import pytest
from pydantic import ValidationError
from tilellm.modules.knowledge_graph_falkor.models import Node, NodeUpdate, Relationship, RelationshipUpdate
from tilellm.modules.knowledge_graph_falkor.models.schemas import (
    GraphQARequest, GraphQAResponse, GraphCreateRequest, GraphCreateResponse,
    GraphClusterRequest, GraphClusterResponse, CommunityQAResponse,
    AddDocumentRequest, AddDocumentResponse, AsyncTaskResponse,
    GraphQAAdvancedRequest, GraphQAAdvancedResponse,
    GraphNetworkResponse, MultimodalSearchResponse, TaskPollResponse
)


class TestNodeModel:
    """Test Node Pydantic model."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(label="Person", properties={"name": "John", "age": 30})
        
        assert node.label == "Person"
        assert node.properties["name"] == "John"
        assert node.properties["age"] == 30
        assert node.id is None
    
    def test_node_with_id(self):
        """Test node creation with ID."""
        node = Node(id=123, label="Organization", properties={"name": "Acme"})
        
        assert node.id == 123
        assert node.label == "Organization"
    
    def test_node_default_properties(self):
        """Test node with default empty properties."""
        node = Node(label="Document")
        
        assert node.properties == {}
    
    def test_node_serialization(self):
        """Test node JSON serialization."""
        node = Node(id=1, label="Person", properties={"name": "Alice"})
        json_data = node.model_dump()
        
        assert json_data["id"] == 1
        assert json_data["label"] == "Person"
        assert json_data["properties"]["name"] == "Alice"
    
    def test_node_example_generation(self):
        """Test that model can generate example."""
        example = Node.model_json_schema().get("example", {})
        # Example should be valid
        assert "label" in example


class TestNodeUpdateModel:
    """Test NodeUpdate Pydantic model."""
    
    def test_node_update_creation(self):
        """Test node update creation."""
        update = NodeUpdate(properties={"age": 35})
        
        assert update.properties["age"] == 35
        assert update.label is None
    
    def test_node_update_label(self):
        """Test node update with label change."""
        update = NodeUpdate(label="Employee", properties={"department": "IT"})
        
        assert update.label == "Employee"
        assert update.properties["department"] == "IT"


class TestRelationshipModel:
    """Test Relationship Pydantic model."""
    
    def test_relationship_creation(self):
        """Test basic relationship creation."""
        rel = Relationship(
            source_id=1,
            target_id=2,
            type="KNOWS",
            properties={"since": "2020"}
        )
        
        assert rel.source_id == 1
        assert rel.target_id == 2
        assert rel.type == "KNOWS"
        assert rel.properties["since"] == "2020"
        assert rel.id is None
    
    def test_relationship_with_id(self):
        """Test relationship with ID."""
        rel = Relationship(
            id=100,
            source_id=1,
            target_id=2,
            type="WORKS_WITH",
            properties={}
        )
        
        assert rel.id == 100
    
    def test_relationship_default_properties(self):
        """Test relationship with default empty properties."""
        rel = Relationship(source_id=1, target_id=2, type="RELATED_TO")
        
        assert rel.properties == {}


class TestRelationshipUpdateModel:
    """Test RelationshipUpdate Pydantic model."""
    
    def test_relationship_update_creation(self):
        """Test relationship update creation."""
        update = RelationshipUpdate(properties={"weight": 0.8})
        
        assert update.properties["weight"] == 0.8
        assert update.type is None
    
    def test_relationship_update_type(self):
        """Test relationship update with type change."""
        update = RelationshipUpdate(type="FRIENDS_WITH")
        
        assert update.type == "FRIENDS_WITH"


class TestGraphQARequestSchema:
    """Test GraphQARequest schema."""
    
    def test_basic_request(self):
        """Test basic QA request creation."""
        request = GraphQARequest(
            question="What is GraphRAG?",
            namespace="test_ns",
            llm_key="test-key"
        )
        
        assert request.question == "What is GraphRAG?"
        assert request.namespace == "test_ns"
        assert request.max_results == 10  # Default value
    
    def test_request_with_index_name(self):
        """Test QA request with index name."""
        request = GraphQARequest(
            question="Query",
            namespace="ns",
            index_name="my_index",
            llm_key="key"
        )
        
        assert request.index_name == "my_index"
    
    def test_request_with_chat_history(self):
        """Test QA request with chat history."""
        request = GraphQARequest(
            question="Follow up",
            namespace="ns",
            chat_history_dict={"messages": [{"role": "user", "content": "Hello"}]},
            llm_key="key"
        )
        
        assert "messages" in request.chat_history_dict


class TestGraphCreateRequestSchema:
    """Test GraphCreateRequest schema."""
    
    def test_basic_create_request(self):
        """Test basic graph creation request."""
        from tilellm.models import Engine
        
        engine = Engine(name="qdrant", index_name="test_idx")
        request = GraphCreateRequest(
            namespace="test_ns",
            engine=engine,
            llm_key="key"
        )
        
        assert request.namespace == "test_ns"
        assert request.engine.name == "qdrant"
        assert request.limit == 100  # Default
        assert request.overwrite is False  # Default
    
    def test_create_request_with_overwrite(self):
        """Test graph creation with overwrite flag."""
        from tilellm.models import Engine
        
        request = GraphCreateRequest(
            namespace="ns",
            engine=Engine(name="pinecone", index_name="idx"),
            overwrite=True,
            llm_key="key"
        )
        
        assert request.overwrite is True


class TestGraphClusterRequestSchema:
    """Test GraphClusterRequest schema."""
    
    def test_basic_cluster_request(self):
        """Test basic cluster request."""
        from tilellm.models import Engine
        
        request = GraphClusterRequest(
            namespace="test_ns",
            engine=Engine(name="qdrant", index_name="idx"),
            llm_key="key"
        )
        
        assert request.namespace == "test_ns"
        assert request.level == 0  # Default
        assert request.overwrite is True  # Default
    
    def test_cluster_request_with_webhook(self):
        """Test cluster request with webhook."""
        from tilellm.models import Engine
        
        request = GraphClusterRequest(
            namespace="ns",
            engine=Engine(name="qdrant", index_name="idx"),
            webhook_url="https://example.com/webhook",
            llm_key="key"
        )
        
        assert request.webhook_url == "https://example.com/webhook"


class TestAddDocumentRequestSchema:
    """Test AddDocumentRequest schema."""
    
    def test_basic_add_document_request(self):
        """Test basic add document request."""
        from tilellm.models import Engine
        
        request = AddDocumentRequest(
            metadata_id="doc_123",
            namespace="test_ns",
            engine=Engine(name="qdrant", index_name="idx"),
            llm_key="key"
        )
        
        assert request.metadata_id == "doc_123"
        assert request.namespace == "test_ns"
        assert request.deduplicate_entities is True  # Default


class TestResponseSchemas:
    """Test response schemas."""
    
    def test_graph_qa_response(self):
        """Test GraphQA response."""
        response = GraphQAResponse(
            answer="GraphRAG is a technique...",
            entities=[{"name": "GraphRAG", "type": "TECHNOLOGY"}],
            relationships=[],
            query_used="What is GraphRAG?"
        )
        
        assert "GraphRAG" in response.answer
        assert len(response.entities) == 1
    
    def test_graph_create_response(self):
        """Test graph creation response."""
        response = GraphCreateResponse(
            namespace="test_ns",
            chunks_processed=10,
            nodes_created=25,
            relationships_created=50,
            status="success"
        )
        
        assert response.chunks_processed == 10
        assert response.nodes_created == 25
    
    def test_graph_cluster_response(self):
        """Test graph cluster response."""
        response = GraphClusterResponse(
            status="success",
            communities_detected=5,
            reports_created=5,
            message="Clustering completed"
        )
        
        assert response.communities_detected == 5
    
    def test_community_qa_response(self):
        """Test community QA response."""
        response = CommunityQAResponse(
            answer="Based on community reports...",
            reports_used=3
        )
        
        assert response.reports_used == 3
    
    def test_add_document_response(self):
        """Test add document response."""
        response = AddDocumentResponse(
            metadata_id="doc_123",
            chunks_processed=5,
            entities_extracted=10,
            entities_new=8,
            entities_reused=2,
            relationships_created=15,
            status="success"
        )
        
        assert response.entities_new == 8
        assert response.entities_reused == 2
    
    def test_async_task_response(self):
        """Test async task response."""
        response = AsyncTaskResponse(task_id="task_123")
        
        assert response.task_id == "task_123"
        assert response.status == "queued"  # Default
    
    def test_task_poll_response(self):
        """Test task poll response variants."""
        # In progress
        response_in_progress = TaskPollResponse(
            task_id="task_123",
            status="in_progress"
        )
        assert response_in_progress.status == "in_progress"
        
        # Success
        response_success = TaskPollResponse(
            task_id="task_123",
            status="success",
            result={"nodes_created": 10}
        )
        assert response_success.result["nodes_created"] == 10
        
        # Failed
        response_failed = TaskPollResponse(
            task_id="task_123",
            status="failed",
            error="Connection timeout"
        )
        assert response_failed.error == "Connection timeout"
    
    def test_graph_network_response(self):
        """Test graph network response."""
        response = GraphNetworkResponse(
            nodes=[
                {"id": "1", "label": "Person", "properties": {"name": "Alice"}}
            ],
            relationships=[
                {"id": "r1", "source_id": "1", "target_id": "2", "type": "KNOWS", "properties": {}}
            ],
            stats={
                "node_count": 1,
                "relationship_count": 1,
                "filtered_by": {"namespace": "test"}
            }
        )
        
        assert len(response.nodes) == 1
        assert len(response.relationships) == 1
        assert response.stats["node_count"] == 1
    
    def test_multimodal_search_response(self):
        """Test multimodal search response."""
        response = MultimodalSearchResponse(
            answer="Found information...",
            sources={
                "text_chunks": [],
                "tables": [],
                "images": []
            },
            query_used="query text"
        )
        
        assert "text_chunks" in response.sources


class TestSchemaValidation:
    """Test schema validation edge cases."""
    
    def test_empty_question_validation(self):
        """Test that empty question is allowed in some schemas."""
        from tilellm.models import Engine
        
        # GraphCreateRequest allows empty question
        request = GraphCreateRequest(
            namespace="ns",
            engine=Engine(name="qdrant", index_name="idx"),
            question="",  # Empty string allowed
            llm_key="key"
        )
        
        assert request.question == ""
    
    def test_required_fields_validation(self):
        """Test required fields are enforced."""
        # Node requires label
        with pytest.raises(ValidationError):
            Node(properties={"name": "Test"})  # Missing required label
    
    def test_relationship_required_fields(self):
        """Test relationship required fields."""
        with pytest.raises(ValidationError):
            Relationship(source_id=1)  # Missing target_id and type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
