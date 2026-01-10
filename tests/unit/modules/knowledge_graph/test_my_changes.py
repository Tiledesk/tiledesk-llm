
import unittest
from unittest.mock import MagicMock, patch
from tilellm.modules.knowledge_graph.services.services import GraphService
from tilellm.modules.knowledge_graph.repository.repository import GraphRepository
from tilellm.modules.knowledge_graph.models.models import Node, Relationship

class TestGraphService(unittest.TestCase):
    def setUp(self):
        self.mock_repo = MagicMock(spec=GraphRepository)
        self.service = GraphService(repository=self.mock_repo)

    def test_get_graph_network_community_true(self):
        # Mock repository response for community network
        self.mock_repo.get_community_network.return_value = {
            "nodes": [
                {"id": "node1", "label": "Entity", "properties": {"name": "A", "namespace": "test", "index_name": "idx"}},
                {"id": "comm1", "label": "CommunityReport", "properties": {"title": "Comm A", "namespace": "test", "index_name": "idx"}}
            ],
            "relationships": [
                {"id": "rel1", "type": "BELONGS_TO_COMMUNITY", "source_id": "node1", "target_id": "comm1", "properties": {}}
            ]
        }

        result = self.service.get_graph_network(
            namespace="test",
            index_name="idx",
            community=True
        )

        # Verify repo method was called with correct parameters
        self.mock_repo.get_community_network.assert_called_once_with(
            namespace="test",
            index_name="idx",
            limit=5000
        )

        # Verify structure of response
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(result["stats"]["filtered_by"]["community"], True)
        self.assertEqual(result["relationships"][0]["type"], "BELONGS_TO_COMMUNITY")

    def test_get_graph_network_community_false(self):
        # Mock repository response for normal node search
        self.mock_repo.find_nodes_by_label.return_value = [
            Node(id="node1", label="PERSON", properties={"name": "Mario"})
        ]
        self.mock_repo.find_relationships_by_node.return_value = []

        result = self.service.get_graph_network(
            namespace="test",
            index_name="idx",
            community=False
        )

        # Verify get_community_network was NOT called
        self.mock_repo.get_community_network.assert_not_called()
        
        # Verify find_nodes_by_label was called (part of normal behavior)
        self.assertTrue(self.mock_repo.find_nodes_by_label.called)
        
        self.assertEqual(len(result["nodes"]), 1)
        self.assertEqual(result["stats"]["filtered_by"]["community"], False)

if __name__ == "__main__":
    unittest.main()
