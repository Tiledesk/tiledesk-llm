"""
Service for Graph Clustering and Community Report generation.
Uses Louvain algorithm from NetworkX for community detection.
"""


from json import loads as json_loads
import logging
import re
import networkx as nx
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..repository import GraphRepository
from ..graphrag.general.community_report_prompt import COMMUNITY_REPORT_PROMPT

logger = logging.getLogger(__name__)

class ClusterService:
    """
    Service to handle graph clustering and community report generation.
    """
    
    def __init__(self, repository: GraphRepository, llm=None):
        self.repository = repository
        self.llm = llm

    async def perform_clustering(self, level: int = 0, namespace: Optional[str] = None, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point for clustering the graph and generating reports.
        """
        logger.info("Starting graph clustering process using Louvain algorithm...")
        
        # 1. Fetch graph from Neo4j
        graph_data = self.repository.get_all_nodes_and_relationships(namespace=namespace, index_name=index_name)
        nodes = graph_data["nodes"]
        relationships = graph_data["relationships"]
        
        if not nodes:
            logger.warning("No nodes found in the graph. Skipping clustering.")
            return {"status": "empty", "reports_created": 0}
            
        # 2. Build NetworkX graph
        G = nx.Graph()
        for node in nodes:
            G.add_node(node["id"], **node["properties"], label=node["label"])
            
        for rel in relationships:
            G.add_edge(rel["source_id"], rel["target_id"], type=rel["type"], **rel["properties"])
            
        logger.info(f"Built NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Note: rank calculation removed as it's not used in Neo4j nodes
        # Pre-calculate rank (degree) for weight calculation - disabled
        # for node_id in G.nodes():
        #     degree = G.degree[node_id]  # Use dictionary-like access
        #     G.nodes[node_id]["rank"] = int(degree)
            
        # 3. Community Detection using Louvain
        try:
            from networkx.algorithms.community import louvain_communities

            # louvain_communities returns a list of sets of nodes
            communities_list = louvain_communities(G, seed=42)
            logger.info(f"Detected {len(communities_list)} communities using Louvain algorithm")
        except Exception as e:
            logger.error(f"Error during Louvain community detection: {e}")
            return {"status": "error", "message": str(e)}

        # 4. Generate Reports for each community
        reports_created = 0
        for i, community_nodes_set in enumerate(communities_list):
            community_nodes = list(community_nodes_set)
            if len(community_nodes) < 2:
                continue # Skip tiny communities
                
            logger.info(f"Processing community {i} with {len(community_nodes)} nodes")
            
            try:
                report = await self._generate_community_report(G, community_nodes)
                if report:
                    # 5. Save report to Neo4j
                    self.repository.save_community_report(
                        community_id=f"community_{datetime.now().strftime('%Y%m%d')}_{i}",
                        report=report,
                        level=level
                    )
                    reports_created += 1
            except Exception as e:
                logger.error(f"Error generating report for community {i}: {e}", exc_info=True)
                continue
                
        return {
            "status": "success",
            "communities_detected": len(communities_list),
            "reports_created": reports_created
        }

    async def _generate_community_report(self, G: nx.Graph, community_nodes: List[str]) -> Optional[Dict[str, Any]]:
        """
        Generate a report for a specific community using LLM.
        """
        if not self.llm:
            logger.warning("LLM not provided. Cannot generate community report.")
            return None
            
        # Prepare entity and relationship dataframes for the prompt
        entities_data = []
        for node_id in community_nodes:
            props = G.nodes[node_id]
            entities_data.append({
                "id": node_id,
                "entity": props.get("name") or props.get("title") or "Unnamed",
                "description": props.get("description") or ""
            })
            
        entity_df = pd.DataFrame(entities_data)
        
        relationships_data = []
        # Get edges between nodes in the community
        subgraph = G.subgraph(community_nodes)
        for u, v, data in subgraph.edges(data=True):
            relationships_data.append({
                "id": f"{u}_{v}",
                "source": G.nodes[u].get("name") or u,
                "target": G.nodes[v].get("name") or v,
                "description": data.get("description") or data.get("type") or "related"
            })
            
        relation_df = pd.DataFrame(relationships_data)
        
        # Prepare prompt
        prompt = COMMUNITY_REPORT_PROMPT.format(
            entity_df=entity_df.to_csv(index=False),
            relation_df=relation_df.to_csv(index=False)
        )
        
        response_text = ""
        try:
            # Call LLM
            if hasattr(self.llm, 'ainvoke'):
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content="You are a helpful assistant that summarizes graph communities."),
                    HumanMessage(content=prompt)
                ]
                response = await self.llm.ainvoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
            elif hasattr(self.llm, 'chat'):
                response_text = await self.llm.chat(
                    system="You are a helpful assistant that summarizes graph communities.",
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                response_text = await self.llm(prompt)
                
            # Parse JSON from response
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                report_data = json_loads(match.group())
            else:
                report_data = json_loads(response_text)
                
            report_data["entities"] = community_nodes
            
            # Create a full report string for storage
            full_report = f"# {report_data.get('title', 'Community Report')}\n\n"
            full_report += f"{report_data.get('summary', '')}\n\n"
            full_report += "## Key Findings\n\n"
            for finding in report_data.get("findings", []):
                # Ensure finding is a dict before accessing it
                if isinstance(finding, dict):
                    full_report += f"### {finding.get('summary', '')}\n{finding.get('explanation', '')}\n\n"
                else:
                    full_report += f"### Finding\n{str(finding)}\n\n"
                
            report_data["full_report"] = full_report
            return report_data
            
        except Exception as e:
            logger.error(f"Error calling LLM or parsing JSON for community report: {e}", exc_info=True)
            logger.debug(f"Response text received: {response_text[:500]}...")
            return None
