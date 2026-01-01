import logging
import asyncio
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any, Optional
from datetime import datetime
from io import BytesIO
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

from tilellm.modules.knowledge_graph.repository.repository import GraphRepository

# Supponiamo tu abbia un client Minio o un repository per i file
# from ..utils.minio_client import minio_client

logger = logging.getLogger(__name__)


class ClusterService:
    def __init__(self, repository: GraphRepository, llm=None, minio_client=None):
        self.repository = repository
        self.llm = llm
        self.minio_client = minio_client
        self.semaphore = asyncio.Semaphore(10)  # Limita le chiamate LLM contemporanee

    async def perform_clustering(self, level: int = 0, namespace: str = "default", index_name: Optional[str] = None, engine_name: Optional[str] = None, engine_type: Optional[str] = None):
        logger.info("Fetching graph data from Neo4j...")
        graph_data = self.repository.get_all_nodes_and_relationships(namespace=namespace, index_name=index_name, engine_name=engine_name, engine_type=engine_type)

        # Trasformiamo in DataFrame per usare DuckDB come motore di supporto
        df_nodes = pd.DataFrame(graph_data["nodes"])
        df_rels = pd.DataFrame(graph_data["relationships"])
        
        if df_nodes.empty:
            logger.warning("No nodes found for clustering.")
            return {"status": "empty", "reports_created": 0, "reports": []}

        # 1. Community Detection (NetworkX va bene per grafi medi)
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        G = nx.Graph()
        for _, row in df_nodes.iterrows():
            G.add_node(row['id'], **row.get('properties', {}))
        for _, row in df_rels.iterrows():
            G.add_edge(row['source_id'], row['target_id'], type=row['type'])

        communities_list = louvain_communities(G, seed=42)

        # 2. Generazione Report in PARALLELO
        tasks = []
        for i, community_nodes in enumerate(communities_list):
            if len(community_nodes) < 3: continue
            tasks.append(self._process_community(i, list(community_nodes), G, level))

        # Esegue tutto e raccoglie i risultati
        all_reports = await asyncio.gather(*tasks)
        # Filtra i None (errori)
        valid_reports = [r for r in all_reports if r is not None]

        # 3. SALVATAGGIO SU NEO4J E PARQUET
        if valid_reports:
            # Save to Neo4j
            for report in valid_reports:
                try:
                    self.repository.save_community_report(
                        community_id=report["community_id"],
                        report=report,
                        level=level,
                        namespace=namespace,
                        index_name=index_name,
                        engine_name=None,
                        engine_type=None,
                        metadata_id=None
                    )
                except Exception as e:
                    logger.error(f"Failed to save report for community {report.get('community_id')} to Neo4j: {e}")

            # Save to Parquet (Local/MinIO) handled by caller or here
            # We return the reports so the caller (CommunityGraphService) can handle Parquet/MinIO centrally
            # await self._save_to_parquet(valid_reports, level, namespace) 

        return {
            "status": "success", 
            "reports_created": len(valid_reports),
            "reports": valid_reports,
            "communities_detected": len(communities_list)
        }

    async def perform_clustering_leiden(self, level: int = 0, namespace: str = "default", index_name: Optional[str] = None, engine_name: Optional[str] = None, engine_type: Optional[str] = None, resolution: float = 1.0):
        """
        Esegue il clustering usando Leiden (via igraph) per una maggiore efficienza e modularità.
        Supporta 'resolution' per il clustering gerarchico (1.0 = fine/specifico, <1.0 = coarse/generale).
        """
        if not IGRAPH_AVAILABLE:
            raise ImportError("igraph library is not installed. Please install it to use Leiden clustering.")

        logger.info(f"Starting Leiden clustering for namespace: {namespace}, level: {level}, resolution: {resolution}")

        # 1. Fetch dei dati (raw dicts, no Pandas overhead)
        graph_data = self.repository.get_all_nodes_and_relationships(namespace=namespace, index_name=index_name, engine_name=engine_name, engine_type=engine_type)
        nodes = graph_data["nodes"]
        rels = graph_data["relationships"]

        if not nodes:
             logger.warning("No nodes found for clustering.")
             return {"status": "empty", "reports_created": 0, "reports": []}

        # 2. Algoritmo Leiden con igraph
        # Creiamo una mappatura ID -> Indice (igraph usa indici interi 0..N)
        id_map = {node['id']: i for i, node in enumerate(nodes)}
        
        # Lista di tuple (int, int) per gli archi
        # Filtriamo archi che potrebbero puntare a nodi non recuperati
        edge_list = []
        for e in rels:
            if e['source_id'] in id_map and e['target_id'] in id_map:
                edge_list.append((id_map[e['source_id']], id_map[e['target_id']]))
        
        # Crea il grafo igraph
        g = ig.Graph(len(nodes), edge_list)
        
        # Salviamo le proprietà nel grafo igraph
        g.vs["original_id"] = [n['id'] for n in nodes]
        
        descriptions = []
        names = []
        for n in nodes:
            props = n.get('properties', {})
            descriptions.append(props.get('description', ''))
            names.append(props.get('name', ''))
        
        g.vs["description"] = descriptions
        g.vs["name"] = names
        
        # Edge types
        edge_types = []
        for e in rels:
             if e['source_id'] in id_map and e['target_id'] in id_map:
                 edge_types.append(e.get('type', 'RELATED'))
        g.es["type"] = edge_types

        # Esegui Leiden con resolution parameter
        # objective_function="modularity" è lo standard, resolution_parameter controlla la granularità
        partition = g.community_leiden(
            objective_function="modularity", 
            resolution_parameter=resolution
        )
        
        logger.info(f"Leiden detected {len(partition)} communities at level {level} (res={resolution}).")

        # 3. Creazione Report parallela (Map-Reduce)
        tasks = []
        # partition è una lista di liste di indici [ [0,1,2], [3,4] ... ]
        for comm_idx, node_indices in enumerate(partition):
            if len(node_indices) < 3: continue # Ignora comunità troppo piccole
            
            # Use a simpler ID for community to avoid massive IDs in high levels
            # But ensure uniqueness across levels if merging later: f"L{level}_C{comm_idx}"
            community_uid = f"L{level}_C{comm_idx}_{datetime.now().strftime('%H%M%S')}"
            
            # Passiamo l'indice della comunità, gli indici dei nodi e l'oggetto grafo
            tasks.append(self._process_community_igraph(community_uid, node_indices, g, level))
        
        all_reports = await asyncio.gather(*tasks)
        valid_reports = [r for r in all_reports if r]

        # 4. Salvataggio su Neo4j (e ritorno per Parquet esterno)
        if valid_reports:
            for report in valid_reports:
                try:
                    self.repository.save_community_report(
                        community_id=report["community_id"],
                        report=report,
                        level=level,
                        namespace=namespace,
                        index_name=index_name,
                        engine_name=None,
                        engine_type=None,
                        metadata_id=None
                    )
                except Exception as e:
                    logger.error(f"Failed to save report {report.get('community_id')} to Neo4j: {e}")

        return {
            "status": "success", 
            "reports_created": len(valid_reports),
            "reports": valid_reports,
            "communities_detected": len(partition)
        }

    async def _process_community(self, comm_id, nodes, G, level):
        """Wrapper con semaforo per gestire la concorrenza"""
        async with self.semaphore:
            return await self._generate_community_report(comm_id, nodes, G, level)
    
    async def _process_community_igraph(self, comm_id, node_indices, g, level):
        """Wrapper con semaforo per igraph"""
        async with self.semaphore:
            return await self._generate_community_report_igraph(comm_id, node_indices, g, level)

    async def _generate_community_report(self, comm_id, nodes, G, level, target_language="the same language as the source text"):
        # Usiamo DuckDB per estrarre velocemente i dati del sottografo
        # (Qui DuckDB è utile se avessimo migliaia di righe, ma lo usiamo per coerenza)
        subgraph = G.subgraph(nodes)

        # Preparazione dati per il Prompt (Input Inglese)
        entities_str = "\n".join([f"- {n}: {G.nodes[n].get('properties', {}).get('description', 'N/A')}" for n in nodes])
        rels_str = "\n".join([f"- {u} -> {v} [{d.get('type')}]" for u, v, d in subgraph.edges(data=True)])

        prompt = f"""
            Sei un analista esperto di grafi. Analizza la seguente comunità di entità e relazioni.

            DATI DI INPUT:
            ENTITÀ: {entities_str}
            RELAZIONI: {rels_str}

            ISTRUZIONI:
            1. Analizza i dati forniti (che potrebbero essere in inglese o altre lingue).
            2. Genera un report sintetico e professionale.
            3. Il report DEVE essere scritto interamente in {target_language}.

            RESTITUISCI UN JSON CON QUESTA STRUTTURA:
            {{
                "title": "Titolo in {target_language}",
                "summary": "Riassunto in {target_language}",
                "findings": ["Punto 1 in {target_language}", "Punto 2 in {target_language}"],
                "rating": 4.5,
                "rating_explanation": "Spiegazione del rating"
            }}
            """

        try:
            # Chiamata LLM (LangChain ainvoke)
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            report_content = self._parse_json(content)

            # Aggiungiamo metadati strutturali per il file Parquet
            return {
                "community_id": str(comm_id),
                "level": level,
                "title": report_content.get("title", f"Community {comm_id}"),
                "summary": report_content.get("summary", "No summary provided"),
                "findings": report_content.get("findings", []),
                "rating": report_content.get("rating", 0.0),
                "rating_explanation": report_content.get("rating_explanation", ""),
                "full_report": str(report_content),  # Full JSON as string or formatted text
                "entities": nodes, # Store entity IDs belonging to this community
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Errore community {comm_id}: {e}")
            return None

    async def _generate_community_report_igraph(self, comm_id, node_indices, g, level, target_language="the same language as the source text"):
        """
        Genera il report per una comunità definita da indici igraph.
        """
        # Estrai sottografo per la comunità
        subgraph = g.subgraph(node_indices)
        
        # Recupera attributi dai nodi del sottografo
        # (igraph mantiene gli attributi nel sottografo)
        node_descriptions = []
        original_ids = []
        
        for v in subgraph.vs:
            name = v["name"] if v["name"] else "Unnamed"
            desc = v["description"] if v["description"] else "N/A"
            node_descriptions.append(f"- {name}: {desc}")
            original_ids.append(v["original_id"])
            
        entities_str = "\n".join(node_descriptions)
        
        # Recupera relazioni nel sottografo
        rels_str_list = []
        for e in subgraph.es:
            source_idx = e.source
            target_idx = e.target
            # Ottieni nomi dei nodi sorgente/destinazione nel sottografo
            source_name = subgraph.vs[source_idx]["name"]
            target_name = subgraph.vs[target_idx]["name"]
            rel_type = e["type"]
            rels_str_list.append(f"- {source_name} -> {target_name} [{rel_type}]")
            
        rels_str = "\n".join(rels_str_list)

        prompt = f"""
            Sei un analista esperto di grafi. Analizza la seguente comunità di entità e relazioni.

            DATI DI INPUT:
            ENTITÀ: {entities_str}
            RELAZIONI: {rels_str}

            ISTRUZIONI:
            1. Analizza i dati forniti (che potrebbero essere in inglese o altre lingue).
            2. Genera un report sintetico e professionale.
            3. Il report DEVE essere scritto interamente in {target_language}.

            RESTITUISCI UN JSON CON QUESTA STRUTTURA:
            {{
                "title": "Titolo in {target_language}",
                "summary": "Riassunto in {target_language}",
                "findings": ["Punto 1 in {target_language}", "Punto 2 in {target_language}"],
                "rating": 4.5,
                "rating_explanation": "Spiegazione del rating"
            }}
            """

        try:
            # Chiamata LLM (LangChain ainvoke)
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            report_content = self._parse_json(content)

            # Aggiungiamo metadati strutturali
            return {
                "community_id": str(comm_id),
                "level": level,
                "title": report_content.get("title", f"Community {comm_id}"),
                "summary": report_content.get("summary", "No summary provided"),
                "findings": report_content.get("findings", []),
                "rating": report_content.get("rating", 0.0),
                "rating_explanation": report_content.get("rating_explanation", ""),
                "full_report": str(report_content),
                "entities": original_ids, # Store original entity IDs
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Errore community {comm_id} (igraph): {e}")
            return None

    async def _save_to_parquet(self, reports: List[Dict], level: int, namespace: str):
        """Salva i report in formato Parquet e li carica su MinIO"""
        df = pd.DataFrame(reports)

        # Convertiamo in tabella PyArrow (più efficiente)
        table = pa.Table.from_pandas(df)

        # Buffer in memoria
        buf = BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)

        # Upload su MinIO
        file_name = f"reports/level_{level}_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet"
        # self.minio_client.put_object("graphrag", file_name, buf, len(buf.getvalue()))

        # Se vuoi salvarlo in locale per ora:
        with open(f"community_reports_level_{level}.parquet", "wb") as f:
            f.write(buf.getvalue())

        logger.info(f"Report salvati in Parquet: {file_name}")

    def _parse_json(self, text):
        # Logica di pulizia JSON (già presente nel tuo codice)
        import json, re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group()) if match else {}