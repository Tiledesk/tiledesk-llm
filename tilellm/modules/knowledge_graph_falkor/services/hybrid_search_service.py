import asyncio
import duckdb
import pyarrow as pa
from typing import Dict, Any, List


class HybridSearchService:
    def __init__(self, vector_repo, graph_repo, llm, reports_parquet_path):
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo
        self.llm = llm
        self.reports_path = reports_parquet_path
        self.db = duckdb.connect(':memory:')
        self._setup_duckdb()

    def _setup_duckdb(self):
        self.db.execute("INSTALL httpfs; LOAD httpfs;")

    async def hybrid_query(self, question: str, namespace: str, user_lang: str = "Italiano"):
        # 1. Eseguiamo le ricerche in PARALLELO
        # - Ricerca vettoriale (Local)
        # - Ricerca sui report delle comunità (Global) via DuckDB
        local_task = self.vector_repo.search(question, namespace=namespace, limit=10)

        # 2. Global Search via DuckDB (FTS o Keyword)
        # Cerchiamo i report che menzionano i concetti chiave
        global_task = self._get_relevant_reports(question)

        local_chunks, community_reports = await asyncio.gather(local_task, global_task)

        # 3. GRAPH EXPANSION (Neo4j)
        # Per ogni chunk trovato, prendiamo i nodi correlati per avere contesto strutturale
        entities = [chunk.metadata.get("entity_id") for chunk in local_chunks if chunk.metadata.get("entity_id")]
        graph_context = []
        if entities:
            graph_context = self.graph_repo.get_neighbors(entities)

        # 4. SINTESI FINALE (LLM)
        # Uniamo i pezzi: Chunk testuali + Relazioni Grafo + Report Comunità
        answer = await self._generate_hybrid_answer(
            question, local_chunks, graph_context, community_reports, user_lang
        )

        return {
            "answer": answer,
            "sources": {
                "vector_chunks": len(local_chunks),
                "graph_triples": len(graph_context),
                "community_reports": len(community_reports)
            }
        }

    async def _get_relevant_reports(self, question: str) -> List[Dict]:
        """Usa DuckDB per trovare report rilevanti senza caricare tutto in RAM."""
        # Esempio di query SQL su Parquet con filtro semantico o keyword
        query = f"""
            SELECT title, summary, level 
            FROM read_parquet('{self.reports_path}')
            WHERE summary ILIKE ? OR title ILIKE ?
            ORDER BY level DESC LIMIT 5
        """
        keyword = f"%{question[:15]}%"
        # Fetch directly as Python dictionaries
        return self.db.execute(query, [keyword, keyword]).fetchall()

    async def _generate_hybrid_answer(self, question, chunks, graph, reports, lang):
        # Costruiamo il contesto unificato
        context_parts = []

        # Dettagli dai Chunk
        context_parts.append("DETTAGLI SPECIFICI (LOCAL):")
        for c in chunks: context_parts.append(f"- {c.page_content}")

        # Relazioni dal Grafo
        context_parts.append("\nRELAZIONI STRUTTURALI:")
        for g in graph: context_parts.append(f"- {g['source']} --{g['type']}--> {g['target']}")

        # Sintesi dai Report
        context_parts.append("\nPROSPETTIVA GLOBALE (COMMUNITIES):")
        for r in reports: context_parts.append(f"- {r[0]}: {r[1]}")

        prompt = f"""
        Domanda: {question}
        Lingua di risposta: {lang}

        Utilizza le informazioni 'Local' per i dettagli precisi e le informazioni 'Global' per il contesto ampio.
        Fornisci una risposta accurata e ben strutturata in {lang}.

        CONTESTO:
        {" ".join(context_parts)}
        """

        response = await self.llm.ainvoke(prompt)
        return response.content