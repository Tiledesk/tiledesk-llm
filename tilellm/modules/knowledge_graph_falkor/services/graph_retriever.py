import asyncio
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun


class GraphHybridRetriever(BaseRetriever):
    dense_retriever: BaseRetriever
    sparse_retriever: BaseRetriever
    graph_store: Any  # Il tuo client (es. Neo4j, NetworkX, ecc.)
    k: int = 5
    rrf_k: int = 60
    alpha: float = 0.5  # Per il tuo score norm originale
    expand_hops: int = 1  # Quanti salti fare nel grafo

    def _apply_rrf(self, dense_docs: List[Document], sparse_docs: List[Document]) -> Dict[str, float]:
        """Combina i rank utilizzando RRF."""
        scores = {}
        for rank, doc in enumerate(dense_docs):
            doc_id = doc.metadata.get("id", doc.page_content)
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + self.rrf_k)

        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.metadata.get("id", doc.page_content)
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + self.rrf_k)
        return scores

    def _get_graph_context(self, seed_documents: List[Document]) -> List[Document]:
        """
        Data una lista di documenti (nodi), interroga il grafo per
        trovare nodi connessi (es. Entità, Concetti, altri Documenti).
        """
        expanded_context = []
        for doc in seed_documents:
            doc_id = doc.metadata.get("id")
            if not doc_id: continue

            # Esempio logica Grafo (pseudo-codice Cypher/Graph)
            # "MATCH (d:Document {id: $id})-[*1..$hops]-(related) RETURN related"
            neighbors = self.graph_store.get_neighbors(doc_id, hops=self.expand_hops)

            for n in neighbors:
                expanded_context.append(
                    Document(
                        page_content=n.text,
                        metadata={"source": "graph_expansion", "origin_node": doc_id}
                    )
                )
        return expanded_context

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. Retrieval
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)

        # 2. Hybrid Fusion (RRF)
        combined_scores = self._apply_rrf(dense_docs, sparse_docs)

        # 3. Ordinamento e selezione "Seed Nodes"
        all_docs_map = {doc.metadata.get("id", doc.page_content): doc for doc in dense_docs + sparse_docs}
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        seed_docs = [all_docs_map[doc_id] for doc_id in sorted_ids[:self.k]]

        # 4. Graph Traversal (Attraversata del grafo)
        graph_docs = self._get_graph_context(seed_docs)

        # Uniamo i documenti originali con il contesto del grafo
        return seed_docs + graph_docs

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Esecuzione parallela dei retriever
        dense_task = self.dense_retriever.ainvoke(query)
        sparse_task = self.sparse_retriever.ainvoke(query)
        dense_docs, sparse_docs = await asyncio.gather(dense_task, sparse_task)

        combined_scores = self._apply_rrf(dense_docs, sparse_docs)

        all_docs_map = {doc.metadata.get("id", doc.page_content): doc for doc in dense_docs + sparse_docs}
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        seed_docs = [all_docs_map[doc_id] for doc_id in sorted_ids[:self.k]]

        # Espansione grafo (async se supportato dal tuo graph_store)
        graph_docs = self._get_graph_context(seed_docs)

        return seed_docs + graph_docs