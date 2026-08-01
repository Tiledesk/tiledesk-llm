"""
Knowledge Graph module for RAG with Neo4j.

.. deprecated::
    **DEPRECATO — NON MANTENUTO.** La feature Neo4j (router ``/api/kg``, logic,
    repository, services) è disabilitata in configurazione (``ENABLE_GRAPHRAG=false``,
    ``'graphrag': False`` in ``register_feature_routers``) e non riceve più fix né
    nuove funzionalità.

    **Usare al suo posto** ``tilellm.modules.knowledge_graph_falkor`` (FalkorDB,
    router ``/api/kg-falkor``), che è il modulo attivo. I due sono copie parallele
    quasi omonime: modificare per sbaglio questo invece di quello è un errore già
    avvenuto più volte — verificare sempre il log di ``register_feature_routers``
    prima di editare.

    **Stato: pronto per la cancellazione.** Tutto ciò che era infrastruttura
    condivisa (nessuna riga di Neo4j, finita qui per accidente storico) è già
    stato estratto ai suoi posti naturali::

        models.models (Node/Relationship/+Update) → tilellm.models.graph
        models.schemas.TaskPollResponse           → tilellm.models.schemas.general_schemas
        utils.rrf                                 → tilellm.shared.rrf
        services.minio_storage                    → tilellm.shared.minio_storage

    Quello che resta qui è genuinamente legato a Neo4j. Cancellando il package
    NON si rompe né l'avvio dell'app né il worker, perché ogni residuo è o
    dentro ``try/except`` o importato a livello di funzione::

        repository.repository.GraphRepository  → pdf_ocr (tutti try/except:
            l'arricchimento-grafo Neo4j si autodisabilita)
        tools.graphrag_extractor               → pdf_ocr (try/except)
        logic + models.schemas (Graph*Request)  → task_executor, nei 6 task legacy
            ``neo4j_graph_create``/``*_cluster``: import a livello di funzione,
            quindi falliscono solo se il task viene invocato — e il loro unico
            dispatcher è il router ``/api/kg`` disabilitato. Da rimuovere insieme
            al package.

    Nessun ``DeprecationWarning`` a runtime: i pochi import rimasti sono
    legittimi e guardati, un warning sarebbe solo rumore.

    Il test ``tests/unit/test_shared_graph_primitives_extraction.py`` verifica
    che nessun modulo attivo torni a dipendere da qui per le primitive estratte.
"""

from .controllers import router
from .models import Node, NodeUpdate, Relationship, RelationshipUpdate
from .services import GraphService
#from .repository.repository import GraphRepository

__all__ = [
    "router",
    "Node",
    "NodeUpdate",
    "Relationship",
    "RelationshipUpdate",
    "GraphService"
#    "GraphRepository"
]
