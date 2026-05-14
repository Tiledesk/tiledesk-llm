NODE_REGISTRY = [
    {
        "name": "guard",
        "node_id": "guardia",
        "description": (
            "Input safety guard — valida che la domanda sia sicura e on-topic "
            "tramite LLM. Blocca domande offensive o prive di senso."
        ),
        "can_disable": True,
        "default_enabled": True,
        "request_field": "use_guard",
    },
    {
        "name": "intent_router",
        "node_id": "intent_router",
        "description": (
            "Classifica l'intent come 'compliance' o 'qa' e smista il flusso. "
            "In caso di compliance estrae i requisiti in formato CSV."
        ),
        "can_disable": False,
        "default_enabled": True,
        "request_field": None,
    },
    {
        "name": "compliance",
        "node_id": "compliance_node",
        "description": (
            "Esegue il compliance check sui documenti quando l'intent è 'compliance'. "
            "Attivato automaticamente da intent_router, non esposto come flag."
        ),
        "can_disable": False,
        "default_enabled": True,
        "request_field": None,
    },
    {
        "name": "rag",
        "node_id": "rag_core",
        "description": (
            "RAG standard — retrieval vettoriale + generazione risposta con LLM. "
            "Supporta similarity, MMR e hybrid search."
        ),
        "can_disable": False,
        "default_enabled": True,
        "request_field": None,
    },
    {
        "name": "hallucination_grader",
        "node_id": "validatore",
        "description": (
            "Valida che la risposta sia supportata dal contesto recuperato. "
            "In caso negativo rilancia il RAG fino a max_retries (default 3)."
        ),
        "can_disable": True,
        "default_enabled": True,
        "request_field": "use_hallucination_grader",
    },
]
