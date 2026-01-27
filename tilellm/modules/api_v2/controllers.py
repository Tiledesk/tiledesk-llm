"""
FastAPI Controllers for Knowledge Graph API endpoints.
Provides RESTful API for managing nodes and relationships in Neo4j.
"""

import logging
from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Union

from tilellm.agents.workflow import app
from tilellm.models import QuestionAnswer
from tilellm.models.schemas import RetrievalResult

# 1. Crea il router per questo modulo
router = APIRouter(
    prefix="/api/v2",
    tags=["Agentic API v2 "] # Tag per la documentazione OpenAPI (Swagger)
)


@router.post("/qa", response_model=RetrievalResult)
async def ask_question(payload: QuestionAnswer):
    # Inizializziamo lo stato con l'oggetto Pydantic ricevuto
    initial_state = {
        "question_answer": payload,
        "retry_count": 0,
        "max_retries": 3,
        "metadata": {
            "search_type": payload.search_type,
            "trace": []}
    }

    # Eseguiamo il grafo asincronamente
    final_state = await app.ainvoke(initial_state)

    # Gestione del caso "fuori tema"
    if final_state.get("is_on_topic") == "no":
        raise HTTPException(status_code=400, detail="Domanda non pertinente o non sicura.")

    # Restituiamo direttamente l'oggetto RetrievalResult prodotto dal nodo rag_core
    return final_state["retrieval_result"]