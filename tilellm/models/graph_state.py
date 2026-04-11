from enum import Enum
from typing import TypedDict, List, Annotated, Optional, Any
from pydantic import BaseModel, Field

from tilellm.models import QuestionAnswer
from tilellm.models.schemas import RetrievalResult


class ValidationScore(BaseModel):
    score: str = Field(description="Valutazione binaria: 'yes' o 'no'")
    explanation: str = Field(description="Se 'no', spiega sinteticamente cosa c'è di sbagliato o mancante")

class NodeStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"

class GraphState(TypedDict):
    # Input originale
    question_answer: QuestionAnswer
    # Output del tuo RAG esistente
    retrieval_result: Optional[RetrievalResult]
    # Variabili di controllo per i Guardrail
    is_on_topic: str
    is_grounded: str

    # Loop check
    max_retries: int
    retry_count: int  # Per evitare loop infiniti
    error_message: Optional[str]

    # Configurazione
    metadata: Optional[dict[str, Any]]

    # Intent routing (v2 qa agent)
    intent: Optional[str]           # "qa" | "compliance"
    parsed_csv: Optional[str]       # requisiti estratti dal messaggio (CSV grezzo)
    compliance_domain: Optional[str]  # dominio rilevato: e_procurement | medical_devices | …
    compliance_report: Optional[dict]  # ComplianceReport.model_dump() dopo il check

    # HyDE (Hypothetical Document Embeddings)
    hyde_document: Optional[str]    # documento ipotetico generato da HyDE per il retrieval

    # Semantic cache
    cache_hit: Optional[bool]           # True se la risposta è stata servita dalla cache
    cache_similarity: Optional[float]   # cosine score del hit (1.0 per exact match)
