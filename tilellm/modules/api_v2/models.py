from pydantic import Field

from tilellm.models import QuestionAnswer
from tilellm.models.llm import TEIConfig, PineconeRerankerConfig
from tilellm.models.chat import ChatEntry


class QASimpleRequest(QuestionAnswer):
    """
    Estende QuestionAnswer con flag per abilitare/disabilitare nodi opzionali
    nel grafo semplificato (guardia + intent_router + compliance/rag + validatore).
    """
    use_guard: bool = Field(
        True,
        description=(
            "Abilita il nodo guardia (input safety guard). "
            "Se False il controllo viene saltato e la domanda è considerata on-topic."
        ),
    )
    use_hallucination_grader: bool = Field(
        True,
        description=(
            "Abilita il validatore allucinazioni. "
            "Se False la risposta RAG viene restituita senza verifica di grounding."
        ),
    )


QASimpleRequest.model_rebuild(_types_namespace={
    "TEIConfig": TEIConfig,
    "PineconeRerankerConfig": PineconeRerankerConfig,
    "ChatEntry": ChatEntry,
})
