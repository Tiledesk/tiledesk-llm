from typing import Callable, Any, List
import threading

from langchain_core.embeddings import Embeddings


def _is_session_closed_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("session is closed" in msg
            or "client is closed" in msg
            or "event loop is closed" in msg
            or "closed pool" in msg
            )

class ResilientEmbeddings(Embeddings):
    """
    Wrapper per LangChain Embeddings:
    - intercetta errori di sessione chiusa e ricrea l'oggetto interno
    - embed_documents / embed_query restano metodi sincroni
    """
    def __init__(self, builder: Callable[[], Any], seed: Any = None, dimension: int = None):
        self._builder = builder     # deve ritornare un nuovo oggetto Embeddings
        self._inner = seed          # eventuale oggetto già costruito
        self._lock = threading.RLock()
        self.dimension = dimension  # embedding dimension for vector store configuration

    def _ensure(self):
        if self._inner is None:
            self._inner = self._builder()

    def _rebuild(self):
        self._inner = self._builder()

    def embed_documents(self, texts: List[str], **kwargs):
        self._ensure()
        try:
            return self._inner.embed_documents(texts, **kwargs)
        except Exception as e:
            if _is_session_closed_error(e):
                with self._lock:
                    self._rebuild()
                return self._inner.embed_documents(texts, **kwargs)
            raise

    def embed_query(self, text: str, **kwargs):
        self._ensure()
        try:
            return self._inner.embed_query(text, **kwargs)
        except Exception as e:
            if _is_session_closed_error(e):
                with self._lock:
                    self._rebuild()
                return self._inner.embed_query(text, **kwargs)
            raise

    # opzionale: close allo shutdown
    async def aclose(self):
        inner = self._inner
        if inner is None:
            return
        try:
            client = getattr(inner, "client", None) or getattr(inner, "_client", None)
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if callable(close):
                res = close()
                if hasattr(res, "__await__"):
                    await res
        except Exception:
            pass