from typing import Any, Dict, Optional

# Configurazione esternalizzata - può stare in un file separato (es. config.py)
ENGINE_CONTENT_PROPERTIES: Dict[str, Dict[str, str]] = {
    'pinecone': {
        'pod': 'content',
        'serverless': 'text',
    },
    'qdrant': {
        '_default': 'page_content',  # _default per engine senza type
    },
    'weaviate': {  # Esempio futuro engine
        '_default': 'content',
    }
}


class EngineConfig:
    """Classe base per la configurazione dell'engine"""

    def __init__(self, name: str, type: Optional[str] = None):
        self.name = name
        self.type = type


def get_content_property(engine: Any) -> str:
    """
    Recupera il nome della proprietà contenuto in base all'engine.

    Args:
        engine: Oggetto con attributi 'name' e 'type'

    Returns:
        str: Nome della proprietà da usare

    Raises:
        ValueError: Se la configurazione dell'engine non è trovata
    """
    engine_name = engine.name.lower()
    engine_type = getattr(engine, 'type', None)

    # Verifica se l'engine esiste nella configurazione
    if engine_name not in ENGINE_CONTENT_PROPERTIES:
        raise ValueError(f"Engine '{engine_name}' non supportato. "
                         f"Engine disponibili: {list(ENGINE_CONTENT_PROPERTIES.keys())}")

    config = ENGINE_CONTENT_PROPERTIES[engine_name]

    # Se l'engine ha tipi specifici
    if engine_type:
        engine_type = engine_type.lower()
        if engine_type in config:
            return config[engine_type]
        else:
            # Fallback al default se esiste, altrimenti errore
            if '_default' in config:
                return config['_default']
            raise ValueError(f"Type '{engine_type}' non supportato per engine '{engine_name}'. "
                             f"Tipi disponibili: {[k for k in config.keys() if k != '_default']}")

    # Se non c'è type, usa il default
    if '_default' in config:
        return config['_default']

    # Se non c'è default e non ci sono type definiti, errore
    raise ValueError(f"Engine '{engine_name}' richiede un type. "
                     f"Tipi disponibili: {[k for k in config.keys() if k != '_default']}")


def get_content_data(engine: Any, data_source: Dict[str, Any]) -> Any:
    """
    Funzione helper per estrarre direttamente il contenuto.

    Args:
        engine: Configurazione dell'engine
        data_source: Dizionario o oggetto con i dati

    Returns:
        Il contenuto estratto o None se non trovato
    """
    property_name = get_content_property(engine)
    return data_source.get(property_name) if isinstance(data_source, dict) else getattr(data_source, property_name,
                                                                                        None)