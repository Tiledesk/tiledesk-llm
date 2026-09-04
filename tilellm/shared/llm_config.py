"""
Configurazione centralizzata per i parametri dei vari provider LLM.
Ogni provider ha regole specifiche su quali parametri accettare.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

from pydantic import SecretStr

# Timeout per singola richiesta LLM, in secondi.
#
# NON e' una protezione da "richiesta appesa all'infinito": la libreria openai
# ha gia' i suoi default (Timeout(connect=5s, read=600s), max_retries=2), quindi
# una connessione morta viene comunque rilevata. Verificato: durante il blackout
# di rete del 2026-07-30 il worker emetteva regolarmente APIConnectionError e
# ritentava — non era bloccato in silenzio.
#
# Serve invece a rendere il valore ESPLICITO e REGOLABILE: 600s di read timeout
# sono molto generosi per una singola chiamata di estrazione, e con i retry
# sovrapposti (openai + quelli di graphrag_extractor) una singola chunk puo'
# occupare uno slot di concorrenza per decine di minuti. 300s stringe il caso
# peggiore lasciando ampio margine alle risposte lente, ed e' tarabile via env
# senza toccare il codice.
LLM_REQUEST_TIMEOUT_S = float(os.environ.get("LLM_REQUEST_TIMEOUT_S", "300"))


@dataclass
class LLMProviderConfig:
    """Configurazione per un provider LLM specifico"""
    name: str
    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_max_tokens: bool = True
    temperature_top_p_exclusive: bool = False  # Se True, accetta solo uno dei due
    prefer_temperature: bool = True  # Quale preferire se entrambi sono settati
    custom_params: Optional[Dict[str, str]] = None  # Mappatura nomi parametri custom (es: num_predict invece di max_tokens)


# Configurazioni dei provider
PROVIDER_CONFIGS = {
    "openai": LLMProviderConfig(
        name="openai",
        supports_temperature=True,
        supports_top_p=True,
        temperature_top_p_exclusive=False
    ),
    "anthropic": LLMProviderConfig(
        name="anthropic",
        supports_temperature=True,
        supports_top_p=True,
        temperature_top_p_exclusive=True
    ),
    "cohere": LLMProviderConfig(
        name="cohere",
        supports_temperature=True,
        supports_top_p=False,  # Cohere non supporta top_p standard
        temperature_top_p_exclusive=False
    ),
    "google": LLMProviderConfig(
        name="google",
        supports_temperature=True,
        supports_top_p=True,
        temperature_top_p_exclusive=False
    ),
    "ollama": LLMProviderConfig(
        name="ollama",
        supports_temperature=True,
        supports_top_p=True,
        custom_params={"max_tokens": "num_predict"}  # Ollama usa num_predict
    ),
    "vllm": LLMProviderConfig(
        name="vllm",
        supports_temperature=True,
        supports_top_p=True,
        temperature_top_p_exclusive=False
    ),
    "groq": LLMProviderConfig(
        name="groq",
        supports_temperature=True,
        supports_top_p=True,
        temperature_top_p_exclusive=False
    ),
    "deepseek": LLMProviderConfig(
        name="deepseek",
        supports_temperature=True,
        supports_top_p=True,
        temperature_top_p_exclusive=False
    ),
    "mistralai": LLMProviderConfig(
        name="mistralai",
        supports_temperature=True,
        supports_top_p=True,
        temperature_top_p_exclusive=False
    ),
    # OpenRouter espone un'API OpenAI-compatibile: stessi parametri di sampling.
    # Il routing verso i provider a monte viaggia separatamente, in extra_body.
    "openrouter": LLMProviderConfig(
        name="openrouter",
        supports_temperature=True,
        supports_top_p=True,
        temperature_top_p_exclusive=False
    ),
}


# OpenRouter e' raggiunto tramite il client OpenAI: serve solo cambiare base_url.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Chiavi accettate dal blocco "provider" di OpenRouter che questa integrazione
# configura. Tutto il resto viene scartato: il valore arriva dal database, non
# dal codice, e non vogliamo inoltrare campi arbitrari all'API.
_OPENROUTER_ROUTING_KEYS = ("order", "allow_fallbacks", "sort", "only", "ignore")

_OPENROUTER_SORT_VALUES = ("price", "throughput", "latency")


def build_openrouter_extra_body(provider_routing: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Traduce il provider_routing salvato nell'integrazione nel corpo che
    OpenRouter si aspetta: {"provider": {...}}.

    Restituisce None quando non c'e' niente da instradare, cosi' il modello si
    comporta esattamente come un qualunque provider OpenAI-compatibile.
    """
    if not provider_routing or not isinstance(provider_routing, dict):
        return None

    routing: Dict[str, Any] = {}

    for key in _OPENROUTER_ROUTING_KEYS:
        if key not in provider_routing:
            continue
        value = provider_routing[key]
        if value is None:
            continue

        if key in ("order", "only", "ignore"):
            if not isinstance(value, (list, tuple)):
                continue
            slugs = [str(item).strip() for item in value if str(item or "").strip()]
            if slugs:
                routing[key] = slugs
        elif key == "allow_fallbacks":
            routing[key] = bool(value)
        elif key == "sort":
            sort = str(value).strip().lower()
            if sort in _OPENROUTER_SORT_VALUES:
                routing[key] = sort

    if not routing:
        return None

    return {"provider": routing}


def get_llm_params(
    provider: str,
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
    **extra_params
) -> Dict[str, Any]:
    """
    Restituisce i parametri corretti per il provider LLM specificato.

    Args:
        provider: Nome del provider (es: "openai", "cohere", etc.)
        temperature: Valore della temperatura
        top_p: Valore di top_p
        max_tokens: Numero massimo di token
        **extra_params: Parametri extra specifici del provider

    Returns:
        Dizionario con i parametri da passare al costruttore del LLM
    """
    config = PROVIDER_CONFIGS.get(provider.lower())

    # Se il provider non è configurato, usa defaults sicuri
    if config is None:
        config = LLMProviderConfig(
            name=provider,
            supports_temperature=True,
            supports_top_p=True
        )

    params = {}

    # Gestione temperature e top_p
    if config.temperature_top_p_exclusive:
        # Provider che accetta solo uno dei due parametri
        if temperature is not None and top_p is not None:
            # Usa solo quello preferito
            if config.prefer_temperature:
                params["temperature"] = temperature
            else:
                params["top_p"] = top_p
        elif temperature is not None:
            params["temperature"] = temperature
        elif top_p is not None:
            params["top_p"] = top_p
    else:
        # Provider che accetta entrambi i parametri
        if config.supports_temperature and temperature is not None:
            params["temperature"] = temperature
        if config.supports_top_p and top_p is not None:
            params["top_p"] = top_p

    # Gestione max_tokens con custom params
    if config.supports_max_tokens and max_tokens is not None:
        param_name = "max_tokens"
        if config.custom_params and "max_tokens" in config.custom_params:
            param_name = config.custom_params["max_tokens"]
        params[param_name] = max_tokens

    # Aggiungi eventuali parametri extra
    params.update(extra_params)

    # Timeout di default per OGNI provider (anche quelli non in PROVIDER_CONFIGS):
    # e' una rete di sicurezza contro le richieste appese, non un parametro
    # specifico del provider. Un timeout passato esplicitamente in extra_params
    # ha la precedenza.
    params.setdefault("timeout", LLM_REQUEST_TIMEOUT_S)

    return params


# Claude models that reject an explicit temperature/top_p outright (HTTP 400
# "temperature is deprecated for this model") regardless of value — Claude
# Opus 5, Sonnet 5, Fable 5, Opus 4.8 and Opus 4.7 removed sampling params
# from the API. PROVIDER_CONFIGS above is provider-level (all Anthropic models
# supports_temperature=True), so this is a model-level override applied at
# ChatAnthropic() construction time, once the resolved model string is known.
ANTHROPIC_NO_SAMPLING_PARAMS_MODELS = frozenset({
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
})


def strip_unsupported_anthropic_sampling_params(model: Optional[str], params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop temperature/top_p in place when `model` rejects them outright."""
    if model in ANTHROPIC_NO_SAMPLING_PARAMS_MODELS:
        params.pop("temperature", None)
        params.pop("top_p", None)
    return params


def should_include_param(provider: str, param_name: str) -> bool:
    """
    Verifica se un parametro dovrebbe essere incluso per un dato provider.

    Args:
        provider: Nome del provider
        param_name: Nome del parametro da verificare

    Returns:
        True se il parametro dovrebbe essere incluso, False altrimenti
    """
    config = PROVIDER_CONFIGS.get(provider.lower())

    if config is None:
        return True  # Default: accetta tutto

    param_map = {
        "temperature": config.supports_temperature,
        "top_p": config.supports_top_p,
        "max_tokens": config.supports_max_tokens
    }

    return param_map.get(param_name, True)


def serialize_with_secrets(obj):
    """Recursively convert Pydantic models/dicts revealing SecretStr values."""
    if isinstance(obj, SecretStr):
        return obj.get_secret_value()
    if isinstance(obj, dict):
        return {k: serialize_with_secrets(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_with_secrets(v) for v in obj]
    return obj