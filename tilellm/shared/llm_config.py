"""
Configurazione centralizzata per i parametri dei vari provider LLM.
Ogni provider ha regole specifiche su quali parametri accettare.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass

from pydantic import SecretStr


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
}


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