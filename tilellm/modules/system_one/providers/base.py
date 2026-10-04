"""
System One provider registry.

Every provider speaks the same wire protocol (POST {url}/v1/systemone with state +
typed questions, answers with calibrated probabilities), so a provider is a set of
defaults, not a client: where it lives, which model to ask for when the caller names
none, whether it needs an API key. A new server (Von, LitJev, ...) is added by
registering a ProviderSpec — nothing else changes.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


class ProviderResponseError(Exception):
    """The provider answered, but not what was asked (e.g. another model than requested)."""


# (requested model or None, raw provider payload) -> warnings; raises ProviderResponseError
ResponseCheck = Callable[[Optional[str], dict], List[str]]


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    default_url: Optional[str]      # None: self-hosted, the caller must give the url
    default_model: Optional[str]    # None: the server picks its own default
    requires_api_key: bool
    description: str = ""
    # Provider-specific checks on a successful response (open/closed: the service calls
    # it without knowing the provider).
    check_response: Optional[ResponseCheck] = None

    def public(self) -> dict:
        return {"name": self.name, "default_url": self.default_url, "default_model": self.default_model,
                "requires_api_key": self.requires_api_key, "description": self.description}


_REGISTRY: Dict[str, ProviderSpec] = {}


def register_provider(spec: ProviderSpec) -> None:
    _REGISTRY[spec.name] = spec


def get_provider(name: str) -> Optional[ProviderSpec]:
    return _REGISTRY.get(name)


def list_providers() -> List[ProviderSpec]:
    return sorted(_REGISTRY.values(), key=lambda s: s.name)
