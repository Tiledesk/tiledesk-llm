from typing import List, Optional

from tilellm.modules.system_one.providers.base import ProviderResponseError, ProviderSpec, register_provider


def check_laya_response(requested_model: Optional[str], payload: dict) -> List[str]:
    """laya-serve (0.3.27) ignores an unknown model name and answers with the base
    checkpoint chosen by language, with HTTP 200: a mistyped fine-tuned path would return
    base-model answers silently. Its "routing" block names the checkpoint that really
    answered — reject the response when it is not the requested one. It also reports
    state truncation in "usage", surfaced as a warning instead of a number nobody reads."""
    warnings = []
    usage = payload.get("usage") or {}
    dropped = usage.get("state_tokens_dropped") or 0
    if usage.get("truncated") or dropped:
        warnings.append(f"stato troncato da laya-serve: {dropped} token dello stato non letti "
                        f"(aumentare max_len in parameters)")
    if requested_model:
        routing = payload.get("routing")
        if not isinstance(routing, dict):
            warnings.append(f"modello servito non verificabile (risposta senza 'routing', es. "
                            f"LAYA_JEV_STRICT=1): richiesto '{requested_model}'")
        elif requested_model not in (routing.get("model"), routing.get("repo")):
            raise ProviderResponseError(
                f"laya-serve non serve il modello richiesto '{requested_model}': ha risposto "
                f"'{routing.get('model')}' ({routing.get('repo')}) — {routing.get('reason')}")
    return warnings


register_provider(ProviderSpec(
    name="laya",
    default_url=None,
    default_model=None,  # laya-serve picks the checkpoint by language when none is named
    requires_api_key=False,
    description=("Laya via laya-serve (self-hosted). model = checkpoint name (english, multilingual, "
                 "typed-decisions) or a fine-tuned path served by laya-serve (e.g. /models/<dir>)."),
    check_response=check_laya_response,
))
