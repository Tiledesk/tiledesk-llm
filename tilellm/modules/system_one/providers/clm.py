from tilellm.modules.system_one.providers.base import ProviderSpec, register_provider

register_provider(ProviderSpec(
    name="clm",
    default_url=None,
    default_model="clm-latest",
    requires_api_key=False,
    description=("Contrastive-LM via clm-serve (self-hosted, Qwen3-8B encoder on vLLM). "
                 "model = clm-latest or the stem of a fine-tuned head served by clm-serve."),
))
