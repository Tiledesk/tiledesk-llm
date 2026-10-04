from tilellm.modules.system_one.providers.base import ProviderSpec, register_provider

register_provider(ProviderSpec(
    name="jev",
    default_url="https://api.typesafe.ai",
    default_model="jev-latest",
    requires_api_key=True,
    description="TypeSafe Jev (hosted).",
))
