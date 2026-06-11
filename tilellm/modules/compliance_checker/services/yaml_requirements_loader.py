"""
YamlRequirementsLoader — downloads and parses a TenderLotRequirements YAML.

Accepts either:
  - yaml_inline : YAML string passed directly in the request body
  - yaml_url    : URL to a YAML file (max 1 MB)

Exactly one must be non-empty (validated upstream by ComplianceRequestV2).
"""
import logging
from typing import Optional

import httpx

from tilellm.modules.compliance_checker.models_v2 import TenderLotRequirements

logger = logging.getLogger(__name__)


class YamlRequirementsLoader:
    MAX_YAML_SIZE: int = 1 * 1024 * 1024  # 1 MB

    async def load(
        self,
        yaml_inline: Optional[str],
        yaml_url: Optional[str],
    ) -> TenderLotRequirements:
        """
        Return a parsed TenderLotRequirements from either inline YAML or a URL.

        Raises:
            ValueError: neither source provided, file too large, malformed YAML,
                        or YAML schema invalid.
            httpx.HTTPError: network / HTTP errors when fetching the URL.
        """
        if yaml_inline and yaml_inline.strip():
            return TenderLotRequirements.from_yaml(yaml_inline)

        if yaml_url and yaml_url.strip():
            return await self._load_from_url(yaml_url)

        raise ValueError(
            "Fornire 'requirements_yaml' (YAML inline) oppure "
            "'requirements_yaml_url' (URL al documento YAML)."
        )

    async def _load_from_url(self, url: str) -> TenderLotRequirements:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
            if len(content) > self.MAX_YAML_SIZE:
                raise ValueError(
                    f"Il file YAML è troppo grande "
                    f"({len(content)} byte > {self.MAX_YAML_SIZE} byte massimi)."
                )
            yaml_text = content.decode("utf-8", errors="replace")
        logger.debug("YamlRequirementsLoader: downloaded %d bytes from %s", len(content), url)
        return TenderLotRequirements.from_yaml(yaml_text)
