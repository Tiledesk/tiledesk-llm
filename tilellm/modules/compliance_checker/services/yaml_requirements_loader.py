"""
YamlRequirementsLoader — downloads and parses a TenderLotRequirements.

Accepts one of:
  - yaml_inline : YAML string passed directly in the request body
  - yaml_url    : URL to a YAML file (max 1 MB)
  - xlsx_url    : URL to a standardized requirements workbook (max 10 MB);
                  parsed deterministically (no LLM) via RequirementsXlsxService,
                  then the lot is selected via *lot_id* (required if multi-lot).

Exactly one must be non-empty (validated upstream by ComplianceRequestV2).
"""
import logging
from typing import Optional

import httpx

from tilellm.modules.compliance_checker.models_v2 import TenderLotRequirements
from tilellm.modules.compliance_checker.services.requirements_xlsx_service import (
    MAX_XLSX_SIZE,
    RequirementsXlsxService,
    select_lot,
)

logger = logging.getLogger(__name__)


class YamlRequirementsLoader:
    MAX_YAML_SIZE: int = 1 * 1024 * 1024  # 1 MB

    async def load(
        self,
        yaml_inline: Optional[str],
        yaml_url: Optional[str],
        xlsx_url: Optional[str] = None,
        lot_id: Optional[str] = None,
    ) -> TenderLotRequirements:
        """
        Return a parsed TenderLotRequirements from inline YAML, a YAML URL or an
        xlsx URL.

        Raises:
            ValueError: no source provided, file too large, malformed YAML/xlsx,
                        schema invalid, or ambiguous lot selection.
            httpx.HTTPError: network / HTTP errors when fetching the URL.
        """
        if yaml_inline and yaml_inline.strip():
            return TenderLotRequirements.from_yaml(yaml_inline)

        if yaml_url and yaml_url.strip():
            return await self._load_from_url(yaml_url)

        if xlsx_url and xlsx_url.strip():
            return await self._load_from_xlsx_url(xlsx_url, lot_id)

        raise ValueError(
            "Fornire 'requirements_yaml' (YAML inline), 'requirements_yaml_url' "
            "(URL YAML) oppure 'requirements_xlsx_url' (URL workbook xlsx)."
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

    async def _load_from_xlsx_url(
        self, url: str, lot_id: Optional[str]
    ) -> TenderLotRequirements:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
            if len(content) > MAX_XLSX_SIZE:
                raise ValueError(
                    f"Il file xlsx è troppo grande "
                    f"({len(content)} byte > {MAX_XLSX_SIZE} byte massimi)."
                )
        logger.debug("YamlRequirementsLoader: downloaded %d xlsx bytes from %s", len(content), url)
        lots = RequirementsXlsxService().parse_workbook(content)
        return select_lot(lots, lot_id)


async def export_requirements_xlsx(
    yaml_inline: Optional[str] = None,
    yaml_url: Optional[str] = None,
    xlsx_url: Optional[str] = None,
    lot_id: Optional[str] = None,
) -> bytes:
    """
    Load a requirements document (YAML inline/URL, or a standardized xlsx URL) and
    serialize it back into the standardized review workbook (bytes).

    Deterministic, no LLM — intended for the "give the business operator an xlsx to
    review" use case starting from an existing requirements YAML.
    """
    lot = await YamlRequirementsLoader().load(
        yaml_inline=yaml_inline, yaml_url=yaml_url, xlsx_url=xlsx_url, lot_id=lot_id,
    )
    return RequirementsXlsxService().build_workbook([lot])

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
