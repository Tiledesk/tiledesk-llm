"""
DOCX image-aware ingestion pipeline.

process_docx_with_images() mirrors the structure of
pdf_ocr.logic.process_pdf_document_with_embeddings() but targets DOCX files:

  1. Load text + tables  (StructuredDocxLoader.load_with_images)
  2. Extract embedded images  (StructuredDocxLoader._extract_images)
  3. Upload images to MinIO   (async, run_in_executor)
  4. Generate LLM captions    (pdf_ocr.logic.generate_image_caption)
  5. Populate ref_images in adjacent paragraph metadata
  6. Index text + tables + image captions to vector store
     (same skip_delete pattern as pdf_ocr)
"""

import asyncio
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document

from tilellm.models.chunk_metadata import CommonChunkMetadata
from tilellm.models.llm import TEIConfig
from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingRequest
from tilellm.shared.situated_context import enrich_chunks_with_situated_context
from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async
from tilellm.tools.document_tools import _extract_file_name
from tilellm.tools.structured_loaders import StructuredDocxLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MinIO helper (optional — graceful degradation when MinIO is unavailable)
# ---------------------------------------------------------------------------

def _get_minio_service():
    """Return MinIOStorageService if configured, else None."""
    try:
        from tilellm.modules.knowledge_graph.services.minio_storage import (
            get_minio_storage_service,
        )
        return get_minio_storage_service()
    except Exception as exc:
        logger.warning("MinIO not available for DOCX image upload: %s", exc)
        return None


async def _upload_image(minio_service, doc_id: str, image_id: str,
                        image_bytes: bytes, content_type: str) -> str:
    """Upload raw image bytes to MinIO bucket_images.  Returns object path."""
    if not minio_service:
        return ""
    ext = "jpg" if "jpeg" in content_type.lower() else "png"
    object_path = f"{doc_id}/docx_images/{image_id}.{ext}"
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: minio_service.upload_data(
                bucket_name=minio_service.bucket_images,
                object_name=object_path,
                data=image_bytes,
                content_type=content_type,
            ),
        )
        logger.debug("Uploaded DOCX image to MinIO: %s/%s", minio_service.bucket_images, object_path)
    except Exception as exc:
        logger.error("MinIO upload failed for %s: %s", image_id, exc)
        return ""
    return object_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@inject_llm_chat_async
@inject_repo_async
async def process_docx_with_images(
    question: PDFScrapingRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs,
) -> Dict[str, Any]:
    """
    DOCX ingestion with embedded-image extraction.

    Called when  type=docx  AND  use_ocr=True  via POST /api/ingestion.
    The question must be a PDFScrapingRequest (ItemSingle subclass) so the
    shared @inject_llm_chat_async / @inject_repo_async decorators work.

    Accepts all ItemSingle fields:
      - situated_context: dedicated LLM config for Contextual Retrieval
      - hybrid / sparse_encoder: passed to aadd_documents
      - tags: propagated to all chunk metadata
    """
    if repo is None:
        raise RuntimeError("Vector store repository not injected")
    if question.engine is None:
        raise ValueError("Engine configuration is required")

    doc_id = question.id
    namespace = question.namespace or "default"
    source = question.file_content or question.source or ""
    resolved_file_name = question.file_name or _extract_file_name(source) or source

    # ── 1. Load text + tables + extract image records ─────────────────────
    loader = StructuredDocxLoader(source)
    try:
        text_table_docs, image_records = loader.load_with_images()
    except Exception as exc:
        logger.error("Failed to load DOCX %s: %s", source, exc, exc_info=True)
        raise

    logger.info(
        "DOCX %s: %d text/table docs, %d embedded images",
        doc_id, len(text_table_docs), len(image_records),
    )

    # ── 2. Upload images to MinIO + generate LLM captions ─────────────────
    minio_service = _get_minio_service()
    # lazy import to avoid circular dependency
    from tilellm.modules.pdf_ocr.logic import generate_image_caption

    for img_rec in image_records:
        image_id = img_rec["image_id"]
        image_bytes: bytes = img_rec["image_bytes"]

        # Upload to MinIO (optional)
        minio_path = await _upload_image(
            minio_service, doc_id, image_id,
            image_bytes, img_rec["content_type"],
        )
        img_rec["minio_path"] = minio_path

        # Context text: alt text or adjacent paragraph text
        context_text = img_rec.get("alt_text") or ""
        if not context_text:
            para_idx = img_rec["para_index"]
            for doc_obj in text_table_docs:
                if doc_obj.metadata.get("_para_index") == para_idx:
                    context_text = doc_obj.page_content[:200]
                    break

        # Vision LLM caption
        if llm is not None:
            try:
                caption = await generate_image_caption(image_bytes, context_text, llm=llm)
            except Exception as exc:
                logger.error("Caption generation failed for %s: %s", image_id, exc)
                caption = img_rec.get("alt_text") or "Image extracted from DOCX"
        else:
            caption = img_rec.get("alt_text") or "Image extracted from DOCX"

        img_rec["caption"] = caption

    # ── 3. Build para_index → [image_ids] lookup ──────────────────────────
    # Include adjacent paragraphs (±1) so a drawing paragraph (often blank)
    # is also linked to the preceding and following text.
    para_to_images: Dict[int, List[str]] = {}
    for img_rec in image_records:
        idx = img_rec["para_index"]
        for adjacent in (idx - 1, idx, idx + 1):
            para_to_images.setdefault(adjacent, []).append(img_rec["image_id"])

    # ── 4. Enrich paragraph metadata with ref_images + standard fields ─────
    enriched_text_docs: List[Document] = []
    for doc_obj in text_table_docs:
        para_idx = doc_obj.metadata.get("_para_index")
        ref_imgs = list(dict.fromkeys(para_to_images.get(para_idx, [])))  # stable dedup

        doc_obj.metadata.update({
            "id": doc_id,
            "metadata_id": doc_id,
            "namespace": namespace,
            "source": resolved_file_name,
            "file_name": resolved_file_name,
            "ref_images": str(ref_imgs),
        })
        # Remove internal cross-ref key before indexing
        doc_obj.metadata.pop("_para_index", None)

        if question.tags:
            doc_obj.metadata["tags"] = question.tags

        enriched_text_docs.append(doc_obj)

    # ── 5. Build image caption documents ──────────────────────────────────
    image_caption_docs: List[Document] = []
    for img_rec in image_records:
        caption = img_rec.get("caption", "")
        if not caption:
            continue
        meta = CommonChunkMetadata(
            id=doc_id,
            metadata_id=doc_id,
            doc_id=doc_id,
            namespace=namespace,
            source=resolved_file_name,
            file_name=resolved_file_name,
            chunk_type="image_caption",
            type="image",
            image_id=img_rec["image_id"],
            path=img_rec.get("minio_path", ""),
            surrounding_text=img_rec.get("alt_text", ""),
            tags=question.tags if question.tags else None,
        ).to_metadata_dict()
        image_caption_docs.append(Document(page_content=caption, metadata=meta))

    # ── 6. Optional: situated context on all chunks ───────────────────────
    sc_config = getattr(question, 'situated_context', None)
    if (sc_config and sc_config.enable) and llm:
        try:
            from tilellm.shared.situated_context import build_llm_from_config
            # Use dedicated LLM for situated context if configured, otherwise fall back to DI LLM
            sc_llm = await build_llm_from_config(sc_config) or llm
            if sc_llm:
                # Extract global doc context from the first few text paragraphs
                combined_text = " ".join(d.page_content[:200] for d in enriched_text_docs[:10])
                doc_context = combined_text[:1500]

                if enriched_text_docs:
                    enriched_text_docs = await enrich_chunks_with_situated_context(
                        enriched_text_docs, sc_llm, doc_context=doc_context
                    )
                    logger.info("Situated context applied to %d DOCX text chunks", len(enriched_text_docs))
                
                if image_caption_docs:
                    image_caption_docs = await enrich_chunks_with_situated_context(
                        image_caption_docs, sc_llm, doc_context=doc_context
                    )
                    logger.info("Situated context applied to %d DOCX image caption chunks", len(image_caption_docs))

        except Exception as exc:
            logger.warning("Situated context enrichment failed: %s", exc)

    # ── 7. Index to vector store (skip_delete pattern) ────────────────────
    sparse_enc: Union[str, TEIConfig, None] = (
        question.sparse_encoder if question.hybrid else None
    )

    first_done = False

    if image_caption_docs:
        await repo.aadd_documents(
            engine=question.engine,
            documents=image_caption_docs,
            namespace=namespace,
            embedding_model=llm_embeddings,
            sparse_encoder=sparse_enc,
            metadata_id=doc_id,
            skip_delete=first_done,
        )
        first_done = True
        logger.info("Indexed %d image caption chunks for %s", len(image_caption_docs), doc_id)

    if enriched_text_docs:
        await repo.aadd_documents(
            engine=question.engine,
            documents=enriched_text_docs,
            namespace=namespace,
            embedding_model=llm_embeddings,
            sparse_encoder=sparse_enc,
            metadata_id=doc_id,
            skip_delete=first_done,
        )
        logger.info("Indexed %d text/table chunks for %s", len(enriched_text_docs), doc_id)

    return {
        "message": f"Item {doc_id} created successfully",
        "text_table_chunks": len(enriched_text_docs),
        "images_extracted": len(image_records),
        "image_chunks_indexed": len(image_caption_docs),
    }
