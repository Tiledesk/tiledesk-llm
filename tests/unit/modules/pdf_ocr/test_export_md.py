"""
Part A — export_md: persist the assembled markdown to MinIO when the flag
is set. The helper must never block ingestion: a storage failure returns
None instead of raising.
"""

from tilellm.modules.pdf_ocr.logic import _export_markdown_to_minio

MINIO_FACTORY = (
    "tilellm.modules.knowledge_graph.services.minio_storage.get_minio_storage_service"
)


def test_export_uploads_and_returns_path(mocker):
    fake = mocker.Mock()
    fake.bucket_pdfs = "tiledesk-ocr-pdfs"
    mocker.patch(MINIO_FACTORY, return_value=fake)

    path = _export_markdown_to_minio("doc1", "# Hello\n\n- a\n- b")

    assert path == "tiledesk-ocr-pdfs/doc1/doc1.md"
    _, kwargs = fake.upload_data.call_args
    assert kwargs["bucket_name"] == "tiledesk-ocr-pdfs"
    assert kwargs["object_name"] == "doc1/doc1.md"
    assert kwargs["data"] == "# Hello\n\n- a\n- b".encode("utf-8")
    assert kwargs["content_type"] == "text/markdown"


def test_export_returns_none_on_failure(mocker):
    mocker.patch(MINIO_FACTORY, side_effect=RuntimeError("minio down"))
    assert _export_markdown_to_minio("doc1", "x") is None
