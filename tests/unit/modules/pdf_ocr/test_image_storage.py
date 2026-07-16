"""
Tests for image-to-object-storage helpers added to the md_simple path:

1. _upload_images_to_object_storage
   - uploads PNG bytes, sets img["path"]
   - skips images with no image_data
   - on service init failure: path stays '', no raise
   - on single upload failure: path stays '', no raise, other images continue
   - clears image_data after upload (memory hygiene)

2. _merge_image_captions
   - merges image_descriptions dict into img["caption"]
   - leaves images without a matching description unchanged
"""

from unittest.mock import MagicMock

import pytest

MINIO_FACTORY = (
    "tilellm.modules.knowledge_graph.services.minio_storage.get_minio_storage_service"
)


def _fake_pil(content: bytes = b"FAKEPNG") -> MagicMock:
    """Mock PIL Image whose .save(buf, format=...) writes fixed bytes."""
    img = MagicMock()

    def _save(buf, format="PNG"):
        buf.write(content)

    img.save.side_effect = _save
    return img


# ---------------------------------------------------------------------------
# _upload_images_to_object_storage
# ---------------------------------------------------------------------------

def test_uploads_png_and_sets_path(mocker):
    from tilellm.modules.pdf_ocr.logic import _upload_images_to_object_storage

    fake_svc = mocker.Mock()
    fake_svc.bucket_images = "document-images"
    mocker.patch(MINIO_FACTORY, return_value=fake_svc)

    img = {"id": "doc1_img_0", "page": 1, "image_data": _fake_pil(b"PNGBYTES")}
    _upload_images_to_object_storage("doc1", [img])

    assert img["path"] == "document-images/doc1/images/doc1_img_0.png"
    fake_svc.upload_data.assert_called_once()
    _, kwargs = fake_svc.upload_data.call_args
    assert kwargs["bucket_name"] == "document-images"
    assert kwargs["object_name"] == "doc1/images/doc1_img_0.png"
    assert kwargs["data"] == b"PNGBYTES"
    assert kwargs["content_type"] == "image/png"


def test_clears_image_data_after_upload(mocker):
    from tilellm.modules.pdf_ocr.logic import _upload_images_to_object_storage

    fake_svc = mocker.Mock()
    fake_svc.bucket_images = "document-images"
    mocker.patch(MINIO_FACTORY, return_value=fake_svc)

    img = {"id": "doc1_img_0", "page": 1, "image_data": _fake_pil()}
    _upload_images_to_object_storage("doc1", [img])

    assert img["image_data"] is None


def test_skips_image_without_image_data(mocker):
    from tilellm.modules.pdf_ocr.logic import _upload_images_to_object_storage

    fake_svc = mocker.Mock()
    fake_svc.bucket_images = "document-images"
    mocker.patch(MINIO_FACTORY, return_value=fake_svc)

    img = {"id": "doc1_img_0", "page": 1, "image_data": None}
    _upload_images_to_object_storage("doc1", [img])

    fake_svc.upload_data.assert_not_called()
    assert img.get("path", "") == ""


def test_service_failure_does_not_raise(mocker):
    from tilellm.modules.pdf_ocr.logic import _upload_images_to_object_storage

    mocker.patch(MINIO_FACTORY, side_effect=RuntimeError("s3 down"))

    img = {"id": "doc1_img_0", "page": 1, "image_data": _fake_pil()}
    _upload_images_to_object_storage("doc1", [img])  # must not raise

    assert img.get("path", "") == ""


def test_single_upload_failure_leaves_path_empty_and_continues(mocker):
    from tilellm.modules.pdf_ocr.logic import _upload_images_to_object_storage

    fake_svc = mocker.Mock()
    fake_svc.bucket_images = "document-images"
    # first upload raises, second succeeds
    fake_svc.upload_data.side_effect = [RuntimeError("upload failed"), None]
    mocker.patch(MINIO_FACTORY, return_value=fake_svc)

    img0 = {"id": "doc1_img_0", "page": 1, "image_data": _fake_pil(b"A")}
    img1 = {"id": "doc1_img_1", "page": 2, "image_data": _fake_pil(b"B")}
    _upload_images_to_object_storage("doc1", [img0, img1])

    assert img0.get("path", "") == ""
    assert img1["path"] == "document-images/doc1/images/doc1_img_1.png"


# ---------------------------------------------------------------------------
# _merge_image_captions
# ---------------------------------------------------------------------------

def test_merge_sets_caption_on_matching_images():
    from tilellm.modules.pdf_ocr.logic import _merge_image_captions

    images = [
        {"id": "doc1_img_0", "page": 1},
        {"id": "doc1_img_1", "page": 2},
    ]
    descs = {"doc1_img_0": "A revenue chart", "doc1_img_2": "Orphan desc"}
    _merge_image_captions(images, descs)

    assert images[0]["caption"] == "A revenue chart"
    assert "caption" not in images[1]


def test_merge_does_not_overwrite_existing_caption():
    from tilellm.modules.pdf_ocr.logic import _merge_image_captions

    images = [{"id": "doc1_img_0", "page": 1, "caption": "original"}]
    descs = {"doc1_img_0": "new desc"}
    _merge_image_captions(images, descs)

    assert images[0]["caption"] == "new desc"


def test_merge_empty_descriptions_is_noop():
    from tilellm.modules.pdf_ocr.logic import _merge_image_captions

    images = [{"id": "doc1_img_0", "page": 1}]
    _merge_image_captions(images, {})

    assert "caption" not in images[0]
