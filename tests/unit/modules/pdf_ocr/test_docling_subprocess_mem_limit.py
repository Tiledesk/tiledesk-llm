"""
PDF_OCR_CHILD_MEM_LIMIT_MB (docs/MIGLIORIE_DA_FARE.md P1#10): RLIMIT_AS caps
virtual address space, not real RSS — on a GPU deployment a CUDA context alone
reserves tens of GB of VA, so applying this cap killed the child the instant
it initialized CUDA, indistinguishable from a real OOM. _child_init() must
now skip the cap (with a warning) whenever CUDA is available, and only apply
it on CPU-only deployments.
"""
from unittest.mock import MagicMock, patch

from tilellm.modules.pdf_ocr.services import docling_subprocess as mod


def test_disabled_by_default_never_calls_setrlimit(monkeypatch):
    monkeypatch.setattr(mod, "CHILD_MEM_LIMIT_MB", 0)
    with patch("resource.setrlimit") as mock_setrlimit:
        mod._child_init()
    mock_setrlimit.assert_not_called()


def test_cuda_available_skips_cap_and_warns(monkeypatch):
    monkeypatch.setattr(mod, "CHILD_MEM_LIMIT_MB", 4096)
    with patch.object(mod, "_cuda_available", return_value=True), \
         patch("resource.setrlimit") as mock_setrlimit, \
         patch.object(mod.logger, "warning") as mock_warning:
        mod._child_init()

    mock_setrlimit.assert_not_called()
    mock_warning.assert_called_once()
    message = mock_warning.call_args[0][0]
    assert "PDF_OCR_CHILD_MEM_LIMIT_MB" in message
    assert "CUDA" in message


def test_cpu_only_applies_the_cap(monkeypatch):
    monkeypatch.setattr(mod, "CHILD_MEM_LIMIT_MB", 4096)
    with patch.object(mod, "_cuda_available", return_value=False), \
         patch("resource.setrlimit") as mock_setrlimit:
        mod._child_init()

    mock_setrlimit.assert_called_once()
    args, _ = mock_setrlimit.call_args
    limit_bytes = 4096 * 1024 * 1024
    assert args[1] == (limit_bytes, limit_bytes)


def test_cuda_check_failure_defaults_to_not_available():
    """torch not importable (e.g. lite image) -> treated as CPU-only, not as
    'unknown -> skip the cap'. Matches the pre-existing behavior for the CPU
    case (best-effort cap), just no longer crashes when torch is absent."""
    with patch("builtins.__import__", side_effect=ImportError("no torch")):
        assert mod._cuda_available() is False
