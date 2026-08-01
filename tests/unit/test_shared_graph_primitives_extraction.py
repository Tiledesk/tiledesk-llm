#!/usr/bin/env python3
"""
Tier-1 extraction out of the deprecated Neo4j package
(tilellm.modules.knowledge_graph, see its __init__ docstring).

These four pieces contain no Neo4j code at all — they ended up inside that
package by historical accident and are imported by ACTIVE modules
(store.graph, pdf_ocr, lgraph, knowledge_graph_tinkerpop, __main__), which is
what kept the deprecated folder undeletable. Moving them to their natural
homes makes the folder droppable: what remains there is genuinely Neo4j-bound
and already guarded by try/except at every call site.

    models.models        (Node/Relationship/+Update) → tilellm.models.graph
    models.schemas       (TaskPollResponse)          → tilellm.models.schemas.general_schemas
    utils.rrf            (reciprocal_rank_fusion)    → tilellm.shared.rrf
    services.minio_storage                           → tilellm.shared.minio_storage
"""
import ast
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


class TestNewCanonicalLocations:
    def test_graph_models_importable_from_tilellm_models(self):
        from tilellm.models.graph import Node, NodeUpdate, Relationship, RelationshipUpdate

        node = Node(label="ORGANIZATION", properties={"name": "Banca ABC"})
        assert node.label == "ORGANIZATION"
        rel = Relationship(source_id="1", target_id="2", type="HAS_LOAN", properties={})
        assert rel.type == "HAS_LOAN"
        assert set(NodeUpdate.model_fields) and set(RelationshipUpdate.model_fields)

    def test_task_poll_response_importable_from_shared_schemas(self):
        from tilellm.models.schemas.general_schemas import TaskPollResponse

        r = TaskPollResponse(task_id="abc", status="in_progress")
        assert r.result is None and r.error is None

    def test_rrf_importable_from_shared(self):
        from tilellm.shared.rrf import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
        assert [doc_id for doc_id, _ in fused][:2] in (["a", "b"], ["b", "a"])
        assert len(fused) == 2

    def test_minio_storage_importable_from_shared(self):
        from tilellm.shared.minio_storage import MinIOStorageService, get_minio_storage_service

        assert callable(get_minio_storage_service)
        assert hasattr(MinIOStorageService, "upload_parquet_file")


class TestNoActiveModuleDependsOnDeprecatedPackage:
    """The whole point: active code must not reach into the deprecated folder
    for these primitives, otherwise deleting it breaks the app."""

    ACTIVE_FILES = [
        "tilellm/__main__.py",
        "tilellm/store/graph/base_graph_repository.py",
        "tilellm/modules/lgraph/logic.py",
        "tilellm/modules/ingestion/docx_processor.py",
        "tilellm/modules/knowledge_graph_tinkerpop/models/__init__.py",
        "tilellm/modules/knowledge_graph_tinkerpop/repository/tinkerpop_repository.py",
        "tilellm/modules/pdf_ocr/logic.py",
        "tilellm/modules/pdf_ocr/services/docling_processor.py",
        "tilellm/modules/pdf_ocr/services/document_structure_extractor.py",
        "tilellm/modules/pdf_ocr/services/pdf_entity_extractor.py",
        "tilellm/modules/pdf_ocr/services/image_semantic_linker.py",
        "tilellm/modules/pdf_ocr/services/table_semantic_linker.py",
    ]
    TIER1_SUFFIXES = ("models", "utils.rrf", "services.minio_storage", "models.schemas")

    @pytest.mark.parametrize("rel_path", ACTIVE_FILES)
    def test_no_tier1_import_from_deprecated_package(self, rel_path):
        path = REPO / rel_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                m = node.module
                if m.startswith("tilellm.modules.knowledge_graph.") and not m.startswith(
                    "tilellm.modules.knowledge_graph_"
                ):
                    tail = m[len("tilellm.modules.knowledge_graph.") :]
                    if any(tail.startswith(s) for s in self.TIER1_SUFFIXES):
                        offenders.append(f"line {node.lineno}: {m}")
        assert not offenders, f"{rel_path} still imports Tier-1 from the deprecated package: {offenders}"
