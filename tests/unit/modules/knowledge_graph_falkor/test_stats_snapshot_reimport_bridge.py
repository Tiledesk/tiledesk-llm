#!/usr/bin/env python3
"""
POST /api/kg-falkor/reimport (GraphReimportRequest) already exists but only
reads GraphOptimizer's own snapshot format (graph_snapshots/{graph_name}/...,
columns id/label/properties). Every /create run *also* auto-saves a
different-shaped snapshot (community_graph_service._save_stats, columns
node_id/labels/entity_type/... under {index_name}/{index_type}/{namespace}/
{timestamp}/) — the one that actually exists for namespace 256237 after the
FalkorDB wipe (2026-07-29), since /optimize/graph_snapshots was never run.

convert_stats_snapshot_to_optimizer_format bridges the two: reads the
_save_stats-schema parquet bytes and re-serialises them into the
id/label/properties (nodes) and id/type/source_id/target_id/properties
(relationships) schema that GraphOptimizer._reimport expects — so the
existing /reimport endpoint can rebuild a graph from either snapshot kind.
"""
import io

import pandas as pd
import pytest


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


class TestConvertStatsSnapshotToOptimizerFormat:
    def test_converts_entities_columns(self):
        from tilellm.modules.knowledge_graph_falkor.services.graph_optimizer import (
            convert_stats_snapshot_to_optimizer_format,
        )

        entities_df = pd.DataFrame([
            {"name": "Banca ABC", "description": "Ente creditore", "entity_type": "ORGANIZATION",
             "labels": ["ORGANIZATION"], "node_id": "17", "namespace": "256237", "source_ids": "[]"},
            {"name": "Mario Rossi", "description": "Debitore", "entity_type": "PERSON",
             "labels": ["PERSON"], "node_id": "42", "namespace": "256237", "source_ids": "[]"},
        ])
        entities_bytes = _to_parquet_bytes(entities_df)
        relationships_df = pd.DataFrame([
            {"relationship_id": "5", "source_id": "17", "target_id": "42",
             "relationship_type": "HAS_LOAN", "amount": 1000},
        ])
        rels_bytes = _to_parquet_bytes(relationships_df)

        nodes_bytes, rels_out_bytes = convert_stats_snapshot_to_optimizer_format(entities_bytes, rels_bytes)

        nodes_df = pd.read_parquet(io.BytesIO(nodes_bytes))
        assert set(nodes_df["id"]) == {"17", "42"}
        assert set(nodes_df["label"]) == {"ORGANIZATION", "PERSON"}
        assert "node_id" not in nodes_df.columns
        assert "labels" not in nodes_df.columns
        assert set(nodes_df["name"]) == {"Banca ABC", "Mario Rossi"}

        rels_df = pd.read_parquet(io.BytesIO(rels_out_bytes))
        assert rels_df.iloc[0]["id"] == "5"
        assert rels_df.iloc[0]["source_id"] == "17"
        assert rels_df.iloc[0]["target_id"] == "42"
        assert rels_df.iloc[0]["type"] == "HAS_LOAN"
        assert rels_df.iloc[0]["amount"] == 1000
        assert "relationship_id" not in rels_df.columns
        assert "relationship_type" not in rels_df.columns

    def test_falls_back_to_entity_type_when_labels_missing(self):
        from tilellm.modules.knowledge_graph_falkor.services.graph_optimizer import (
            convert_stats_snapshot_to_optimizer_format,
        )

        entities_df = pd.DataFrame([
            {"name": "Ditta Rossi SRL", "entity_type": "ORGANIZATION", "node_id": "9", "source_ids": "[]"},
        ])
        entities_bytes = _to_parquet_bytes(entities_df)
        rels_bytes = _to_parquet_bytes(pd.DataFrame(columns=["relationship_id", "source_id", "target_id", "relationship_type"]))

        nodes_bytes, _ = convert_stats_snapshot_to_optimizer_format(entities_bytes, rels_bytes)

        nodes_df = pd.read_parquet(io.BytesIO(nodes_bytes))
        assert nodes_df.iloc[0]["label"] == "ORGANIZATION"
