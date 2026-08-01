#!/usr/bin/env python3
"""
_reimport's list coercion must survive everything parquet hands back.

A missing cell comes out of pandas as NaN — a float, and `NaN is not None` is
True, so `list(value)` blew up on the real 256237 snapshot:
    TypeError: 'float' object is not iterable
Parquet list columns also come back as numpy arrays, not Python lists.
"""
import io

import numpy as np
import pandas as pd
import pytest


def _as_list():
    from tilellm.modules.knowledge_graph_falkor.services.graph_optimizer import _as_list
    return _as_list


class TestAsListCoercion:
    def test_nan_becomes_empty_list(self):
        assert _as_list()(float("nan")) == []

    def test_none_becomes_empty_list(self):
        assert _as_list()(None) == []

    def test_json_string_is_parsed(self):
        assert _as_list()('["a", "b"]') == ["a", "b"]

    def test_broken_json_string_becomes_empty_list(self):
        assert _as_list()("not json") == []

    def test_real_list_passes_through(self):
        assert _as_list()(["a", "b"]) == ["a", "b"]

    def test_numpy_array_from_parquet_becomes_list(self):
        assert _as_list()(np.array(["a", "b"], dtype=object)) == ["a", "b"]

    def test_stray_scalar_becomes_empty_list(self):
        assert _as_list()(3.14) == []

    def test_nan_read_back_from_a_real_parquet_roundtrip(self):
        buf = io.BytesIO()
        pd.DataFrame([{"source_ids": None}]).to_parquet(buf, index=False)
        df = pd.read_parquet(io.BytesIO(buf.getvalue()))
        assert _as_list()(df.iloc[0]["source_ids"]) == []
