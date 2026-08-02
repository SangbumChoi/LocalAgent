"""Unit checks for the cross-surface public continuation runner."""

from pathlib import Path
import json

import pytest

from scripts.train_cross_surface_continuation import (
    _parse_labeled_path,
    _parse_source_reference,
    _source_profile,
)
from localagent.data.schema import Conversation


def test_parse_labeled_path_preserves_equals_in_path() -> None:
    label, path = _parse_labeled_path("mobile=/tmp/fixture=with-equals.jsonl")
    assert label == "mobile"
    assert path == Path("/tmp/fixture=with-equals.jsonl")


def test_parse_labeled_path_rejects_missing_label_or_path() -> None:
    with pytest.raises(Exception, match="expected LABEL=PATH"):
        _parse_labeled_path("=/tmp/rows.jsonl")
    with pytest.raises(Exception, match="expected LABEL=PATH"):
        _parse_labeled_path("mobile=")


def test_parse_source_reference_requires_dataset_and_original_url() -> None:
    assert _parse_source_reference("mobile=dataset/name|https://example.test/data") == (
        "mobile",
        {"dataset": "dataset/name", "url": "https://example.test/data"},
    )
    with pytest.raises(Exception, match=r"expected LABEL=DATASET\|URL"):
        _parse_source_reference("mobile=dataset/name")


def test_source_profile_binds_public_metadata_and_visual_omission(tmp_path: Path) -> None:
    row = Conversation.from_json(
        json.dumps(
            {
            "messages": [{"role": "user", "content": "open Gmail"}],
            "tools": [],
            "meta": {
                "source_dataset": "google/androidcontrol",
                "source_revision": "generation=1",
                "source_split": "train",
                "parent_record_id": "episode-1",
                "visual_input_omitted": True,
            },
            }
        )
    )
    source_path = tmp_path / "rows.jsonl"
    source_path.write_text(row.to_json() + "\n", encoding="utf-8")
    profile = _source_profile(
        "mobile",
        source_path,
        [row],
        {"dataset": "google/androidcontrol", "url": "https://example.test/androidcontrol"},
    )
    assert profile["label"] == "mobile"
    assert profile["datasets"] == ["google/androidcontrol"]
    assert profile["revisions"] == ["generation=1"]
    assert profile["splits"] == ["train"]
    assert profile["unique_parent_records"] == 1
    assert profile["visual_input_omitted_rows"] == 1
    assert profile["public_reference"]["dataset"] == "google/androidcontrol"
    assert profile["public_reference"]["url"] == "https://example.test/androidcontrol"
