from pathlib import Path

import pytest

from localagent.data.realistic_catalog import load_catalog, train_entries, validate_catalog


CATALOG = Path(__file__).parents[1] / "configs/data/realistic-agent-eval.catalog.yaml"


def test_realistic_catalog_is_pinned_and_split_safe() -> None:
    catalog, fingerprint = load_catalog(CATALOG)
    assert len(fingerprint) == 64
    assert len(catalog["entries"]) >= 10
    train = train_entries(catalog)
    assert {row["id"] for row in train} == {
        "androidcontrol",
        "android_in_the_wild",
        "xlam_function_calling",
        "mind2web_train",
    }
    assert all(row["train_policy"] == "train" for row in train)
    assert all(row["train_policy"] != "train" for row in catalog["entries"] if row not in train)


def test_train_rows_require_public_download_and_verified_license() -> None:
    catalog, _ = load_catalog(CATALOG)
    broken = {**catalog, "entries": [dict(catalog["entries"][0])]}
    broken["entries"][0]["access_status"] = "public_runtime"
    with pytest.raises(ValueError, match="public_download"):
        validate_catalog(broken)

    broken = {**catalog, "entries": [dict(catalog["entries"][0])]}
    broken["entries"][0]["license"] = {
        "name": "unknown",
        "evidence": "https://example.com/license",
    }
    with pytest.raises(ValueError, match="verified license"):
        validate_catalog(broken)


def test_catalog_rejects_duplicate_ids() -> None:
    catalog, _ = load_catalog(CATALOG)
    duplicate = {**catalog, "entries": list(catalog["entries"][:2])}
    duplicate["entries"][1] = dict(duplicate["entries"][0])
    with pytest.raises(ValueError, match="duplicate catalog id"):
        validate_catalog(duplicate)
