"""Download and normalize the corrected single-rater NuCLS dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image

HF_REPOSITORY = "minhanhto09/NuCLS_dataset"
HF_REVISION = "8d2694a5d2e4ca5864f9909e102847c75b9e8cc5"
LOADER_URL = f"https://huggingface.co/datasets/{HF_REPOSITORY}/raw/{HF_REVISION}/NuCLS_dataset.py"
ARCHIVE_URL = (
    "https://www.dropbox.com/scl/fi/zsm9l3bkwx808wfryv5zm/"
    "NuCLS_dataset.zip?rlkey=x3358slgrxt00zpn7zpkpjr2h&dl=1"
)
ARCHIVE_SIZE = 808_942_273
RAW_LABELS = frozenset(
    {
        "apoptotic_body",
        "ductal_epithelium",
        "eosinophil",
        "fibroblast",
        "lymphocyte",
        "macrophage",
        "mitotic_figure",
        "myoepithelium",
        "neutrophil",
        "plasma_cell",
        "tumor",
        "unlabeled",
        "vascular_endothelium",
    }
)
EXTRACT_DIRS = frozenset({"rgb", "csv", "train_test_splits"})


def sha256(path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archive(path, *, expected_size: int = ARCHIVE_SIZE) -> str:
    """Validate archive size and return its content digest."""
    path = Path(path)
    observed = path.stat().st_size
    if observed != expected_size:
        raise ValueError(f"{path}: expected {expected_size} bytes, found {observed}")
    return sha256(path)


def download_archive(
    destination,
    *,
    url: str = ARCHIVE_URL,
    expected_size: int = ARCHIVE_SIZE,
    opener=urllib.request.urlopen,
) -> str:
    """Download the external archive atomically, then validate it."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            with opener(url) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        digest = validate_archive(temporary, expected_size=expected_size)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return digest


def write_provenance(path, archive, digest: str | None = None) -> None:
    """Record the pinned loader and downloaded archive identity."""
    archive = Path(archive)
    record = {
        "archive_url": ARCHIVE_URL,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest or sha256(archive),
        "huggingface_repository": HF_REPOSITORY,
        "huggingface_revision": HF_REVISION,
        "loader_url": LOADER_URL,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _selected_member(name: str) -> tuple[str, tuple[str, ...]] | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    matches = [index for index, part in enumerate(path.parts) if part in EXTRACT_DIRS]
    if not matches:
        return None
    index = matches[0]
    tail = path.parts[index + 1 :]
    if not tail or any(part.startswith("._") for part in tail):
        return None
    return path.parts[index], tail


def extract_archive(archive, destination) -> None:
    """Safely extract only RGB, CSV, and official split files."""
    destination = Path(destination)
    with zipfile.ZipFile(archive) as source:
        selected = []
        targets = set()
        for member in source.infolist():
            resolved = _selected_member(member.filename)
            if resolved is None or member.is_dir():
                continue
            directory, tail = resolved
            target = destination / directory / Path(*tail)
            if target in targets:
                raise ValueError(f"duplicate archive target: {target}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(
                    f"archive member is a symbolic link: {member.filename}"
                )
            targets.add(target)
            selected.append((member, target))
        if not selected:
            raise ValueError("archive contains none of the required NuCLS files")
        existing = [target for _, target in selected if target.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite {existing[0]}")
        for member, target in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_, target.open("xb") as output:
                for block in iter(lambda: input_.read(1024 * 1024), b""):
                    output.write(block)


def _fold_slides(split_dir) -> dict[int, set[str]]:
    folds = {}
    for fold in range(1, 6):
        matches = sorted(Path(split_dir).glob(f"fold_{fold}_test.csv"))
        if len(matches) != 1:
            raise ValueError(f"expected one test split file for fold {fold}")
        table = pd.read_csv(matches[0], dtype=str)
        column = "slide_name" if "slide_name" in table else table.columns[1]
        slides = set(table[column].dropna().astype(str))
        if not slides or any(not slide.strip() for slide in slides):
            raise ValueError(f"{matches[0]} has invalid slide names")
        folds[fold] = slides
    all_slides = [slide for slides in folds.values() for slide in slides]
    if len(all_slides) != len(set(all_slides)):
        raise ValueError("a slide occurs in multiple official NuCLS test folds")
    return folds


def build_split(
    sample_ids,
    split_dir,
    *,
    test_fold: int = 1,
    val_fold: int = 2,
) -> pd.DataFrame:
    """Map FOVs to a hospital-disjoint train/validation/test split."""
    if test_fold == val_fold or {test_fold, val_fold} - set(range(1, 6)):
        raise ValueError("test_fold and val_fold must be distinct values from 1 to 5")
    folds = _fold_slides(split_dir)
    slide_fold = {slide: fold for fold, slides in folds.items() for slide in slides}
    rows = []
    for sample_id in map(str, sample_ids):
        matches = [slide for slide in slide_fold if slide in sample_id]
        if len(matches) != 1:
            raise ValueError(
                f"{sample_id}: expected exactly one official slide match, found {matches}"
            )
        slide = matches[0]
        fold = slide_fold[slide]
        split = "test" if fold == test_fold else "val" if fold == val_fold else "train"
        rows.append((sample_id, split, slide, fold))
    result = pd.DataFrame(
        rows,
        columns=[
            "sample_name",
            "train_test_val_split",
            "slide_id",
            "official_test_fold",
        ],
    )
    if result["sample_name"].duplicated().any():
        raise ValueError("duplicate NuCLS FOV identifiers")
    if result.groupby("slide_id")["train_test_val_split"].nunique().max() != 1:
        raise ValueError("NuCLS slide leakage across splits")
    return result


def _integer_bbox(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    columns = ["xmin", "ymin", "xmax", "ymax"]
    numeric = table[columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{path}: bounding boxes must contain finite integers")
    return numeric.astype(np.int64)


def _read_annotations(path: Path, sample_id: str, image_name: str) -> pd.DataFrame:
    table = pd.read_csv(path, keep_default_na=False)
    required = {
        "raw_classification",
        "type",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }
    if not required <= set(table):
        raise ValueError(f"{path}: missing columns {sorted(required - set(table))}")
    labels = table["raw_classification"].astype(str)
    if labels.str.strip().eq("").any():
        raise ValueError(f"{path}: blank raw_classification")
    unknown = set(labels) - RAW_LABELS
    if unknown:
        raise ValueError(f"{path}: unknown raw classifications {sorted(unknown)}")

    raw_bounds = _integer_bbox(table, path)
    with Image.open(path.parent.parent / "rgb" / image_name) as image:
        width, height = image.size
        image.verify()
    zero_width = raw_bounds["xmax"] == raw_bounds["xmin"]
    zero_height = raw_bounds["ymax"] == raw_bounds["ymin"]
    points = table["type"].astype(str).eq("point")
    bad = (
        (raw_bounds["xmin"] < 0)
        | (raw_bounds["ymin"] < 0)
        | (raw_bounds["xmin"] >= width)
        | (raw_bounds["ymin"] >= height)
        | (raw_bounds["xmax"] < raw_bounds["xmin"])
        | (raw_bounds["ymax"] < raw_bounds["ymin"])
        | ((zero_width | zero_height) & ~points)
        | (raw_bounds["xmax"] > width + 2)
        | (raw_bounds["ymax"] > height + 2)
    )
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} invalid bounding boxes")

    bounds = raw_bounds.copy()
    bounds.loc[zero_width, "xmax"] += 1
    bounds.loc[zero_height, "ymax"] += 1
    bounds["xmax"] = bounds["xmax"].clip(upper=width)
    bounds["ymax"] = bounds["ymax"].clip(upper=height)
    adjusted = bounds.ne(raw_bounds).any(axis=1)
    cell_id = np.arange(len(table), dtype=np.int64)
    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "image_name": image_name,
            "tile_id": "image",
            "cell_id": cell_id,
            "cell_uid": [f"{sample_id}/{value}" for value in cell_id],
            "label": labels,
            "annotation_type": table["type"].astype(str),
            "bbox_adjusted": adjusted,
            "x": ((raw_bounds["xmin"] + raw_bounds["xmax"]) / 2).clip(upper=width - 1),
            "y": ((raw_bounds["ymin"] + raw_bounds["ymax"]) / 2).clip(upper=height - 1),
            "x0": bounds["xmin"],
            "y0": bounds["ymin"],
            "x1": bounds["xmax"],
            "y1": bounds["ymax"],
        }
    )


def _validate_label_support(metadata: pd.DataFrame) -> None:
    labeled = metadata.loc[metadata["label"] != "unlabeled"]
    train = set(labeled.loc[labeled["split"] == "train", "label"])
    for split in ("val", "test"):
        missing = set(labeled.loc[labeled["split"] == split, "label"]) - train
        if missing:
            raise ValueError(
                f"raw classes in {split} are absent from train: {sorted(missing)}"
            )


def prepare_dataset(raw_dir, output_dir) -> tuple[Path, Path]:
    """Create canonical cell metadata and the explicit three-way split."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    rgb_dir, csv_dir = raw_dir / "rgb", raw_dir / "csv"
    images = {path.stem: path.name for path in rgb_dir.glob("*.png")}
    annotations = {path.stem: path for path in csv_dir.glob("*.csv")}
    if not images or set(images) != set(annotations):
        raise ValueError(
            "NuCLS RGB and annotation files do not have matching non-empty stems"
        )
    split = build_split(sorted(images), raw_dir / "train_test_splits")
    split_by_sample = split.set_index("sample_name")
    cells = []
    for sample_id in sorted(images):
        table = _read_annotations(annotations[sample_id], sample_id, images[sample_id])
        assignment = split_by_sample.loc[sample_id]
        table["slide_id"] = assignment["slide_id"]
        table["official_test_fold"] = assignment["official_test_fold"]
        table["split"] = assignment["train_test_val_split"]
        cells.append(table)
    metadata = pd.concat(cells, ignore_index=True)
    metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))
    _validate_label_support(metadata)

    split_path = output_dir / "splits" / "train_test_val_split.csv"
    metadata_path = output_dir / "nucls_info.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    split.to_csv(split_path, index=False)
    return metadata_path, split_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    root = args.output_dir
    archive = args.archive or root / "NuCLS_dataset.zip"
    digest = (
        validate_archive(archive) if archive.exists() else download_archive(archive)
    )
    write_provenance(root / "provenance.json", archive, digest)
    raw_dir = root / "raw"
    required = [raw_dir / directory for directory in EXTRACT_DIRS]
    if not all(path.is_dir() for path in required):
        extract_archive(archive, raw_dir)
    prepare_dataset(raw_dir, root)


if __name__ == "__main__":
    main()
