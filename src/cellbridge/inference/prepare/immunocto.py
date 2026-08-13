"""Prepare Immunocto tar archives as patient-sharded LMDB data.

Example:
    python -m cellbridge.inference.prepare.immunocto \
        --metadata data/eval/immunocto/curated_ImmunoctoV1_refinement2.csv \
        --archives data/eval/immunocto \
        --output data/eval/immunocto/prepared
"""

import argparse
import shutil
import struct
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image

TEST_PATIENT = "CRC01"
# Exhaustive 4-of-39 selection minimizing mean squared relative error from the
# target aggregate class counts and total cell count (4/39 of the development set).
VALIDATION_PATIENTS = ("CRC18", "CRC30", "CRC32", "CRC37")
_BBOX = struct.Struct(">4I")


def _source_metadata(path):
    path = Path(path)
    table = pd.read_csv(
        path,
        usecols=["patient", "class", "cx", "cy", "mask"],
        dtype={"patient": str, "class": str, "mask": str},
    )
    required = ["patient", "class", "cx", "cy", "mask"]
    if table[required].isna().any().any():
        raise ValueError(f"{path} contains null source metadata")
    if table.duplicated(["patient", "mask"]).any():
        raise ValueError(f"{path} contains duplicate patient/mask pairs")
    invalid_names = [name for name in table["mask"] if PurePosixPath(name).name != name]
    if invalid_names:
        raise ValueError(
            f"{path} contains non-basename mask paths: {invalid_names[:5]}"
        )

    table.insert(0, "row_id", np.arange(len(table), dtype=np.int64))
    table = table.rename(columns={"patient": "sample_id", "class": "label"})
    table["cell_uid"] = table["sample_id"] + "/" + table["mask"]
    table["store_key"] = table["mask"]
    return table


def make_split(metadata):
    """Return the frozen 35/4/1 patient split, with CRC01 held out."""
    patients = sorted(metadata["sample_id"].astype(str).unique())
    required = {TEST_PATIENT, *VALIDATION_PATIENTS}
    missing = required - set(patients)
    if missing:
        raise ValueError(
            f"Immunocto is missing frozen split patients: {sorted(missing)}"
        )
    assignments = {
        patient: (
            "test"
            if patient == TEST_PATIENT
            else "val"
            if patient in VALIDATION_PATIENTS
            else "train"
        )
        for patient in patients
    }
    return pd.DataFrame(
        {
            "sample_name": patients,
            "train_test_val_split": [assignments[patient] for patient in patients],
        }
    )


def mask_bbox(png):
    """Return exclusive non-background bounds from one encoded mask."""
    mask = np.asarray(Image.open(BytesIO(png)))
    foreground = np.any(mask != 0, axis=-1) if mask.ndim == 3 else mask != 0
    y, x = np.nonzero(foreground)
    if not len(x):
        raise ValueError("cell mask is empty")
    return int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)


def _write_shard(archive, output, cells, *, map_size, batch_size=1_000):
    try:
        import lmdb
    except ImportError as error:
        raise ImportError("preparing Immunocto requires the lmdb package") from error

    patients = cells["sample_id"].astype(str).unique()
    if len(patients) != 1:
        raise ValueError("one Immunocto shard must contain exactly one patient")
    patient = patients[0]
    output.mkdir(parents=True)
    environment = lmdb.open(str(output), map_size=int(map_size), subdir=True)
    targets = set(cells["store_key"].astype(str))
    seen_images, seen_masks, bounds = set(), set(), {}
    transaction, pending = environment.begin(write=True), 0
    try:
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) < 3 or parts[-3] != patient or parts[-1] not in targets:
                    continue
                kind, name = parts[-2].casefold(), parts[-1]
                if kind not in {"he", "mask"}:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"could not read {member.name}")
                payload = extracted.read()
                key = name.encode()
                if kind == "he":
                    if name in seen_images or not transaction.put(
                        b"I:" + key, payload, overwrite=False
                    ):
                        raise ValueError(f"{archive}: duplicate H&E crop {name}")
                    seen_images.add(name)
                else:
                    bbox = mask_bbox(payload)
                    if name in seen_masks or not transaction.put(
                        b"B:" + key, _BBOX.pack(*bbox), overwrite=False
                    ):
                        raise ValueError(f"{archive}: duplicate mask {name}")
                    seen_masks.add(name)
                    bounds[name] = bbox
                pending += 1
                if pending >= batch_size:
                    transaction.commit()
                    transaction, pending = environment.begin(write=True), 0
        if pending:
            transaction.commit()
        else:
            transaction.abort()
        transaction = None
        missing_images = targets - seen_images
        missing_masks = targets - seen_masks
        if missing_images or missing_masks:
            raise ValueError(
                f"{archive}: missing required members: "
                f"HE={sorted(missing_images)[:5]}, masks={sorted(missing_masks)[:5]}"
            )
        environment.sync()
        return bounds
    except Exception:
        if transaction is not None:
            transaction.abort()
        raise
    finally:
        environment.close()


def prepare_dataset(
    metadata_path,
    archives_dir,
    output_dir,
    *,
    map_size=64 * 1024**3,
):
    """Build canonical metadata, an explicit split, and one LMDB per patient."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    archives_dir = Path(archives_dir)
    metadata = _source_metadata(metadata_path)
    split = make_split(metadata)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        shards = staging / "shards"
        shards.mkdir()
        all_bounds = np.empty((len(metadata), 4), dtype=np.uint32)
        for sample_id, cells in metadata.groupby("sample_id", sort=False):
            archive = archives_dir / f"{sample_id}.tar.gz"
            if not archive.exists():
                raise FileNotFoundError(f"missing Immunocto archive: {archive}")
            patient_bounds = _write_shard(
                archive,
                shards / f"{sample_id}.lmdb",
                cells,
                map_size=map_size,
            )
            all_bounds[cells.index] = [
                patient_bounds[str(key)] for key in cells["store_key"]
            ]

        metadata[["x0", "y0", "x1", "y1"]] = all_bounds
        columns = [
            "row_id",
            "sample_id",
            "cell_uid",
            "label",
            "store_key",
            "x0",
            "y0",
            "x1",
            "y1",
        ]
        metadata[columns].to_csv(staging / "metadata.csv", index=False)
        split.to_csv(staging / "split.csv", index=False)
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir / "metadata.csv", output_dir / "split.csv", output_dir / "shards"


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="curated Immunocto CSV")
    parser.add_argument("--archives", required=True, help="directory of CRC*.tar.gz")
    parser.add_argument("--output", required=True, help="new prepared output directory")
    parser.add_argument("--map-size-gb", type=_positive_int, default=64)
    args = parser.parse_args()
    metadata, split, shards = prepare_dataset(
        args.metadata,
        args.archives,
        args.output,
        map_size=args.map_size_gb * 1024**3,
    )
    print(f"Metadata: {metadata}")
    print(f"Split: {split}")
    print(f"Shards: {shards}")


if __name__ == "__main__":
    main()
