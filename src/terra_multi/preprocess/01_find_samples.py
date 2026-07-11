import argparse
from pathlib import Path

import pandas as pd
import yaml

INPUTS = {
    "transcripts": ("transcripts", "_transcripts.parquet"),
    "wsi": ("wsis", ".tif"),
    "xenium_cell_seg": ("xenium_seg", "_xenium_cell_seg.parquet"),
    "cellvit_seg": ("cellvit_seg", "_cellvit_seg.parquet"),
}


def discover_samples(cfg):
    """Return one manifest row per selected sample."""
    hest_root = Path(cfg["hest_root"])
    meta = pd.read_csv(hest_root / cfg["metadata_csv"], low_memory=False)

    technologies = cfg["technologies"]
    sel = meta[
        (meta["species"] == cfg["species"]) & (meta["st_technology"].isin(technologies))
    ]

    output_root = Path(cfg["output_root"])
    rows = []
    missing = []
    for _, r in sel.iterrows():
        sid = r["id"]
        row = {
            "id": sid,
            "st_technology": r["st_technology"],
            "pixel_size_um": r["pixel_size_um_estimated"],
            "output_dir": str(output_root / sid),
        }
        for col, (subdir, suffix) in INPUTS.items():
            path = hest_root / subdir / f"{sid}{suffix}"
            if not path.exists():
                missing.append(str(path))
            row[col] = str(path)
        rows.append(row)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} required input file(s) missing, e.g.:\n  "
            + "\n  ".join(missing[:10])
        )

    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Discover HEST-1k samples.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    manifest = discover_samples(cfg)

    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"Discovered {len(manifest)} samples -> {manifest_path}")
    print(manifest["st_technology"].value_counts().to_string())


if __name__ == "__main__":
    main()
