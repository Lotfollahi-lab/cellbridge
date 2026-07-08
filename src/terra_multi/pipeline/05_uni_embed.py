"""Create UNI2-h image embeddings aligned to TERRA rows."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

DST_PIXEL_SIZE = 0.5
TARGET_PATCH_SIZE = 224
ENCODER_NAME = "uni_v2"
BATCH_SIZE = 128


def src_patch_size(px_um):
    """Level-0 crop side, in pixels."""
    return round(TARGET_PATCH_SIZE * DST_PIXEL_SIZE / px_um)


def crop_patch(wsi, cx, cy, patch_size_src):
    """Return a padded 224x224 RGB patch."""
    H, W = wsi.shape[:2]
    half = patch_size_src // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + patch_size_src, y0 + patch_size_src
    vx0, vy0, vx1, vy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    canvas = np.full((patch_size_src, patch_size_src, 3), 255, dtype=np.uint8)
    if vx1 > vx0 and vy1 > vy0:
        canvas[vy0 - y0 : vy1 - y0, vx0 - x0 : vx1 - x0] = wsi[vy0:vy1, vx0:vx1, :3]
    img = Image.fromarray(canvas).resize(
        (TARGET_PATCH_SIZE, TARGET_PATCH_SIZE), Image.BILINEAR
    )
    return np.asarray(img)


def open_wsi(path, tifffile, zarr):
    try:
        return zarr.open(tifffile.imread(path, aszarr=True, level=0), mode="r")
    except Exception as exc:
        print(f"level=0 TIFF open failed for {path}: {exc}; trying key=0")
        return zarr.open(tifffile.imread(path, aszarr=True, key=0), mode="r")


def main():
    parser = argparse.ArgumentParser(description="UNI2-h embed one sample's cells.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    manifest = pd.read_csv(Path(cfg["output_root"]) / "manifest.csv").set_index("id")
    row = manifest.loc[args.sample_id]
    out_dir = Path(row["output_dir"])

    terra = np.load(out_dir / f"{args.sample_id}_terra.npz", allow_pickle=True)
    cell_ids = terra["cell_ids"]
    centers = np.stack([terra["he_x"], terra["he_y"]], axis=1).astype(float)

    patch_size_src = src_patch_size(float(row["pixel_size_um"]))

    import tifffile
    import torch
    import zarr
    from trident.patch_encoder_models import encoder_factory

    wsi = open_wsi(row["wsi"], tifffile, zarr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder_factory(ENCODER_NAME).to(device).eval()
    eval_transform = encoder.eval_transforms
    precision = encoder.precision

    embs = []
    with torch.inference_mode():
        for i in range(0, len(centers), args.batch_size):
            batch = [
                eval_transform(Image.fromarray(crop_patch(wsi, cx, cy, patch_size_src)))
                for cx, cy in centers[i : i + args.batch_size]
            ]
            x = torch.stack(batch).to(device)
            with torch.autocast(
                device_type="cuda", dtype=precision, enabled=(device == "cuda")
            ):
                out = encoder(x)
            embs.append(out.float().cpu().numpy())

    uni_emb = np.concatenate(embs, axis=0)
    assert uni_emb.shape[0] == len(cell_ids)

    out_path = out_dir / f"{args.sample_id}_uni2.npz"
    np.savez(out_path, cell_ids=cell_ids, uni_emb=uni_emb)
    print(
        f"{args.sample_id}: embedded {len(cell_ids):,} patches "
        f"(crop {patch_size_src}px @ {float(row['pixel_size_um']):.4f} um/px) -> {out_path}"
    )


if __name__ == "__main__":
    main()
