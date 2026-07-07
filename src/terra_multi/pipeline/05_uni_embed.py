"""UNI2-h image embeddings for one sample's cells.

    PYTHONPATH=src python -m terra_multi.pipeline.uni_embed \\
        --config configs/pipeline/pipeline.yaml --sample-id TENX133

Takes the cell set + H&E-pixel centroids from the sample's ``{id}_terra.npz`` (so
the UNI rows line up 1:1 with the TERRA rows), cuts a 20x field of view around
each cell from the WSI, and embeds it with UNI2-h. UNI2-h expects 20x
(~0.5 um/px), so the level-0 crop side scales with the slide's own pixel size --
a wider window on finer slides -- and is resized to 224 px. Patches are cropped
and embedded in batches (never all held in memory). Writes
``{output_dir}/{id}_uni2.npz``. Needs a GPU.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

DST_PIXEL_SIZE = 0.5  # UNI2-h target: 20x magnification (~0.5 um/px)
TARGET_PATCH_SIZE = 224  # px fed to UNI2-h
ENCODER_NAME = "uni_v2"  # UNI2-h in the HEST/TRIDENT registry
BATCH_SIZE = 128


def src_patch_size(px_um):
    """Level-0 window side (px) spanning the 20x FOV for a slide at ``px_um``.

    ``TARGET_PATCH_SIZE`` px at ``DST_PIXEL_SIZE`` um/px is a fixed physical FOV;
    at the slide's native ``px_um`` that FOV is this many level-0 pixels, which we
    then resize down to ``TARGET_PATCH_SIZE`` (finer slide -> wider crop).
    """
    return round(TARGET_PATCH_SIZE * DST_PIXEL_SIZE / px_um)


def crop_patch(wsi, cx, cy, patch_size_src):
    """A 224x224x3 uint8 patch centered on level-0 pixel ``(cx, cy)``.

    Cuts a ``patch_size_src`` window, white-pads any part off the slide edge, and
    resizes to ``TARGET_PATCH_SIZE`` (bilinear) -- i.e. resamples to ~0.5 um/px.
    """
    H, W = wsi.shape[:2]
    half = patch_size_src // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + patch_size_src, y0 + patch_size_src
    vx0, vy0, vx1, vy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    canvas = np.full((patch_size_src, patch_size_src, 3), 255, dtype=np.uint8)
    if vx1 > vx0 and vy1 > vy0:  # some of the window overlaps the slide
        canvas[vy0 - y0 : vy1 - y0, vx0 - x0 : vx1 - x0] = wsi[vy0:vy1, vx0:vx1, :3]
    img = Image.fromarray(canvas).resize(
        (TARGET_PATCH_SIZE, TARGET_PATCH_SIZE), Image.BILINEAR
    )
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser(description="UNI2-h embed one sample's cells.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    manifest = pd.read_csv(Path(cfg["output_root"]) / "manifest.csv").set_index("id")
    row = manifest.loc[args.sample_id]
    out_dir = Path(row["output_dir"])

    # Cell set + H&E centroids come from the TERRA npz, so UNI rows stay 1:1 with
    # the TERRA rows (the fusion concatenates the two embedding matrices directly).
    terra = np.load(out_dir / f"{args.sample_id}_terra.npz", allow_pickle=True)
    cell_ids = terra["cell_ids"]
    centers = np.stack([terra["he_x"], terra["he_y"]], axis=1).astype(float)

    patch_size_src = src_patch_size(float(row["pixel_size_um"]))

    # Heavy, GPU-touching deps: import here so the module stays importable
    # (crop_patch/src_patch_size are unit-testable) without torch/trident.
    import tifffile
    import torch
    import zarr
    from trident.patch_encoder_models import encoder_factory

    wsi = zarr.open(tifffile.imread(row["wsi"], aszarr=True, level=0), mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder_factory(ENCODER_NAME).to(device).eval()
    eval_transform = encoder.eval_transforms
    precision = encoder.precision

    # Crop + embed one batch of cells at a time so patches never all sit in RAM.
    embs = []
    with torch.inference_mode():
        for i in range(0, len(centers), BATCH_SIZE):
            batch = [
                eval_transform(Image.fromarray(crop_patch(wsi, cx, cy, patch_size_src)))
                for cx, cy in centers[i : i + BATCH_SIZE]
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
