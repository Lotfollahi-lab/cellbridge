# NOVAE teacher comparison

NOVAE uses the same 4L/all-cells student trainer as the TERRA model. `configs/train/lora/novae.yaml` changes only the teacher input, target dimension (64), run name and destinations. Teacher artifacts are `${CELLBRIDGE_DATA_ROOT}/train/hest/<sample>/<sample>_novae.npz`.

Existing cached artifacts can be used directly. To regenerate them, use an environment with `novae==1.1.1`, AnnData, Scanpy, PyTorch and Hugging Face Hub, with the pinned `prism-oncology/novae-human-0` revision from `configs/preprocess/novae.yaml` already cached. The loader intentionally uses local files only. It reads reconstructed counts and the TERRA reference cell IDs/coordinates, not image embeddings, and refuses to overwrite existing teacher artifacts.

```bash
python -m cellbridge.preprocess.novae_embed \
  --config configs/preprocess/novae.yaml --sample-id <SAMPLE_ID> --device cuda
```

Run this for the `distill_train` and `distill_val` samples in the split CSV. To regenerate alongside existing artifacts, override the preprocessing `output_dir` and training `teacher_embeddings_root` to the same fresh directory. Reference IDs preserve the original validation-cell selection; cells without valid NOVAE vectors are omitted rather than replaced. Consequently, all eligible NOVAE training cells need not equal the TERRA cell count.

```bash
python -m cellbridge.train.lora.train --config configs/train/lora/novae.yaml
```

The canonical model name is `uni_novae_four_layers_adapted_all_cells`; both training outputs and checkpoint paths follow the README layout. For a fresh rerun, supply new `--output-dir` and `--checkpoint-dir`, as for TERRA. The generic LSF training launcher accepts this config too.

Evaluate with the same dataset configs and adapted representation, explicitly overriding the checkpoint:

```bash
python -m cellbridge.inference.run run --config configs/inference/nucls.yaml \
  --method cellbridge --device cuda \
  --checkpoint "$CELLBRIDGE_CHECKPOINTS_ROOT/uni_novae_four_layers_adapted_all_cells/best.pt" \
  --output-dir "$CELLBRIDGE_OUTPUTS_ROOT/reproducibility/inference/cellbridge/uni_novae_four_layers_adapted_all_cells/nucls"
```

Use the corresponding dataset config for the other benchmarks, and a per-section subdirectory for HEST. Notebook 10 reads these NOVAE results alongside the main TERRA model.
