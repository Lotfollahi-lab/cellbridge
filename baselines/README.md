# External baselines

This directory documents external models evaluated by `terra-multi`. The model
repositories themselves live in the workspace-level `repos/` directory; they
are not copied or added as Git submodules here.

For each baseline, `terra-multi` owns only the integration layer:

- a checked-in YAML configuration under `configs/baselines/`;
- extraction and evaluation code under `src/terra_multi/baselines/`;
- local and LSF launchers under `scripts/baselines/`;
- a short, reproducible runbook in this directory.

Keep upstream model code in its own fork when a small compatibility change is
needed. Record the upstream commit, make baseline-specific changes on a named
branch, and keep an `upstream` remote so changes can be rebased deliberately.
Generated embeddings, checkpoints, caches, and metrics belong under
`outputs/baselines/` and remain untracked.

Available integrations:

- [SpatialWhisperer](spatialwhisperer/README.md)
