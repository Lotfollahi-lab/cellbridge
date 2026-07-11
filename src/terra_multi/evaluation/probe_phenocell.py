"""Compare frozen raw UNI and distilled features on PhenoCell labels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMMUNE_CLASSES = (
    "B cells",
    "Macrophages/Monocytes",
    "Dendritic cells",
    "T cells",
    "Granulocytes",
    "NK cells",
    "Plasma cells",
)
ARMS = ("uni", "distilled")


def load_uni(manifest):
    features, records = [], []
    for path in manifest["path"]:
        with np.load(path, allow_pickle=False) as data:
            features.append(data["uni_emb"].astype(np.float32))
            records.append(
                pd.DataFrame(
                    {
                        "row_id": data["row_id"],
                        "sample_id": data["sample_id"].astype(str),
                        "tile_id": data["tile_id"].astype(str),
                        "cell_id": data["cell_id"],
                        "label": data["label"].astype(str),
                        "class_id": data["class_id"],
                        "split": data["split"].astype(str),
                    }
                )
            )
    table = pd.concat(records, ignore_index=True)
    split = table["split"].replace({"valid": "val"}).to_numpy()
    return np.concatenate(features), table, split


def load_distilled(manifest, expected_row_ids):
    features, row_ids = [], []
    for path in manifest["path"]:
        with np.load(path, allow_pickle=False) as data:
            features.append(data["distilled_emb"].astype(np.float32))
            row_ids.append(data["row_id"])
    row_ids = np.concatenate(row_ids)
    if not np.array_equal(row_ids, expected_row_ids):
        raise ValueError("distilled embeddings are not aligned with cached UNI rows")
    return np.concatenate(features)


def validate_official_split(table, split_path):
    official = pd.read_csv(split_path).rename(
        columns={"sample_name": "sample_id", "train_test_val_split": "official_split"}
    )
    official["official_split"] = official["official_split"].replace({"valid": "val"})
    observed = table[["sample_id", "split"]].drop_duplicates().copy()
    observed["split"] = observed["split"].replace({"valid": "val"})
    merged = observed.merge(official, on="sample_id", how="left", validate="one_to_one")
    bad = merged["official_split"].isna() | (merged["split"] != merged["official_split"])
    if bad.any():
        raise ValueError(f"metadata disagrees with official split for {bad.sum()} samples")


def reduce_uni(features, train_indices, components, seed):
    if not components:
        return features, None
    components = int(components)
    maximum = min(len(train_indices), features.shape[1])
    if components > maximum:
        raise ValueError(f"cannot fit {components} PCA components from {maximum} available")
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(features[train_indices])
    return pca.transform(features).astype(np.float32), float(pca.explained_variance_ratio_.sum())


def classifier(C, class_weight, max_iter):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter),
            ),
        ]
    )


def evaluate(features, table, split, task, arm, settings):
    selected = np.ones(len(table), dtype=bool)
    if task == "immune":
        selected = table["label"].isin(IMMUNE_CLASSES).to_numpy()
    train = np.flatnonzero(selected & (split == "train"))
    val = np.flatnonzero(selected & (split == "val"))
    test = np.flatnonzero(selected & (split == "test"))
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"{task}: an official data split is empty")

    explained_variance = None
    if arm == "uni":
        features, explained_variance = reduce_uni(
            features,
            train,
            settings.get("uni_pca_components", 256),
            settings.get("seed", 0),
        )

    y = table["label"].to_numpy()
    best_C, best_score = None, -np.inf
    for C in settings["Cs"]:
        model = classifier(C, settings.get("class_weight", "balanced"), settings.get("max_iter", 2000))
        model.fit(features[train], y[train])
        score = f1_score(y[val], model.predict(features[val]), average="macro", zero_division=0)
        print(f"[{task}/{arm}] C={C:g} valid macro-F1={score:.4f}")
        if score > best_score:
            best_C, best_score = float(C), float(score)

    fit_indices = np.concatenate([train, val])
    model = classifier(best_C, settings.get("class_weight", "balanced"), settings.get("max_iter", 2000))
    model.fit(features[fit_indices], y[fit_indices])
    predicted = model.predict(features[test])
    labels = np.asarray(model.classes_)
    per_class = f1_score(y[test], predicted, labels=labels, average=None, zero_division=0)
    metrics = {
        "task": task,
        "arm": arm,
        "n_features": features.shape[1],
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_classes": len(labels),
        "best_C": best_C,
        "val_macro_f1": best_score,
        "test_accuracy": accuracy_score(y[test], predicted),
        "test_balanced_accuracy": balanced_accuracy_score(y[test], predicted),
        "test_macro_f1": f1_score(y[test], predicted, average="macro", zero_division=0),
        "test_weighted_f1": f1_score(y[test], predicted, average="weighted", zero_division=0),
        "uni_pca_explained_variance": explained_variance,
    }
    class_rows = [
        {"task": task, "arm": arm, "class": label, "f1": score}
        for label, score in zip(labels, per_class)
    ]
    predictions = {
        "row_id": table.iloc[test]["row_id"].to_numpy(),
        "y_true": y[test],
        "y_pred": predicted,
        "classes": labels,
    }
    return metrics, class_rows, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    output_dir = Path(cfg["output_dir"])
    uni_dir = Path(cfg["embedding"]["output_dir"])
    uni_manifest = pd.read_csv(uni_dir / "manifest.csv")
    uni, table, split = load_uni(uni_manifest)
    validate_official_split(table, cfg["data"]["split"])
    arms = [args.arm] if args.arm else list(ARMS)
    all_metrics, all_classes = [], []
    for arm in arms:
        if arm == "uni":
            features = uni
        else:
            projection_manifest = pd.read_csv(
                output_dir / "embeddings" / "distilled" / "manifest.csv"
            )
            features = load_distilled(projection_manifest, table["row_id"].to_numpy())
        for task in cfg["probe"].get("tasks", ["all", "immune"]):
            metrics, class_rows, predictions = evaluate(
                features, table, split, task, arm, cfg["probe"]
            )
            result_dir = output_dir / "probes" / task / arm
            result_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([metrics]).to_csv(result_dir / "metrics.csv", index=False)
            pd.DataFrame(class_rows).to_csv(result_dir / "per_class_f1.csv", index=False)
            np.savez(result_dir / "predictions.npz", **predictions)
            all_metrics.append(metrics)
            all_classes.extend(class_rows)
            print(f"[{task}/{arm}] test macro-F1={metrics['test_macro_f1']:.4f}")

    summary_path = output_dir / "probes" / f"summary_{args.arm or 'all'}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(summary_path, index=False)
    pd.DataFrame(all_classes).to_csv(
        output_dir / "probes" / f"per_class_{args.arm or 'all'}.csv", index=False
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
