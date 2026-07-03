import numpy as np


def per_gene_pcc(Y_true, Y_pred):
    """Pearson correlation per gene (column). Zero-variance genes -> NaN.

    Vectorized; never raises on constant columns -- a gene with no variance in
    either ``Y_true`` or ``Y_pred`` gets ``NaN`` instead of a divide-by-zero.
    """
    Yt = np.asarray(Y_true, dtype=np.float64)
    Yp = np.asarray(Y_pred, dtype=np.float64)
    if Yt.shape != Yp.shape:
        raise ValueError(f"shape mismatch: {Yt.shape} vs {Yp.shape}")

    Yt_c = Yt - Yt.mean(axis=0, keepdims=True)
    Yp_c = Yp - Yp.mean(axis=0, keepdims=True)
    num = (Yt_c * Yp_c).sum(axis=0)
    den = np.sqrt((Yt_c**2).sum(axis=0) * (Yp_c**2).sum(axis=0))

    out = np.full(Yt.shape[1], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def summarize(Y_true, Y_pred):
    pcc = per_gene_pcc(Y_true, Y_pred)
    Yt = np.asarray(Y_true, dtype=np.float64)
    Yp = np.asarray(Y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((Yt - Yp) ** 2)))
    return {
        "mean_pcc": float(np.nanmean(pcc)),
        "median_pcc": float(np.nanmedian(pcc)),
        "pct_genes_pcc_gt_0.3": float(100.0 * np.mean(pcc > 0.3)),
        "rmse": rmse,
        "n_genes": int(pcc.shape[0]),
        "n_valid_genes": int(np.count_nonzero(~np.isnan(pcc))),
    }
