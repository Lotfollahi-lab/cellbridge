"""Deterministic harmonization of the 91 normalized HEST test labels."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .taxonomy import TERMS

_CLUSTER_PREFIX = re.compile(r"^(?P<cluster>\d+)_(?P<label>.+)$")


@dataclass(frozen=True)
class LabelRule:
    term: str | None
    status: str = "full"
    state: str = ""
    malignancy: str = ""
    context: str = ""
    note: str = ""


RULES: dict[str, LabelRule] = {}


def _add(labels: str | tuple[str, ...], term: str | None, **kwargs) -> None:
    labels = (labels,) if isinstance(labels, str) else labels
    rule = LabelRule(term, **kwargs)
    for label in labels:
        if label in RULES:
            raise ValueError(f"duplicate hierarchy rule for {label!r}")
        RULES[label] = rule


# Epithelial identities. Dataset-specific states never become hierarchy nodes.
_add("Acinar", "pancreatic_acinar")
_add("Ductal", "pancreatic_ductal")
_add(("Endocrine 1", "Endocrine 2"), "pancreatic_endocrine")
_add("Alveolar_Epithelial_Cells", "mammary_epithelial", state="alveolar")
_add("Ductal_Epithelial_Cells", "mammary_luminal", state="ductal")
_add("Ciliated Epithelial Cells", "ciliated_epithelial")
_add("Fallopian Tube Epithelium", "fallopian_epithelial")
_add(("Myoepithelial", "Myoepithelial_Cells"), "myoepithelial")
_add(
    "Normal_Myoepithelial_Cells",
    "myoepithelial",
    malignancy="non-malignant",
    context="normal",
)
_add(
    "Tumor_Adjacent_Myoepithelial_Cells", "myoepithelial", context="tumor-adjacent"
)
_add(
    "Normal_Epithelial_Cells",
    "mammary_epithelial",
    malignancy="non-malignant",
    context="normal",
)
_add(
    "Epithelial_&_Myoepithelial_Cells",
    "epithelial",
    status="partial",
    context="mixed epithelial/myoepithelial",
    note="Composite label; retained only to the epithelial lineage.",
)
_add(
    "Epithelial_&_Tumor_Cells",
    "epithelial",
    status="partial",
    malignancy="ambiguous",
    context="mixed epithelial/tumor",
    note="Composite label; malignancy cannot be assigned per cell.",
)
_add(
    "Metaplastic Cells",
    "epithelial",
    status="partial",
    state="metaplastic",
    note="A state label without a defensible finer epithelial identity.",
)
_add(
    "Urothelial-like Cells",
    "epithelial",
    status="partial",
    state="urothelial-like",
    note="Similarity wording is not treated as an exact urothelial identity.",
)

_TUMOR = {
    "CFTR- Tumor Cells": "CFTR-negative",
    "Inflammatory Tumor Cells": "inflammatory",
    "Inflammatory_Tumor_Cells": "inflammatory",
    "Proliferating_Tumor_Cells": "proliferating",
    "Proliferative Tumor Cells": "proliferating",
    "Proliferative_Tumor Cells": "proliferating",
    "Proliferative_Tumor_Cells": "proliferating",
    "SOX2-OT+ Tumor Cells": "SOX2-OT-positive",
    "VEGFA+ Tumor Cells": "VEGFA-positive",
}
_add(("Tumor Cells", "Tumor_Cells"), "neoplastic", malignancy="malignant")
for _label, _state in _TUMOR.items():
    _add(_label, "neoplastic", state=_state, malignancy="malignant")
_add(
    "Malignant Cells Lining Cyst",
    "neoplastic",
    malignancy="malignant",
    context="cyst-lining",
)

# Lymphoid identities.
_add(
    "Activated_Lymphoid_Cells",
    "lymphocyte",
    status="partial",
    state="activated",
    note="Activation does not distinguish T, B, or NK identity.",
)
_add(("B Cell", "B Cells", "B_Lymphocytes"), "b_cell")
_add("CD4+ αβ Memory T Cell", "cd4_memory", state="memory")
_add("CD8+ αβ Memory T Cell", "cd8_memory", state="memory")
_add("Effector CD4+ αβ T Cell", "effector_cd4", state="effector")
_add("Effector CD8+ αβ T Cell", "effector_cd8", state="effector")
_add("Effector_T_Lymphocytes", "t_cell", state="effector")
_add("Innate Lymphoid Cell", "ilc")
_add("Mature NK T Cell", "mature_nkt")
_add("Memory B Cell", "memory_b", state="memory")
_add("Naive B Cell", "naive_b", state="naive")
_add("Naive Thymus Derived CD4+ αβ T Cell", "naive_cd4", state="naive")
_add("Plasma Cell", "plasma")
_add("Plasma_Cells", "plasma")
_add("Regulatory T Cell", "reg_t", state="regulatory")
_add(("T Cell", "T Cells", "T_Lymphocytes"), "t_cell")
_add(
    "T and NK Cells",
    "lymphocyte",
    status="partial",
    note="Composite label; retained only to the lymphoid lineage.",
)
_add("Type I NK T Cell", "type1_nkt")

# Myeloid and other hematopoietic identities.
_add("Activated_Macrophages", "macrophage", state="activated")
_add("CD141+ Myeloid Dendritic Cell", "cd141_dc")
_add("CD1c+ Myeloid Dendritic Cell", "cd1c_dc")
_add(("Dendritic_Cells",), "dendritic")
_add("Mature Conventional Dendritic Cell", "mature_dc")
_add("Plasmacytoid Dendritic Cell", "pdc")
_add("Classical Monocyte", "classical_monocyte")
_add("Intermediate Monocyte", "intermediate_monocyte")
_add("Non Classical Monocyte", "nonclassical_monocyte")
_add("Macrophage", "macrophage")
_add("Macrophages", "macrophage")
_add(("Mast Cell", "Mast Cells", "Mast_Cells"), "mast")
_add("Neutrophil", "neutrophil")
_add("Erythrocyte", "erythrocyte")
_add("Hematopoietic Stem Cell", "hsc")
_add(
    "Myeloid_Cells",
    "myeloid",
    status="partial",
    note="Broad source label; retained only to the myeloid lineage.",
)
_add(
    "MMP9+_Myeloid_Cells",
    "myeloid",
    status="partial",
    state="MMP9-positive",
    note="Marker does not establish a finer myeloid identity.",
)

# Stromal and vascular identities.
_add("Adipocytes", "adipocyte")
_add("Fibroblasts", "fibroblast")
_add("Pericytes", "pericyte")
_add("Perivascular_Cells", "perivascular")
_add("Smooth Muscle Cells", "smooth_muscle")
_add("CXCL14+_CAFs", "fibroblast", state="CXCL14-positive", context="cancer-associated")
_add("MMP11+_CAFs", "fibroblast", state="MMP11-positive", context="cancer-associated")
_add("Stromal Associated Fibroblasts", "fibroblast", context="stromal-associated")
_add("Tumor Associated Fibroblasts", "fibroblast", context="tumor-associated")
_add(
    ("Stromal Cell", "Stromal_Cells"),
    "stromal",
    status="partial",
    note="Broad source label; retained only to the stromal lineage.",
)
_add(("Endothelial", "Endothelial Cell", "Endothelial_Cells"), "endothelial")
_add(("Lymphatic Endothelial Cells", "Lymphatic_Endothelial_Cells"), "lymphatic_endothelial")
_add("Stromal Associated Endothelial Cells", "endothelial", context="stromal-associated")
_add("Tumor Associated Endothelial Cells", "endothelial", context="tumor-associated")

# Identity-free source labels are visible in outputs but never used as positives.
_add(
    "CXCL9/10 Cells",
    None,
    status="excluded",
    state="CXCL9/10-positive",
    note="Marker-only label with no defensible cell identity.",
)
_add(
    "Necrotic_Cells",
    None,
    status="excluded",
    state="necrotic",
    note="Necrosis is a state, not a cell identity.",
)
_add("Unassigned", None, status="excluded", note="No source identity was assigned.")


def split_source_label(raw_label: str) -> tuple[str, str]:
    """Return ``(numeric source cluster, biological label)``."""
    raw_label = str(raw_label).strip()
    match = _CLUSTER_PREFIX.match(raw_label)
    return (match.group("cluster"), match.group("label")) if match else ("", raw_label)


def map_label(raw_label: str) -> dict:
    """Map one source label without treating numeric cluster IDs as biology."""
    source_cluster, normalized = split_source_label(raw_label)
    if normalized not in RULES:
        raise KeyError(f"unreviewed HEST label: {raw_label!r} (normalized: {normalized!r})")
    rule = RULES[normalized]
    fields = asdict(rule)
    fields["mapping_status"] = fields.pop("status")
    record = {
        "raw_label": str(raw_label),
        "source_cluster": source_cluster,
        "normalized_label": normalized,
        **fields,
    }
    if rule.term is None:
        record.update(
            canonical_identity="",
            cl_id="",
            level_0="",
            level_1_compartment="",
            level_2_lineage="",
            level_3_identity="",
            hierarchy_depth=0,
            hierarchy_path="",
            use_in_hierarchy=False,
        )
        return record

    term = TERMS[rule.term]
    levels = ["cell", term.compartment, term.lineage]
    leaf = term.name if rule.status == "full" else ""
    if leaf:
        levels.append(leaf)
    record.update(
        canonical_identity=term.name,
        cl_id=term.cl_id,
        level_0="cell",
        level_1_compartment=term.compartment,
        level_2_lineage=term.lineage,
        level_3_identity=leaf,
        hierarchy_depth=3 if leaf else 2,
        hierarchy_path=" > ".join(levels),
        use_in_hierarchy=True,
    )
    return record
