"""Reviewed Cell Ontology terms used by the HEST label hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

CL_RELEASE = "2026-03-26"
CL_SHA256 = "f513ca2abe1ce360d7a51d6a7b14473a82dded19f59b06684e46ced445b45577"


@dataclass(frozen=True)
class CellTerm:
    """A canonical ontology identity and its task-specific tree placement."""

    cl_id: str
    name: str
    compartment: str
    lineage: str


def _term(cl_id: str, name: str, compartment: str, lineage: str) -> CellTerm:
    return CellTerm(cl_id, name, compartment, lineage)


TERMS = {
    # Epithelial identities
    "epithelial": _term("CL:0000066", "epithelial cell", "epithelial", "epithelial"),
    "pancreatic_acinar": _term(
        "CL:0002064", "pancreatic acinar cell", "epithelial", "pancreatic epithelial"
    ),
    "pancreatic_ductal": _term(
        "CL:0002079", "pancreatic ductal cell", "epithelial", "pancreatic epithelial"
    ),
    "pancreatic_endocrine": _term(
        "CL:0008024", "pancreatic endocrine cell", "epithelial", "pancreatic epithelial"
    ),
    "mammary_epithelial": _term(
        "CL:0002327", "mammary gland epithelial cell", "epithelial", "mammary epithelial"
    ),
    "mammary_luminal": _term(
        "CL:0002326",
        "luminal epithelial cell of mammary gland",
        "epithelial",
        "mammary epithelial",
    ),
    "myoepithelial": _term(
        "CL:0000185", "myoepithelial cell", "epithelial", "myoepithelial"
    ),
    "ciliated_epithelial": _term(
        "CL:0000067", "ciliated epithelial cell", "epithelial", "ciliated epithelial"
    ),
    "fallopian_epithelial": _term(
        "CL:4052018", "fallopian tube epithelial cell", "epithelial", "fallopian epithelial"
    ),
    "neoplastic": _term(
        "CL:0001063", "neoplastic cell", "epithelial", "neoplastic epithelial"
    ),
    # Hematopoietic identities
    "lymphocyte": _term("CL:0000542", "lymphocyte", "hematopoietic", "lymphoid"),
    "t_cell": _term("CL:0000084", "T cell", "hematopoietic", "lymphoid"),
    "b_cell": _term("CL:0000236", "B cell", "hematopoietic", "lymphoid"),
    "plasma": _term("CL:0000786", "plasma cell", "hematopoietic", "lymphoid"),
    "cd4_memory": _term(
        "CL:0000897", "CD4-positive, alpha-beta memory T cell", "hematopoietic", "lymphoid"
    ),
    "cd8_memory": _term(
        "CL:0000909", "CD8-positive, alpha-beta memory T cell", "hematopoietic", "lymphoid"
    ),
    "effector_cd4": _term(
        "CL:0001044", "effector CD4-positive, alpha-beta T cell", "hematopoietic", "lymphoid"
    ),
    "effector_cd8": _term(
        "CL:0001050", "effector CD8-positive, alpha-beta T cell", "hematopoietic", "lymphoid"
    ),
    "ilc": _term("CL:0001065", "innate lymphoid cell", "hematopoietic", "lymphoid"),
    "mature_nkt": _term(
        "CL:0000814", "mature natural killer T cell", "hematopoietic", "lymphoid"
    ),
    "memory_b": _term("CL:0000787", "memory B cell", "hematopoietic", "lymphoid"),
    "naive_b": _term("CL:0000788", "naive B cell", "hematopoietic", "lymphoid"),
    "naive_cd4": _term(
        "CL:0000895",
        "naive thymus-derived CD4-positive, alpha-beta T cell",
        "hematopoietic",
        "lymphoid",
    ),
    "reg_t": _term("CL:0000815", "regulatory T cell", "hematopoietic", "lymphoid"),
    "type1_nkt": _term(
        "CL:0000921", "type I natural killer T cell", "hematopoietic", "lymphoid"
    ),
    "myeloid": _term("CL:0000763", "myeloid cell", "hematopoietic", "myeloid"),
    "macrophage": _term("CL:0000235", "macrophage", "hematopoietic", "myeloid"),
    "mast": _term("CL:0000097", "mast cell", "hematopoietic", "myeloid"),
    "dendritic": _term("CL:0000451", "dendritic cell", "hematopoietic", "myeloid"),
    "cd141_dc": _term(
        "CL:0002394", "CD141-positive myeloid dendritic cell", "hematopoietic", "myeloid"
    ),
    "cd1c_dc": _term(
        "CL:0002399", "CD1c-positive myeloid dendritic cell", "hematopoietic", "myeloid"
    ),
    "mature_dc": _term(
        "CL:0000841", "mature conventional dendritic cell", "hematopoietic", "myeloid"
    ),
    "pdc": _term(
        "CL:0000784", "plasmacytoid dendritic cell", "hematopoietic", "myeloid"
    ),
    "classical_monocyte": _term(
        "CL:0000860", "classical monocyte", "hematopoietic", "myeloid"
    ),
    "intermediate_monocyte": _term(
        "CL:0002393", "intermediate monocyte", "hematopoietic", "myeloid"
    ),
    "nonclassical_monocyte": _term(
        "CL:0000875", "non-classical monocyte", "hematopoietic", "myeloid"
    ),
    "neutrophil": _term("CL:0000775", "neutrophil", "hematopoietic", "myeloid"),
    "erythrocyte": _term("CL:0000232", "erythrocyte", "hematopoietic", "erythroid"),
    "hsc": _term(
        "CL:0000037", "hematopoietic stem cell", "hematopoietic", "stem/progenitor"
    ),
    # Stromal and vascular identities
    "stromal": _term("CL:0000499", "stromal cell", "stromal", "stromal"),
    "fibroblast": _term("CL:0000057", "fibroblast", "stromal", "fibroblast"),
    "smooth_muscle": _term("CL:0000192", "smooth muscle cell", "stromal", "mural"),
    "perivascular": _term("CL:4033054", "perivascular cell", "stromal", "mural"),
    "pericyte": _term("CL:0000669", "pericyte", "stromal", "mural"),
    "adipocyte": _term("CL:0000136", "adipocyte", "stromal", "adipocyte"),
    "endothelial": _term(
        "CL:0000115", "endothelial cell", "vascular", "blood endothelial"
    ),
    "lymphatic_endothelial": _term(
        "CL:0002138",
        "endothelial cell of lymphatic vessel",
        "vascular",
        "lymphatic endothelial",
    ),
}
