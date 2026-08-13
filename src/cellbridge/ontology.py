"""Validated dataset mappings and pinned Cell Ontology topology."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Ontology:
    """A rooted Cell Ontology DAG keyed by stable term IDs."""

    release: str
    source: str
    sha256: str
    root: str
    names: dict[str, str]
    parents: dict[str, tuple[str, ...]]
    children: dict[str, tuple[str, ...]]

    def ancestors(self, node: str) -> tuple[str, ...]:
        """Return ``node`` and every ancestor reachable from it."""
        if node not in self.parents:
            raise KeyError(f"unknown ontology term {node!r}")
        found, pending = {node}, [node]
        while pending:
            current = pending.pop()
            for parent in self.parents[current]:
                if parent not in found:
                    found.add(parent)
                    pending.append(parent)
        return tuple(sorted(found))


@dataclass(frozen=True)
class OntologyMapping:
    """One dataset's exact source-label to Cell Ontology mapping."""

    dataset: str
    release: str
    sha256: str
    source_to_class: dict[str, str]
    class_to_term: dict[str, str]
    class_to_term_name: dict[str, str]
    excluded_classes: frozenset[str]

    def map_labels(self, labels) -> tuple[np.ndarray, np.ndarray]:
        """Return harmonized labels and a mask excluding non-ontology classes."""
        labels = np.asarray(labels).astype(str)
        if labels.ndim != 1:
            raise ValueError("source labels must be one-dimensional")
        unknown = sorted(set(labels) - set(self.source_to_class))
        if unknown:
            raise ValueError(f"unmapped source labels: {unknown}")
        harmonized = np.asarray(
            [self.source_to_class[label] for label in labels],
            dtype=str,
        )
        keep = ~np.isin(harmonized, tuple(self.excluded_classes))
        return harmonized, keep

    def class_terms(self, classes: Iterable[str]) -> dict[str, str]:
        """Resolve the exact included classifier classes to ontology terms."""
        classes = tuple(map(str, classes))
        if len(set(classes)) != len(classes):
            raise ValueError("classifier classes must be unique")
        unknown = sorted(set(classes) - set(self.class_to_term))
        if unknown:
            raise ValueError(f"classifier classes are unmapped or excluded: {unknown}")
        return {label: self.class_to_term[label] for label in classes}

    def validate_ontology(self, ontology: Ontology) -> None:
        if ontology.release != self.release:
            raise ValueError(
                f"ontology release {ontology.release!r} does not match "
                f"mapping release {self.release!r}"
            )
        if self.sha256 and ontology.sha256 and ontology.sha256 != self.sha256:
            raise ValueError("ontology SHA-256 does not match the dataset mapping")
        missing = sorted(set(self.class_to_term.values()) - set(ontology.names))
        if missing:
            raise ValueError(f"mapped terms are absent from the ontology: {missing}")
        mismatched = sorted(
            class_label
            for class_label, term in self.class_to_term.items()
            if ontology.names[term] != self.class_to_term_name[class_label]
        )
        if mismatched:
            raise ValueError(
                f"mapped ontology IDs and names disagree for classes: {mismatched}"
            )


def _read_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def load_mapping(path: str | Path) -> OntologyMapping:
    """Load the reviewed mapping schema used by label harmonization."""
    path = Path(path)
    config = _read_yaml(path)
    labels = config.get("labels")
    ontology = config.get("ontology")
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"{path}: labels must be a non-empty mapping")
    if not isinstance(ontology, dict):
        raise ValueError(f"{path}: ontology must be a mapping")

    source_to_class: dict[str, str] = {}
    class_to_term: dict[str, str] = {}
    class_to_term_name: dict[str, str] = {}
    excluded_classes = set()
    for class_label, entry in labels.items():
        class_label = str(class_label)
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: mapping for {class_label!r} must be a mapping")
        source_labels = entry.get("source_labels")
        if not isinstance(source_labels, list) or not source_labels:
            raise ValueError(f"{path}: {class_label!r} requires source_labels")
        for source_label in map(str, source_labels):
            if source_label in source_to_class:
                raise ValueError(
                    f"{path}: source label is mapped twice: {source_label!r}"
                )
            source_to_class[source_label] = class_label

        term = entry.get("ontology_id")
        term_name = entry.get("ontology_name")
        excluded = bool(entry.get("excluded", False))
        if (term is None) != (term_name is None):
            raise ValueError(
                f"{path}: {class_label!r} has inconsistent ontology ID and name"
            )
        if excluded != (term is None):
            raise ValueError(
                f"{path}: {class_label!r} has inconsistent excluded/ontology_id"
            )
        if excluded:
            excluded_classes.add(class_label)
        else:
            class_to_term[class_label] = str(term)
            class_to_term_name[class_label] = str(term_name)

    dataset = config.get("dataset")
    release = ontology.get("release")
    if not dataset or not release:
        raise ValueError(f"{path}: dataset and ontology.release are required")
    return OntologyMapping(
        dataset=str(dataset),
        release=str(release),
        sha256=str(ontology.get("sha256", "")),
        source_to_class=source_to_class,
        class_to_term=class_to_term,
        class_to_term_name=class_to_term_name,
        excluded_classes=frozenset(excluded_classes),
    )


def load_ontology(path: str | Path) -> Ontology:
    """Load and validate a compact, rooted ontology snapshot."""
    path = Path(path)
    config = _read_yaml(path)
    terms = config.get("terms")
    root_entry = config.get("root")
    if not isinstance(terms, dict) or not terms:
        raise ValueError(f"{path}: terms must be a non-empty mapping")
    if not isinstance(root_entry, dict) or not root_entry.get("id"):
        raise ValueError(f"{path}: root.id is required")

    root = str(root_entry["id"])
    names: dict[str, str] = {}
    parents: dict[str, tuple[str, ...]] = {}
    for term, entry in terms.items():
        term = str(term)
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(f"{path}: invalid ontology term {term!r}")
        names[term] = str(entry["name"])
        term_parents = tuple(sorted(set(map(str, entry.get("parents", ())))))
        if term in term_parents:
            raise ValueError(f"{path}: ontology contains a self-edge at {term!r}")
        parents[term] = term_parents

    if root not in parents:
        raise ValueError(f"{path}: root {root!r} is absent from terms")
    missing = sorted(
        {parent for term_parents in parents.values() for parent in term_parents}
        - set(parents)
    )
    if missing:
        raise ValueError(f"{path}: ontology contains unknown parents: {missing}")
    roots = sorted(term for term, term_parents in parents.items() if not term_parents)
    if roots != [root]:
        raise ValueError(f"{path}: expected root {root!r}; found roots {roots}")

    children_sets = {term: set() for term in parents}
    indegree = {term: len(term_parents) for term, term_parents in parents.items()}
    for child, term_parents in parents.items():
        for parent in term_parents:
            children_sets[parent].add(child)
    pending = deque([root])
    visited = []
    while pending:
        node = pending.popleft()
        visited.append(node)
        for child in sorted(children_sets[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    if len(visited) != len(parents):
        raise ValueError(f"{path}: ontology contains a cycle or unreachable term")

    configured_root_name = root_entry.get("name")
    if configured_root_name and names[root] != str(configured_root_name):
        raise ValueError(f"{path}: root ID and name disagree")
    return Ontology(
        release=str(config.get("release", "")),
        source=str(config.get("source", "")),
        sha256=str(config.get("sha256", "")),
        root=root,
        names=names,
        parents=parents,
        children={
            term: tuple(sorted(term_children))
            for term, term_children in children_sets.items()
        },
    )
