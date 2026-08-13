"""Build a task-specific DAG from harmonized classes and Cell Ontology terms."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from cellbridge.ontology import Ontology, OntologyMapping


@dataclass(frozen=True)
class TaskGraph:
    """Ontology ancestors plus one synthetic leaf per classifier class."""

    root: str
    classes: tuple[str, ...]
    class_nodes: tuple[str, ...]
    names: dict[str, str]
    parents: dict[str, tuple[str, ...]]
    children: dict[str, tuple[str, ...]]
    descendant_columns: dict[str, tuple[int, ...]]
    distances: tuple[dict[str, int], ...]

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self.children)

    @property
    def leaves(self) -> tuple[str, ...]:
        return self.class_nodes

    def class_node(self, class_label: str) -> str:
        try:
            return self.class_nodes[self.classes.index(str(class_label))]
        except ValueError as error:
            raise KeyError(f"unknown classifier class {class_label!r}") from error

    def descendant_classes(self, node: str) -> tuple[str, ...]:
        try:
            columns = self.descendant_columns[node]
        except KeyError as error:
            raise KeyError(f"unknown task-graph node {node!r}") from error
        return tuple(self.classes[column] for column in columns)

    def ancestors_for_column(self, column: int) -> dict[str, int]:
        if not 0 <= column < len(self.classes):
            raise IndexError(f"class column is out of bounds: {column}")
        return self.distances[column]


def _topological_order(
    root: str,
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> tuple[str, ...]:
    indegree = {node: len(node_parents) for node, node_parents in parents.items()}
    pending = deque([root])
    order = []
    while pending:
        node = pending.popleft()
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    if len(order) != len(parents):
        raise ValueError("task graph contains a cycle or unreachable node")
    return tuple(order)


def _ancestor_distances(
    node: str,
    parents: dict[str, set[str]],
) -> dict[str, int]:
    distances = {node: 0}
    pending = deque([node])
    while pending:
        current = pending.popleft()
        for parent in parents[current]:
            distance = distances[current] + 1
            if parent not in distances or distance < distances[parent]:
                distances[parent] = distance
                pending.append(parent)
    return distances


def build_task_graph(
    mapping: OntologyMapping,
    ontology: Ontology,
    classes,
) -> TaskGraph:
    """Attach each harmonized classifier class below its mapped ontology term."""
    mapping.validate_ontology(ontology)
    classes = tuple(map(str, classes))
    if not classes:
        raise ValueError("classifier classes must not be empty")
    class_terms = mapping.class_terms(classes)

    ontology_nodes = {ontology.root}
    for term in class_terms.values():
        ontology_nodes.update(ontology.ancestors(term))

    parents = {
        node: set(ontology.parents[node]) & ontology_nodes for node in ontology_nodes
    }
    children = {node: set() for node in ontology_nodes}
    for child, node_parents in parents.items():
        for parent in node_parents:
            children[parent].add(child)

    names = {node: ontology.names[node] for node in ontology_nodes}
    class_nodes = tuple(f"task-class:{column}" for column in range(len(classes)))
    for class_label, class_node in zip(classes, class_nodes):
        term = class_terms[class_label]
        parents[class_node] = {term}
        children[class_node] = set()
        children[term].add(class_node)
        names[class_node] = class_label

    order = _topological_order(ontology.root, parents, children)
    descendant_columns = {node: set() for node in order}
    for column, class_node in enumerate(class_nodes):
        descendant_columns[class_node].add(column)
    for node in reversed(order):
        for child in children[node]:
            descendant_columns[node].update(descendant_columns[child])
    if descendant_columns[ontology.root] != set(range(len(classes))):
        raise ValueError("task-graph root does not contain every classifier class")
    unexpected_leaves = [node for node in ontology_nodes if not children[node]]
    if unexpected_leaves:
        raise ValueError(
            f"ontology branches contain no classifier class: {sorted(unexpected_leaves)}"
        )

    return TaskGraph(
        root=ontology.root,
        classes=classes,
        class_nodes=class_nodes,
        names=names,
        parents={node: tuple(sorted(values)) for node, values in parents.items()},
        children={node: tuple(sorted(values)) for node, values in children.items()},
        descendant_columns={
            node: tuple(sorted(columns)) for node, columns in descendant_columns.items()
        },
        distances=tuple(
            _ancestor_distances(class_node, parents) for class_node in class_nodes
        ),
    )
