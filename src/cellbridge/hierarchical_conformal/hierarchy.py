"""Small validated DAG representation for phenotype hierarchies."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Hierarchy:
    root: str
    children: dict[str, tuple[str, ...]]
    parents: dict[str, tuple[str, ...]]
    depths: dict[str, int]

    @classmethod
    def from_edges(cls, root: str, edges: Iterable[Iterable[str]]) -> Hierarchy:
        root = str(root)
        child_sets: dict[str, set[str]] = {}
        parent_sets: dict[str, set[str]] = {}
        for edge in edges:
            pair = tuple(map(str, edge))
            if len(pair) != 2:
                raise ValueError(f"hierarchy edge must have two nodes: {pair}")
            parent, child = pair
            if parent == child:
                raise ValueError(f"hierarchy contains a self-edge at {parent!r}")
            child_sets.setdefault(parent, set()).add(child)
            child_sets.setdefault(child, set())
            parent_sets.setdefault(child, set()).add(parent)
            parent_sets.setdefault(parent, set())

        nodes = set(child_sets)
        if root not in nodes:
            raise ValueError(f"hierarchy root {root!r} is absent")
        roots = sorted(node for node in nodes if not parent_sets[node])
        if roots != [root]:
            raise ValueError(f"hierarchy must have one root {root!r}; found {roots}")

        indegree = {node: len(parent_sets[node]) for node in nodes}
        depths = {root: 0}
        pending = deque([root])
        visited = []
        while pending:
            node = pending.popleft()
            visited.append(node)
            for child in child_sets[node]:
                depth = depths[node] + 1
                depths[child] = min(depths.get(child, depth), depth)
                indegree[child] -= 1
                if indegree[child] == 0:
                    pending.append(child)
        if len(visited) != len(nodes):
            raise ValueError("hierarchy contains a cycle or unreachable node")

        return cls(
            root=root,
            children={
                node: tuple(sorted(values)) for node, values in child_sets.items()
            },
            parents={
                node: tuple(sorted(values)) for node, values in parent_sets.items()
            },
            depths=depths,
        )

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self.children)

    @property
    def leaves(self) -> tuple[str, ...]:
        return tuple(
            sorted(node for node, children in self.children.items() if not children)
        )

    @property
    def is_tree(self) -> bool:
        return all(len(parents) <= 1 for parents in self.parents.values())

    def ancestors(self, node: str) -> tuple[str, ...]:
        """Return ancestors including the node itself."""
        if node not in self.children:
            raise KeyError(f"unknown hierarchy node {node!r}")
        found, pending = {node}, [node]
        while pending:
            current = pending.pop()
            for parent in self.parents[current]:
                if parent not in found:
                    found.add(parent)
                    pending.append(parent)
        return tuple(sorted(found))

    def descendant_leaves(self, node: str) -> tuple[str, ...]:
        if node not in self.children:
            raise KeyError(f"unknown hierarchy node {node!r}")
        leaves, pending = set(), [node]
        while pending:
            current = pending.pop()
            if not self.children[current]:
                leaves.add(current)
            else:
                pending.extend(self.children[current])
        return tuple(sorted(leaves))

    def distance(self, ancestor: str, descendant: str) -> int:
        pending = deque([(ancestor, 0)])
        seen = {ancestor}
        while pending:
            node, distance = pending.popleft()
            if node == descendant:
                return distance
            for child in self.children[node]:
                if child not in seen:
                    seen.add(child)
                    pending.append((child, distance + 1))
        raise ValueError(f"{ancestor!r} is not an ancestor of {descendant!r}")

    def validate_classes(self, classes) -> None:
        observed = set(map(str, classes))
        expected = set(self.leaves)
        if observed != expected:
            raise ValueError(
                "classifier classes must equal hierarchy leaves; "
                f"missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
