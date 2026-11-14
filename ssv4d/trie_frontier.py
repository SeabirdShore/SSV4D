"""Prefix-trie state machine with node-level verification frontiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class CandidateEdges:
    canonical_token: int
    candidate_tokens: tuple[int, ...]
    margin: float
    expanded: bool


def plan_candidate_edges(
    logprobs: Iterable[float],
    token_ids: Iterable[int],
    *,
    branch_threshold: float = 0.375,
    max_candidates: int = 8,
) -> CandidateEdges:
    """Return canonical or top-k candidate edges for one trie node."""

    probabilities = list(logprobs)
    tokens = [int(token) for token in token_ids]
    if not tokens:
        raise ValueError("token_ids must not be empty")
    canonical = tokens[0]
    margin = (
        float(probabilities[0] - probabilities[1])
        if len(probabilities) >= 2
        else 0.0
    )
    expanded = margin <= branch_threshold
    candidates = tokens[:max_candidates] if expanded else [canonical]
    return CandidateEdges(
        canonical_token=canonical,
        candidate_tokens=tuple(dict.fromkeys(candidates)),
        margin=margin,
        expanded=expanded,
    )


@dataclass
class TrieNode:
    node_id: int
    token_id: int | None
    depth: int
    parent_id: int | None
    children: dict[int, int] = field(default_factory=dict)
    canonical_count: int = 0
    shadow_count: int = 0
    verifier_count: int = 0
    verified: bool = False
    kv_page_ref: int | None = None
    candidate_labels: tuple[int, ...] = ()


@dataclass(frozen=True)
class RollbackTarget:
    depth: int
    node: TrieNode | None


class PrefixTrie:
    """A request-local prefix trie with verified-frontier tracking."""

    def __init__(self) -> None:
        root = TrieNode(
            node_id=0,
            token_id=None,
            depth=0,
            parent_id=None,
            verified=True,
        )
        self.nodes = {0: root}
        self.root = root
        self._path_cache: dict[tuple[int, ...], int] = {(): 0}

    def _child(
        self,
        parent: TrieNode,
        token_id: int,
    ) -> TrieNode:
        child_id = parent.children.get(token_id)
        if child_id is not None:
            return self.nodes[child_id]

        child = TrieNode(
            node_id=len(self.nodes),
            token_id=token_id,
            depth=parent.depth + 1,
            parent_id=parent.node_id,
        )
        self.nodes[child.node_id] = child
        parent.children[token_id] = child.node_id
        return child

    def insert(
        self,
        tokens: Iterable[int],
        *,
        source: str = "canonical",
        verified: bool = False,
    ) -> TrieNode:
        if source not in {"canonical", "shadow", "verifier"}:
            raise ValueError(f"Unknown source: {source}")

        prefix = []
        node = self.root
        for token in tokens:
            token = int(token)
            node = self._child(node, token)
            prefix.append(token)
            self._path_cache[tuple(prefix)] = node.node_id
            if source == "canonical":
                node.canonical_count += 1
            elif source == "shadow":
                node.shadow_count += 1
            else:
                node.verifier_count += 1
            if verified:
                node.verified = True
        return node

    def node_for_path(self, tokens: Iterable[int]) -> TrieNode | None:
        key = tuple(int(token) for token in tokens)
        node_id = self._path_cache.get(key)
        return self.nodes.get(node_id) if node_id is not None else None

    def verify_path(self, tokens: Iterable[int]) -> TrieNode:
        return self.insert(tokens, source="verifier", verified=True)

    def longest_verified_prefix(
        self,
        tokens: Iterable[int],
    ) -> RollbackTarget:
        node = self.root
        for depth, token in enumerate(tokens, start=1):
            child_id = node.children.get(int(token))
            if child_id is None:
                break
            child = self.nodes[child_id]
            if not child.verified:
                break
            node = child
        return RollbackTarget(depth=node.depth, node=node)

    def active_frontier(self) -> list[TrieNode]:
        """Return non-root leaves currently eligible for continuation."""

        leaves = [
            node
            for node in self.nodes.values()
            if node.node_id != 0 and not node.children
        ]
        return sorted(leaves, key=lambda node: (node.depth, node.node_id))

    def frontier_by_depth(self) -> dict[int, list[TrieNode]]:
        result: dict[int, list[TrieNode]] = {}
        for node in self.nodes.values():
            if node.node_id == 0:
                continue
            result.setdefault(node.depth, []).append(node)
        return result

    def branch_nodes(self) -> list[TrieNode]:
        return [
            node
            for node in self.nodes.values()
            if len(node.children) > 1
        ]

    def stats(self) -> dict[str, int | float]:
        branches = self.branch_nodes()
        frontier = self.frontier_by_depth()
        return {
            "nodes": len(self.nodes) - 1,
            "branch_nodes": len(branches),
            "max_branch_degree": max(
                (len(node.children) for node in self.nodes.values()),
                default=0,
            ),
            "max_frontier_width": max(
                (len(nodes) for nodes in frontier.values()),
                default=0,
            ),
            "leaf_count": len(self.active_frontier()),
        }
