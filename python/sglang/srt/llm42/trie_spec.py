"""GPU-ready tree layout for SSV4D trie verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class SSV4DTrieSpecInput:
    """Flattened tree metadata for one request verification window."""

    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    parent_indices: torch.Tensor
    depth: torch.Tensor
    retrieve_index: torch.Tensor
    num_tokens_per_req: int
    prefix_len: int
    token_paths: tuple[tuple[int, ...], ...]
    tree_page_table: torch.Tensor | None = None
    tree_cache_seqlens: torch.Tensor | None = None

    is_ssv4d_trie_spec = True

    def set_kv_layout(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> None:
        if len(page_table) != self.num_tokens_per_req:
            raise ValueError("Tree page table row count mismatch")
        if len(cache_seqlens) != self.num_tokens_per_req:
            raise ValueError("Tree cache length count mismatch")
        self.tree_page_table = page_table
        self.tree_cache_seqlens = cache_seqlens

    def get_spec_adjust_token_coefficient(self) -> tuple[int, int]:
        return self.num_tokens_per_req, self.num_tokens_per_req


def build_trie_spec(
    paths: Iterable[Iterable[int]],
    *,
    prefix_len: int,
    device: str = "cuda",
) -> SSV4DTrieSpecInput:
    """Build a causal tree mask over unique token prefixes.

    The root is the last verified context token. Each unique path token is one
    query/key node. Query ``i`` attends the common prefix and all of its own
    ancestors, but not sibling branches.
    """

    normalized = tuple(tuple(int(token) for token in path) for path in paths)
    if not normalized or not normalized[0]:
        raise ValueError("paths must contain at least one non-empty path")

    children: dict[tuple[int, ...], int] = {(): -1}
    nodes: list[tuple[int, ...]] = []
    parents: list[int] = []
    for path in normalized:
        for depth in range(1, len(path) + 1):
            prefix = path[:depth]
            if prefix in children:
                continue
            children[prefix] = len(nodes)
            nodes.append(prefix)
            parents.append(children[path[: depth - 1]])

    num_nodes = len(nodes)
    draft_token = torch.tensor(
        [prefix[-1] for prefix in nodes],
        dtype=torch.int64,
        device=device,
    )
    parent_indices = torch.tensor(
        parents,
        dtype=torch.int64,
        device=device,
    )
    depth = torch.tensor(
        [len(prefix) for prefix in nodes],
        dtype=torch.int64,
        device=device,
    )
    positions = depth + int(prefix_len) - 1

    custom_mask = torch.ones(
        (num_nodes, int(prefix_len) + num_nodes),
        dtype=torch.bool,
        device=device,
    )
    node_index = {prefix: index for index, prefix in enumerate(nodes)}
    for index, prefix in enumerate(nodes):
        custom_mask[index, int(prefix_len) :] = False
        ancestor = prefix[:-1]
        while ancestor:
            custom_mask[index, int(prefix_len) + node_index[ancestor]] = True
            ancestor = ancestor[:-1]

    max_path = max(len(path) for path in normalized)
    retrieve_index = torch.full(
        (len(normalized), max_path),
        -1,
        dtype=torch.int64,
        device=device,
    )
    for row, path in enumerate(normalized):
        for depth_index, token in enumerate(path):
            retrieve_index[row, depth_index] = children[path[: depth_index + 1]]

    return SSV4DTrieSpecInput(
        draft_token=draft_token,
        custom_mask=custom_mask.flatten(),
        positions=positions,
        parent_indices=parent_indices,
        depth=depth,
        retrieve_index=retrieve_index,
        num_tokens_per_req=num_nodes,
        prefix_len=int(prefix_len),
        token_paths=normalized,
    )


def build_tree_kv_layout(
    spec: SSV4DTrieSpecInput,
    prefix_cache_locs: torch.Tensor,
    tree_cache_locs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-node page tables and cache lengths for tree attention."""

    prefix_len = int(spec.prefix_len)
    num_nodes = spec.num_tokens_per_req
    if prefix_cache_locs.numel() != prefix_len:
        raise ValueError(
            f"Expected {prefix_len} prefix cache locs, "
            f"got {prefix_cache_locs.numel()}"
        )
    if tree_cache_locs.numel() < num_nodes:
        raise ValueError(
            f"Expected at least {num_nodes} tree cache locs, "
            f"got {tree_cache_locs.numel()}"
        )

    device = tree_cache_locs.device
    page_table = torch.zeros(
        (num_nodes, prefix_len + num_nodes),
        dtype=torch.int32,
        device=device,
    )
    page_table[:, :prefix_len] = prefix_cache_locs.to(torch.int32)
    cache_seqlens = torch.empty(
        num_nodes,
        dtype=torch.int32,
        device=device,
    )

    for index in range(num_nodes):
        ancestors = []
        parent = int(spec.parent_indices[index].item())
        while parent >= 0:
            ancestors.append(parent)
            parent = int(spec.parent_indices[parent].item())
        ancestors.reverse()
        path_nodes = ancestors + [index]
        page_table[
            index, prefix_len : prefix_len + len(path_nodes)
        ] = tree_cache_locs[path_nodes].to(torch.int32)
        cache_seqlens[index] = prefix_len + len(path_nodes)

    return page_table, cache_seqlens
