"""Runtime branch and copy-on-write bookkeeping for trie verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ssv4d.trie_frontier import PrefixTrie, TrieNode


class PagePool:
    """Reference-counted page allocator used by the trie prototype."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.refcounts = [0] * capacity
        self._next_page = 0

    def allocate(self, count: int) -> tuple[int, ...]:
        start = self._next_page
        self._next_page += count
        if self._next_page > self.capacity:
            raise RuntimeError("Page pool capacity exceeded")
        pages = tuple(range(start, self._next_page))
        for page in pages:
            self.refcounts[page] = 1
        return pages

    def retain(self, pages: Iterable[int]) -> None:
        for page in pages:
            self.refcounts[page] += 1

    def release(self, pages: Iterable[int]) -> None:
        for page in pages:
            if self.refcounts[page] <= 0:
                raise RuntimeError(f"Page {page} is already free")
            self.refcounts[page] -= 1

    def live_pages(self) -> int:
        return sum(count > 0 for count in self.refcounts)


@dataclass
class BranchState:
    node: TrieNode
    pages: tuple[int, ...]
    active: bool = True


class TrieBranchManager:
    """Own active branches and copy-only divergent suffix pages."""

    def __init__(self, page_pool: PagePool | None = None) -> None:
        self.trie = PrefixTrie()
        self.page_pool = page_pool
        self.branches: dict[int, BranchState] = {}

    def add_branch(
        self,
        parent_pages: Iterable[int],
        tokens: Iterable[int],
        *,
        source: str = "shadow",
    ) -> BranchState:
        parent_pages = tuple(parent_pages)
        token_list = [int(token) for token in tokens]
        if self.page_pool is not None:
            self.page_pool.retain(parent_pages)
            suffix_pages = self.page_pool.allocate(len(token_list))
        else:
            suffix_pages = ()
        node = self.trie.insert(token_list, source=source)
        branch = BranchState(node=node, pages=parent_pages + suffix_pages)
        self.branches[node.node_id] = branch
        return branch

    def verify_path(self, tokens: Iterable[int]) -> TrieNode:
        node = self.trie.verify_path(tokens)
        for branch_id, branch in list(self.branches.items()):
            if branch_id != node.node_id:
                if self.page_pool is not None:
                    self.page_pool.release(branch.pages)
                branch.active = False
                del self.branches[branch_id]
        return node

    def add_candidate_edges(
        self,
        prefix: Iterable[int],
        candidate_tokens: Iterable[int],
    ) -> list[TrieNode]:
        prefix = [int(token) for token in prefix]
        nodes = []
        for token in candidate_tokens:
            nodes.append(
                self.trie.insert(prefix + [int(token)], source="shadow")
            )
        return nodes

    def record_candidate_labels(
        self,
        prefix: Iterable[int],
        candidate_tokens: Iterable[int],
    ) -> TrieNode:
        normalized_prefix = [int(token) for token in prefix]
        node = self.trie.node_for_path(normalized_prefix)
        if node is None:
            node = self.trie.insert(
                normalized_prefix, source="canonical"
            )
        node.candidate_labels = tuple(
            dict.fromkeys(int(token) for token in candidate_tokens)
        )
        return node

    def rollback(self, tokens: Iterable[int]) -> int:
        return self.trie.longest_verified_prefix(tokens).depth


class TreeVerificationPool:
    """Fixed scratch KV and request-pool storage for tree verification."""

    def __init__(
        self,
        fixed_size: int,
        node_capacity: int,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        device: str = "cuda",
    ) -> None:
        self.fixed_size = fixed_size
        self.node_capacity = node_capacity
        self.req_to_token_pool = req_to_token_pool

        slots_needed = fixed_size * node_capacity
        page_size = token_to_kv_pool_allocator.page_size
        if page_size > 1:
            slots_needed = (
                (slots_needed + page_size - 1) // page_size
            ) * page_size
        self.cache_locs = token_to_kv_pool_allocator.alloc(slots_needed)
        if self.cache_locs is None or not len(self.cache_locs):
            raise RuntimeError("Unable to allocate tree verification KV scratch")

        if len(req_to_token_pool.free_slots) < fixed_size:
            token_to_kv_pool_allocator.free(self.cache_locs)
            raise RuntimeError("Unable to allocate tree verification request slots")
        self.pool_indices = req_to_token_pool.free_slots[:fixed_size]
        req_to_token_pool.free_slots = req_to_token_pool.free_slots[fixed_size:]
        self.device_pool_indices = torch.tensor(
            self.pool_indices,
            dtype=torch.int64,
            device=device,
        )
        self.active = set()

    def acquire(self, slot: int) -> int:
        if slot < 0 or slot >= self.fixed_size:
            raise IndexError(f"Tree pool slot {slot} out of range")
        if slot in self.active:
            raise RuntimeError(f"Tree pool slot {slot} is already active")
        self.active.add(slot)
        return int(self.pool_indices[slot])

    def node_cache_locs(self, slot: int, count: int) -> torch.Tensor:
        if slot not in self.active:
            raise RuntimeError(f"Tree pool slot {slot} is not active")
        if count < 0 or count > self.node_capacity:
            raise ValueError("Invalid tree node capacity request")
        start = slot * self.node_capacity
        return self.cache_locs[start : start + count]

    def release(self, slot: int) -> None:
        if slot in self.active:
            self.active.remove(slot)

    def free(self, token_to_kv_pool_allocator) -> None:
        self.active.clear()
        if self.cache_locs is not None:
            token_to_kv_pool_allocator.free(self.cache_locs)
            self.cache_locs = None
        self.req_to_token_pool.free_slots.extend(self.pool_indices)
        self.pool_indices = []
