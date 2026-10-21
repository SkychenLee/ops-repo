# SPDX-License-Identifier: Apache-2.0
"""Fully-sharded w2 (input-sharded) accuracy tests; no vLLM imports.

Adapted from vllm-ascend tests/ut/lora/test_fully_sharded_two_op.py
(test_w2 branches, test_w2_partial_expand_fold). The cross-rank
collectives live in the serving framework, not in these operators, so
each rank's projection runs locally and torch.cat / summation models the
all_gather / all_reduce the framework would perform between shrink and
expand. CPU fp32 references are independent of the migrated kernels.
Tests do not select or probe any NPU during collection.

Run on an available device after compiling/installing the library:
  OPS_REPO_RUN_NPU_TESTS=1 OPS_REPO_TEST_DEVICE=npu:0 python -m pytest tests/npu -q
"""

import os

import pytest

if os.environ.get("OPS_REPO_RUN_NPU_TESTS") != "1":
    pytest.skip("NPU tests require explicit OPS_REPO_RUN_NPU_TESTS=1", allow_module_level=True)

import torch
import torch_npu  # noqa: F401
import ops_repo  # noqa: F401

DEV = os.environ.get("OPS_REPO_TEST_DEVICE", "npu:0")

H, I, R, E, L, TP = 1024, 1024, 64, 4, 2, 8
I_LOC, H_LOC = I // TP, H // TP
ATOL = 3e-2


@pytest.fixture(scope="module", autouse=True)
def _select_device():
    torch.npu.set_device(DEV)


def _flag(value):
    # Control flags are intentionally CPU tensors: C++ branches on their data.
    return torch.tensor(value, dtype=torch.bool)


def _routing(tokens, seed):
    """Rows sorted by expert (gmm requirement); mixed slots including -1."""
    generator = torch.Generator().manual_seed(seed)
    slots = torch.randint(-1, L, (tokens,), generator=generator)
    experts = torch.sort(torch.randint(0, E, (tokens,), generator=generator)).values
    combined = torch.where(slots >= 0, slots * E + experts, torch.full_like(slots, -1))
    counts = torch.bincount(experts, minlength=E).to(torch.int64)
    return slots, experts, combined, counts


def _random(shape, seed, scale, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * scale).to(dtype)


def _shrink_partial(x_r, a, slots, experts):
    """fp32 CPU reference for one rank's partial shrink [T, R]."""
    out = torch.zeros(x_r.shape[0], R)
    for row in range(x_r.shape[0]):
        slot = int(slots[row])
        if slot >= 0:
            out[row] = x_r[row].float() @ a[slot, int(experts[row])].float().T
    return out


def _expand_reference(y0, z, b, slots, experts, offset=0):
    """y0 + z @ B into the [offset, offset + H_LOC) slice, fp32 CPU."""
    ref = y0.float().clone()
    for row in range(z.shape[0]):
        slot = int(slots[row])
        if slot >= 0:
            ref[row, offset:offset + H_LOC] += z[row] @ b[slot, int(experts[row])].float().T
    return ref


def _fs_buffers(tokens):
    """One fully-sharded w2 setup: per-rank input shards, shared A layout."""
    slots, experts, combined, counts = _routing(tokens, 11)
    x_r = [_random((tokens, I_LOC), 31 + rank, 0.3) for rank in range(TP)]
    a_r = [_random((L, E, R, I_LOC), 71 + rank, 0.05) for rank in range(TP)]
    b_r = [_random((L, E, H_LOC, R), 91 + rank, 0.05) for rank in range(TP)]
    return slots, experts, combined.to(DEV), counts.to(DEV), x_r, a_r, b_r


@pytest.mark.parametrize("use_gmm", [False, True], ids=["bgmv", "gmm"])
def test_w2_input_sharded_matches_reference(use_gmm):
    """Sum of per-rank partial shrinks (the framework's all-reduce) feeds a
    single expand; must equal the full-input projection."""
    tokens = 48
    slots, experts, combined, counts, x_r, a_r, b_r = _fs_buffers(tokens)
    y0 = _random((tokens, H_LOC), 17, 0.2)

    partials = []
    for rank in range(TP):
        buffers = [torch.zeros(tokens, L * R, device=DEV, dtype=torch.float32),
                   torch.zeros(1, tokens, R, device=DEV, dtype=torch.float32)]
        assert ops_repo.add_lora_shrink(
            buffers, x_r[rank].to(DEV), [a_r[rank].to(DEV)], combined, counts,
            combined, 1.0, _flag(use_gmm), _flag(False), True, 1) is None
        partials.append(buffers[0] if use_gmm else buffers[1])
    # all_reduce: sum the buffers that were actually written
    summed = torch.stack(partials).sum(0)

    y = y0.to(DEV).clone()
    # the bgmv expand reads the stacked buffer (last list element); the gmm
    # expand reads the per-slice flat buffers
    buffers = [summed, torch.empty(1, tokens, R, device=DEV, dtype=torch.float32)] \
        if use_gmm else [torch.empty(tokens, L * R, device=DEV, dtype=torch.float32), summed]
    ops_repo.add_lora_expand(
        y, buffers, [b_r[0].to(DEV)], combined, counts, combined, [H_LOC], 0,
        True, _flag(use_gmm), _flag(False), True, 1)

    full_z = torch.zeros(tokens, R)
    for rank in range(TP):
        full_z += _shrink_partial(x_r[rank], a_r[rank], slots, experts)
    expected = _expand_reference(y0, full_z, b_r[0], slots, experts)
    torch.testing.assert_close(y.cpu().float(), expected, atol=ATOL, rtol=0)

    if use_gmm:
        # the standalone delta must reproduce the same correction without
        # touching its gate_up argument
        frozen = y.cpu().clone()
        delta = ops_repo.add_lora_expand_delta(
            y, buffers, [b_r[0].to(DEV)], counts, 1)
        assert delta.data_ptr() != y.data_ptr()
        assert torch.equal(y.cpu(), frozen)
        torch.testing.assert_close(delta.cpu().float(), expected - y0.float(),
                                   atol=ATOL, rtol=0)


@pytest.mark.parametrize("use_gmm", [False, True], ids=["bgmv", "gmm"])
def test_w2_partial_expand_rank_slices(use_gmm):
    """partial_expand targets a tp_rank output slice of the full-width
    partial down output; the framework's final all-reduce sums the ranks.
    Each rank's slice must carry exactly its own block delta and leave the
    other slices untouched."""
    tokens = 32
    slots, experts, combined, counts, x_r, a_r, b_r = _fs_buffers(tokens)
    y0 = _random((tokens, H), 23, 0.2).to(DEV)

    rank_outputs = []
    for rank in range(TP):
        buffers = [torch.zeros(tokens, L * R, device=DEV, dtype=torch.float32),
                   torch.zeros(1, tokens, R, device=DEV, dtype=torch.float32)]
        ops_repo.add_lora_shrink(
            buffers, x_r[rank].to(DEV), [a_r[rank].to(DEV)], combined, counts,
            combined, 1.0, _flag(use_gmm), _flag(False), True, 1)
        partial = buffers[0] if use_gmm else buffers[1]
        buffers = [torch.empty(tokens, L * R, device=DEV, dtype=torch.float32), partial] \
            if not use_gmm else [partial, buffers[1]]
        y_rank = y0.clone()
        offset = rank * H_LOC
        ops_repo.add_lora_expand(
            y_rank, buffers, [b_r[rank].to(DEV)], combined, counts, combined,
            [H_LOC], offset, True, _flag(use_gmm), _flag(False), True, 1)
        rank_outputs.append(y_rank)

    total = torch.stack(rank_outputs).sum(0)
    for rank in range(TP):
        z = _shrink_partial(x_r[rank], a_r[rank], slots, experts)
        expected_slice = _expand_reference(
            y0.cpu(), z, b_r[rank], slots, experts, offset=rank * H_LOC)
        actual = rank_outputs[rank].cpu()
        # this rank's own slice carries y0 + its delta ...
        torch.testing.assert_close(
            actual[:, rank * H_LOC:(rank + 1) * H_LOC].float(),
            expected_slice[:, rank * H_LOC:(rank + 1) * H_LOC], atol=ATOL, rtol=0)
        # ... and every other slice is left untouched
        mask = torch.ones(H, dtype=torch.bool)
        mask[rank * H_LOC:(rank + 1) * H_LOC] = False
        assert torch.equal(actual[:, mask].cpu(), y0.cpu()[:, mask])

    # after the final all-reduce: y0 counted TP times + every rank's delta
    full_z = torch.zeros(tokens, R)
    for rank in range(TP):
        full_z += _shrink_partial(x_r[rank], a_r[rank], slots, experts)
    reduced_expected = y0.cpu().float() * TP
    for rank in range(TP):
        z = _shrink_partial(x_r[rank], a_r[rank], slots, experts)
        blk = torch.zeros(tokens, H_LOC)
        for row in range(tokens):
            slot = int(slots[row])
            if slot >= 0:
                blk[row] = z[row] @ b_r[rank][slot, int(experts[row])].float().T
        reduced_expected[:, rank * H_LOC:(rank + 1) * H_LOC] += blk
    torch.testing.assert_close(total.cpu().float(), reduced_expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("use_gmm", [False, True], ids=["bgmv", "gmm"])
def test_expand_does_not_mutate_shrink_buffers(use_gmm):
    """The serving framework may replay one shrink into several expands
    (phase split / partial expand), so the expand must only read the
    buffers."""
    tokens = 24
    slots, experts, combined, counts, x_r, a_r, b_r = _fs_buffers(tokens)
    buffers = [torch.zeros(tokens, L * R, device=DEV, dtype=torch.float32),
               torch.zeros(1, tokens, R, device=DEV, dtype=torch.float32)]
    ops_repo.add_lora_shrink(
        buffers, x_r[0].to(DEV), [a_r[0].to(DEV)], combined, counts, combined,
        1.0, _flag(use_gmm), _flag(False), True, 1)
    active = buffers[0] if use_gmm else buffers[1]
    frozen = active.clone()

    y = _random((tokens, H_LOC), 29, 0.2).to(DEV)
    expand_buffers = [torch.empty_like(buffers[0]), active]
    ops_repo.add_lora_expand(
        y, expand_buffers, [b_r[0].to(DEV)], combined,
        counts, combined, [H_LOC], 0, True, _flag(use_gmm), _flag(False), True, 1)
    assert torch.equal(active.cpu(), frozen.cpu()), "expand mutated the shrink buffer"
    if use_gmm:
        delta = ops_repo.add_lora_expand_delta(
            y, [active, buffers[1]], [b_r[0].to(DEV)], counts, 1)
        assert torch.equal(active.cpu(), frozen.cpu()), "expand_delta mutated the shrink buffer"
