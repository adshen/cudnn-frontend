"""GPU-free unit tests for the SDPA facts analyzer and per-engine capability probes."""

from __future__ import annotations

import cudnn
import cudnn.sdpa  # noqa: F401 — registers the FROST DSL engines
import pytest
import torch

from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd import engines


def _eligible(graph):
    """Names of registered FROST SDPA engines whose caps match this graph."""
    return {s.name for s in engines.ENGINE_SPECS if engines.probe(s, graph)}


# The default pytest.ini addopts is `-m L0`; mark the whole module so it runs.
pytestmark = pytest.mark.L0

B, H, S, D = 2, 8, 256, 512
DIMS = (B, H, S, D)
STRIDES = (S * H * D, D, H * D, 1)
DTYPE = cudnn.data_type.HALF


@pytest.fixture(autouse=True)
def _fake_sm100(monkeypatch):
    """Fake an SM100 device so the device-family gate passes without a real GPU."""
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))


def _mk_graph() -> cudnn.pygraph:
    return cudnn.pygraph(
        io_data_type=DTYPE,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )


def _mk_qkv(g: cudnn.pygraph, d: int = D):
    dims = (B, H, S, d)
    strides = (S * H * d, d, H * d, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="v")
    return q, k, v, dims, strides


def _finish_output(o, dims, strides, dtype=DTYPE):
    """Set O's dims/stride/dtype directly, standing in for build_operation_graph()."""
    o.set_output(True).set_dim(dims).set_stride(strides)
    o.set_data_type(dtype)


def _facts(graph):
    facts = ga.analyze(graph)
    assert facts is not None, "expected a single SDPA node on the graph"
    assert facts.invalid is None, facts.invalid
    return facts


def test_engines_registered():
    for spec in engines.ENGINE_SPECS:
        assert spec.name in cudnn.frost.engine_names()
        assert cudnn.frost.is_frost_engine(spec.name)
    assert engines.engine_name(512) == "sdpa_fwd_prefill_sm100_d512"


def test_single_sdpa_node_found():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True)
    _finish_output(o, dims, strides)
    node = ga._single_sdpa_node(g)
    assert node is not None
    rec = ga._record_from_node(node)
    assert rec["q"] is q and rec["k"] is k and rec["v"] is v
    assert rec["o"] is o
    assert rec["use_causal_mask"] is True
    assert rec["attn_scale"] == 0.1


def test_probe_accepts_dsv4_causal():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True)
    _finish_output(o, dims, strides)
    assert engines.engine_name(512) in _eligible(g)


def test_probe_accepts_bf16():
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dims = (B, H, S, D)
    strides = (S * H * D, D, H * D, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.BFLOAT16, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.BFLOAT16, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.BFLOAT16, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides, dtype=cudnn.data_type.BFLOAT16)
    assert engines.engine_name(512) in _eligible(g)


def test_probe_rejects_non_dsv4_head_dim():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=128)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_wrong_device_family(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (8, 0))
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_bias():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    bias = g.tensor(dim=(1, H, S, S), stride=(H * S * S, S * S, S, 1), data_type=cudnn.data_type.FLOAT, name="bias")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, bias=bias)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_alibi():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_alibi_mask=True)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_second_op_on_graph():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides)
    r = g.relu(input=o, name="r")
    r.set_output(True).set_dim(dims).set_stride(strides)
    assert not _eligible(g)


def test_probe_rejects_padding_mask_without_seq_len_kv():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_padding_mask=True)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_accepts_seq_len_q_with_padding_mask():
    # padding_mask requires a seq_len_q companion; the engine accepts it (KV-only trim).
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_kv")
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_q")
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        use_padding_mask=True,
        seq_len_kv=seq_kv,
        seq_len_q=seq_q,
    )
    _finish_output(o, dims, strides)
    assert engines.engine_name(512) in _eligible(g)


def test_probe_rejects_seq_len_q_without_padding_mask():
    # Bare seq_len_q is per-batch Q trimming, which the kernel has no path for.
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_q")
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        seq_len_q=seq_q,
    )
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def _mk_thd_qkvo(g, *, mask_kwargs):
    """Build a ragged (THD) sdpa graph with the given mask kwargs."""
    q = g.tensor(dim=DIMS, stride=STRIDES, data_type=DTYPE, name="q")
    k = g.tensor(dim=DIMS, stride=STRIDES, data_type=DTYPE, name="k")
    v = g.tensor(dim=DIMS, stride=STRIDES, data_type=DTYPE, name="v")
    ro = g.tensor(dim=(B + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name="ro")
    q.set_ragged_offset(ro)
    k.set_ragged_offset(ro)
    v.set_ragged_offset(ro)
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="sq")
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="skv")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_padding_mask=True, seq_len_q=seq_q, seq_len_kv=seq_kv, **mask_kwargs)
    o.set_output(True).set_dim(DIMS).set_stride(STRIDES)
    o.set_data_type(DTYPE)
    o.set_ragged_offset(ro)


def test_probe_accepts_thd_top_left_causal():
    g = _mk_graph()
    _mk_thd_qkvo(g, mask_kwargs=dict(use_causal_mask=True))
    assert engines.engine_name(512) in _eligible(g)


def test_probe_rejects_thd_bottom_right():
    # THD + bottom-right causal is a kernel gap (BR diagonal needs global, not per-sequence, Q length).
    g = _mk_graph()
    _mk_thd_qkvo(g, mask_kwargs=dict(use_causal_mask_bottom_right=True))
    assert not _eligible(g)


def test_probe_rejects_right_band_widening():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, diagonal_band_right_bound=16)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_bad_sink_shape():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    bad_sink = g.tensor(dim=(1, H, 2, 1), stride=(2 * H, 2, 1, 1), data_type=cudnn.data_type.FLOAT, name="badsink")
    try:
        o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, sink_token=bad_sink)
    except TypeError:
        pytest.skip("this cuDNN wheel's sdpa() binding predates sink_token")
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_resolve_causal_plus_swa():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        use_causal_mask=True,
        sliding_window_length=128,
    )
    _finish_output(o, dims, strides)
    cfg = _facts(g)
    assert cfg.causal is True
    assert cfg.bottom_right is False
    # cuDNN's sliding_window_length is a length; Frost's swa_window is an offset
    # (length - 1), so 128 maps to 127.
    assert cfg.window_left == 127


def test_resolve_plain_swa_no_causal():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, sliding_window_length=64)
    _finish_output(o, dims, strides)
    cfg = _facts(g)
    assert cfg.causal is False
    assert cfg.window_left == 63


def test_resolve_causal_bottom_right():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask_bottom_right=True)
    _finish_output(o, dims, strides)
    cfg = _facts(g)
    assert cfg.causal is True
    assert cfg.bottom_right is True


def test_resolve_padding_mask_with_seq_len_kv():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_kv")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_padding_mask=True, seq_len_kv=seq_kv)
    _finish_output(o, dims, strides)
    cfg = _facts(g)
    assert cfg.padded is True
    assert engines.engine_name(512) in _eligible(g)


def test_resolve_generate_stats():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, stats = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, generate_stats=True)
    _finish_output(o, dims, strides)
    if stats is not None:
        stats.set_output(True).set_dim((B, H, S, 1)).set_stride((H * S, S, 1, 1))
        stats.set_data_type(cudnn.data_type.FLOAT)
    cfg = _facts(g)
    assert cfg.wants_stats is True
    assert cfg.stats_t is not None


def test_probe_rejects_bottom_right_plus_swa():
    # Kernel gap: CAUSAL_BOTTOM_RIGHT excludes SWA (config would assert at import).
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        use_causal_mask_bottom_right=True,
        sliding_window_length=128,
    )
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_bottom_right_swa_only():
    # Kernel gap: CAUSAL_BOTTOM_RIGHT requires MASK_CAUSAL; BOTTOM_RIGHT
    # alignment with only a left band has no causal bit.
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        diagonal_band_left_bound=64,
        diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT,
    )
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_ragged_skv_without_padding_or_causal():
    # KV tail (S_kv % 128 != 0) is only masked on the padded / causal paths;
    # a dense graph with a ragged S_kv would silently read the tail columns.
    g = _mk_graph()
    s_kv = 300
    q = g.tensor(dim=(B, H, S, D), stride=(S * H * D, D, H * D, 1), data_type=DTYPE, name="q")
    k = g.tensor(dim=(B, H, s_kv, D), stride=(s_kv * H * D, D, H * D, 1), data_type=DTYPE, name="k")
    v = g.tensor(dim=(B, H, s_kv, D), stride=(s_kv * H * D, D, H * D, 1), data_type=DTYPE, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, (B, H, S, D), (S * H * D, D, H * D, 1))
    assert not _eligible(g)


def test_probe_accepts_ragged_skv_with_top_left_causal():
    # Top-left causal with S_q <= S_kv provably masks the KV tail columns.
    g = _mk_graph()
    s_kv = 300
    q = g.tensor(dim=(B, H, S, D), stride=(S * H * D, D, H * D, 1), data_type=DTYPE, name="q")
    k = g.tensor(dim=(B, H, s_kv, D), stride=(s_kv * H * D, D, H * D, 1), data_type=DTYPE, name="k")
    v = g.tensor(dim=(B, H, s_kv, D), stride=(s_kv * H * D, D, H * D, 1), data_type=DTYPE, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True)
    _finish_output(o, (B, H, S, D), (S * H * D, D, H * D, 1))
    assert engines.engine_name(512) in _eligible(g)


def _mk_eligible_graph():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True)
    _finish_output(o, dims, strides)
    return g


def test_knob_request_within_domain_keeps_engine_eligible():
    g = _mk_eligible_graph()
    g.set_engine_knobs(engines.SdpaFwdKnobs(sched_policy=0, tile_n=128, cga=2))
    assert engines.engine_name(512) in _eligible(g)


def test_knob_request_outside_domain_rejects_engine():
    # No engine advertises LPT scheduling yet: honored or ineligible, never degraded.
    g = _mk_eligible_graph()
    g.set_engine_knobs(engines.SdpaFwdKnobs(sched_policy=1))
    assert not _eligible(g)


def test_knob_request_unsupported_tile_rejects_engine():
    g = _mk_eligible_graph()
    g.set_engine_knobs(engines.SdpaFwdKnobs(tile_n=64))
    assert not _eligible(g)


def test_knob_request_wrong_vocabulary_rejects_engine():
    # A different op's knob object must not silently pass.
    g = _mk_eligible_graph()
    g.set_engine_knobs(object())
    assert not _eligible(g)


def test_knob_request_none_fields_are_no_preference():
    g = _mk_eligible_graph()
    g.set_engine_knobs(engines.SdpaFwdKnobs())
    assert engines.engine_name(512) in _eligible(g)
