# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapter over the Frost SM100 (Blackwell) DSL SDPA prefill kernels (d=512 / d=256)."""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TupleDict
from cudnn.frost.template_loader import load_template
from cudnn.sdpa.fwd.config_sm100 import (
    MASK_CAUSAL as _MASK_CAUSAL,
    MASK_NONE as _MASK_NONE,
    MASK_PADDED as _MASK_PADDED,
    MASK_SWA as _MASK_SWA,
    SCHED_NATURAL as _SCHED_NATURAL,
    TemplateParams,
)

_FLAVOR_DIMS = {512: (512, 512), 256: (256, 256)}  # template key -> (D_QK, D_V)
_KERNEL_FILES = {
    512: "prefill_d512_f16_sm100.py",
    256: "prefill_d256_f16_sm100.py",
}
_DTYPE_QKV_CODE = {torch.bfloat16: 2, torch.float16: 3}  # config.DTYPE_BF16 / DTYPE_FP16
# Both flavors tile KV in TILE_N=128 columns; the KV tail is only masked when
# the padded/causal mask paths are active (see check_support).
_TILE_N = 128


def _pick_flavor(d_qk: int, d_v: int) -> int:
    """Exact-match flavor for ``(d_qk, d_v)`` (no envelope padding)."""
    for flavor, (fdqk, fdv) in _FLAVOR_DIMS.items():
        if d_qk == fdqk and d_v == fdv:
            return flavor
    raise ValueError(f"Frost SM100 DSL SDPA: no exact-match flavor for (D_QK={d_qk}, " f"D_V={d_v}); supported: {_FLAVOR_DIMS} (no envelope padding on SM100).")


def _load_kernel_module(flavor: int, params: TemplateParams):
    """One uniquely-named module per (flavor, params) via the shared FROST loader."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", _KERNEL_FILES[flavor])
    return load_template(path, params, tag=f"sdpa_fwd_d{flavor}")


class SdpaFwdDsl(APIBase):
    """SM100 (Blackwell) SDPA forward via the Frost DSL kernels."""

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_lse: torch.Tensor,
        is_causal: bool = False,
        causal_bottom_right: bool = False,
        window_size_left: Optional[int] = None,
        scale_softmax: Optional[float] = None,
        seq_kv_lens_present: bool = False,
        has_sink: bool = False,
        thd: bool = False,
        sched_policy: int = _SCHED_NATURAL,
    ):
        super().__init__()
        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="lse")

        self.is_causal = bool(is_causal)
        self.causal_bottom_right = bool(causal_bottom_right)
        # window_size_left is an offset W ("keep k in [q-W, q]"); callers pass W = L - 1 for a cuDNN window length L.
        self.window_size_left = window_size_left
        self.scale_softmax = scale_softmax
        self.seq_kv_lens_present = bool(seq_kv_lens_present)
        self.has_sink = bool(has_sink)
        self.thd = bool(thd)
        # Tuning-knob choice, already validated against the engine's
        # Capabilities domain by the probe (engines.mismatch).
        self.sched_policy = int(sched_policy)

        self.batch_size: Optional[int] = None
        self.s_q_max: Optional[int] = None
        self.s_k_max: Optional[int] = None
        self.h_q: Optional[int] = None
        self.h_kv: Optional[int] = None
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.flavor: Optional[int] = None  # kernel-template key (head dim)
        self.mask_flags: int = 0
        self.swa_window_runtime: int = 0
        self.dtype: Optional[torch.dtype] = None
        self._k_mod = None

        self._dummy_cache: dict = {}

        self._logger.debug("__init__ completed")

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        # Required stride order (3, 1, 2, 0): BHSD logical / BSHD physical, size-1 dims wildcarded.
        _REQ = (3, 1, 2, 0)
        for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc"]:
            d = getattr(self, desc_name)
            self._value_error_if(
                d.ndim != 4,
                f"{d.name} must be rank-4 (B, H, S, D); got {d.ndim}",
            )
            _shape = d.shape
            _act = tuple(ax for ax in d.stride_order if _shape[ax] != 1)
            _exp = tuple(ax for ax in _REQ if _shape[ax] != 1)
            self._value_error_if(
                _act != _exp,
                f"{d.name} must have d, h, s, b stride order (3, 1, 2, 0) " f"(size-1 dims wildcarded); got {d.stride_order} shape {_shape}",
            )

        b, h_qo, s_qo, d_qk = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        _, _, _, d_v = self.v_desc.shape

        self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_qo, s_qo, d_v), name="O")

        for label, val in (
            ("B", b),
            ("H_q", h_qo),
            ("H_kv", h_kv),
            ("S_q", s_qo),
            ("S_kv", s_kv),
            ("D_QK", d_qk),
            ("D_V", d_v),
        ):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")

        self._value_error_if(
            h_qo % h_kv != 0,
            f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )

        # O dtype must equal Q/K/V dtype: the kernel pins DTYPE_O to DTYPE_QKV.
        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for desc in [self.k_desc, self.v_desc, self.o_desc]:
            self._check_dtype(
                desc,
                self.dtype,
                name=desc.name,
                extra_error_msg=f"{desc.name} must match Q dtype (FP16/BF16 on SM100 DSL)",
            )
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
        self._value_error_if(not self.lse_desc.is_contiguous(), "LSE must be contiguous on SM100 DSL")

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM100 DSL SDPA")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        self._value_error_if(
            (major, minor) != (10, 0),
            f"SdpaFwdDsl requires SM100 (Blackwell, cc=10.0); found SM{major}{minor} on {device}",
        )

        self.flavor = _pick_flavor(d_qk, d_v)

        swa_left = self.window_size_left
        self._value_error_if(
            swa_left is not None and swa_left < 0,
            f"window_size_left must be >= 0; got {swa_left}",
        )
        # The kernels' bottom-right diagonal path supports plain causal only:
        # CAUSAL_BOTTOM_RIGHT requires MASK_CAUSAL and excludes MASK_SWA
        # (see config._validate_knobs).
        self._value_error_if(
            self.causal_bottom_right and not self.is_causal,
            "SM100 DSL SDPA: causal_bottom_right requires is_causal=True",
        )
        self._value_error_if(
            self.causal_bottom_right and swa_left is not None,
            "SM100 DSL SDPA: causal_bottom_right cannot be combined with a " "left sliding-window (kernel gap)",
        )
        if self.is_causal:
            self.mask_flags = _MASK_CAUSAL | (_MASK_SWA if swa_left is not None else 0)
            self.swa_window_runtime = swa_left if swa_left is not None else 0
        elif swa_left is not None:
            self.mask_flags = _MASK_SWA
            self.swa_window_runtime = swa_left
        else:
            self.mask_flags = _MASK_NONE
            self.swa_window_runtime = 0
        if self.thd:
            self.seq_kv_lens_present = True
        if self.seq_kv_lens_present:
            self.mask_flags |= _MASK_PADDED
        # KV-tail correctness: the kernel zero-fills the last KV tile via TMA
        # OOB but only *masks* those columns on the padded / causal paths. A
        # ragged S_kv is safe when a padding mask carries the real lengths, or
        # when the causal diagonal provably covers the tail (kv >= S_kv implies
        # kv > q for every query row). Otherwise the tail columns leak into
        # the softmax and the output is silently wrong.
        if int(s_kv) % _TILE_N != 0:
            causal_covers_tail = self.is_causal and (self.causal_bottom_right or int(s_qo) <= int(s_kv))
            self._value_error_if(
                not (self.seq_kv_lens_present or causal_covers_tail),
                f"S_kv ({s_kv}) must be a multiple of {_TILE_N} unless a "
                f"padding mask (seq_len_kv) is provided or the causal mask "
                f"covers the KV tail — the tail is otherwise unmasked on "
                f"SM100 DSL",
            )

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self.batch_size = int(b)
        self.s_q_max = int(s_qo)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_qo)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        params = TemplateParams(
            dtype_qkv=_DTYPE_QKV_CODE[self.dtype],
            mask_flags=self.mask_flags,
            swa_window=int(self.swa_window_runtime),
            causal_bottom_right=self.causal_bottom_right,
            has_sink=self.has_sink,
            seq_kv_lens_present=self.seq_kv_lens_present,
            sched_policy=self.sched_policy,
            thd_varlen=self.thd,
        )
        self._k_mod = _load_kernel_module(self.flavor, params)
        if self.thd:
            # T (total tokens) is a runtime value, so the per-shape compile is deferred to execute().
            self._compiled_kernel = "thd-deferred"
        else:
            self._compiled_kernel = self._k_mod.compile(
                b=self.batch_size,
                qh=self.h_q,
                kh=self.h_kv,
                sq=self.s_q_max,
                skv=self.s_k_max,
            )
        self._logger.debug("compile completed")

    @staticmethod
    def _to_bshd(t: torch.Tensor) -> torch.Tensor:
        """BHSD-logical -> BSHD-physical view (zero-copy in the common case)."""
        view = t.transpose(1, 2)
        return view if view.is_contiguous() else view.contiguous()

    @staticmethod
    def _to_bshd_writable(t: torch.Tensor):
        """BSHD-physical view of an output tensor; non-contiguous views route through a scratch buffer copied back after launch."""
        view = t.transpose(1, 2)
        if view.is_contiguous():
            return view, False, None
        scratch = torch.empty_like(view, memory_format=torch.contiguous_format)
        return view, True, scratch

    def _dummy(self, key: str, device: torch.device, factory) -> torch.Tensor:
        cache_key = (key, device)
        t = self._dummy_cache.get(cache_key)
        if t is None:
            t = factory()
            self._dummy_cache[cache_key] = t
        return t

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: torch.Tensor,
        scale_softmax: Optional[float] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        seq_len_q: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpaFwdDsl is not compiled")
        if current_stream is not None:
            raise NotImplementedError(
                "SdpaFwdDsl.execute: explicit current_stream is not "
                "yet supported. Wrap the call in `with torch.cuda.stream(s):` to "
                "dispatch onto a non-default stream."
            )

        scale_val = self.scale_softmax if (scale_softmax is None or scale_softmax == 0.0) else float(scale_softmax)
        scale_softmax_log2 = scale_val * math.log2(math.e)

        if self.thd:
            self._execute_thd(q_tensor, k_tensor, v_tensor, o_tensor, scale_softmax_log2, sinks, seq_kv_lens, seq_len_q)
            return

        Q = self._to_bshd(q_tensor)
        K = self._to_bshd(k_tensor)
        V = self._to_bshd(v_tensor)
        O_view, o_needs_copy_back, O_scratch = self._to_bshd_writable(o_tensor)

        device = q_tensor.device
        sinks_t = (
            sinks.reshape(-1).to(torch.float32)
            if sinks is not None
            else self._dummy("sinks", device, lambda: torch.zeros(self.h_q, dtype=torch.float32, device=device))
        )
        seq_kv_t = (
            seq_kv_lens.reshape(-1).to(torch.int32)
            if seq_kv_lens is not None
            else self._dummy("seq_kv", device, lambda: torch.zeros(self.batch_size, dtype=torch.int32, device=device))
        )
        o_desc_dummy = self._dummy("o_desc", device, lambda: torch.zeros(1, dtype=torch.int64, device=device))

        import cutlass

        self._compiled_kernel(
            Q,
            K,
            V,
            O_scratch if o_needs_copy_back else O_view,
            lse_tensor.reshape(self.batch_size, self.h_q, self.s_q_max),
            sinks_t,
            seq_kv_t,
            o_desc_dummy,
            (self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, 0),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Int32(0),
        )
        if o_needs_copy_back:
            O_view.copy_(O_scratch)
        self._logger.debug("execute completed")

    def _execute_thd(self, q_buf, k_buf, v_buf, o_buf, scale_softmax_log2, sinks, seq_len_kv, seq_len_q):
        """THD / varlen execute: reconstruct the kernel's packed [1, T, H, D] views and metadata buffer from the cuDNN ragged buffers, then launch."""
        import cutlass

        dev = q_buf.device
        if seq_len_q is None or seq_len_kv is None:
            raise ValueError("THD execute requires seq_len_q and seq_len_kv")
        slq = seq_len_q.reshape(-1).to(torch.int32)
        slk = seq_len_kv.reshape(-1).to(torch.int32)
        b = slq.numel()
        z = torch.zeros(1, dtype=torch.int32, device=dev)
        cu_q = torch.cat([z, slq.cumsum(0).to(torch.int32)])
        cu_k = torch.cat([z, slk.cumsum(0).to(torch.int32)])
        t_q = int(cu_q[-1].item())
        t_kv = int(cu_k[-1].item())

        qh, kh = self.h_q, self.h_kv
        d_qk, d_v = self.head_dim_qk, self.head_dim_v

        # Metadata buffer: [ seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1) ].
        meta = torch.empty(3 * b + 2, dtype=torch.int32, device=dev)
        meta[0:b] = slk
        meta[b : 2 * b + 1] = cu_q
        meta[2 * b + 1 :] = cu_k
        o_desc = torch.zeros(b * 16 + 16, dtype=torch.int64, device=dev)
        # One THD unit per CGA-height slice of each sequence's Q rows.
        cga_tile_m = int(self._k_mod.CGA_TILE_M)
        units = int((((slq + cga_tile_m - 1) // cga_tile_m).sum()).item()) * qh

        def _packed(buf, t, h, d):
            return buf.as_strided((1, t, h, d), (t * h * d, h * d, d, 1), buf.storage_offset())

        Q = _packed(q_buf, t_q, qh, d_qk)
        K = _packed(k_buf, t_kv, kh, d_qk)
        V = _packed(v_buf, t_kv, kh, d_v)
        O = _packed(o_buf, t_q, qh, d_v)
        LSE = torch.zeros(1, qh, t_q, dtype=torch.float32, device=dev)
        sinks_t = sinks.reshape(-1).to(torch.float32) if sinks is not None else torch.zeros(qh, dtype=torch.float32, device=dev)

        fn = self._k_mod.compile(b=b, qh=qh, kh=kh, sq=t_q, skv=t_kv)
        fn(Q, K, V, O, LSE, sinks_t, meta, o_desc, (b, qh, kh, t_q, t_kv, 0), cutlass.Float32(scale_softmax_log2), cutlass.Int32(units))
        self._logger.debug("execute (THD) completed")


_logger = logging.getLogger(__name__)
_cache_of_objects: dict = {}


def _allocate_lse_tensor(q_tensor: torch.Tensor) -> torch.Tensor:
    if q_tensor.ndim != 4:
        raise ValueError(f"Expected BHSD q_tensor to be rank-4, got {q_tensor.ndim}")
    b, h, s_q, _ = q_tensor.shape
    return torch.empty((b, h, s_q), dtype=torch.float32, device=q_tensor.device)


def sdpa_fwd_wrapper_dsl(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size_left: Optional[int] = None,
    causal_bottom_right: bool = False,
    scale_softmax: Optional[float] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """SM100 SDPA forward; returns ``TupleDict(o_tensor=..., lse_tensor=...)``."""
    if current_stream is not None:
        raise NotImplementedError(
            "sdpa_fwd_wrapper_dsl: explicit current_stream is not "
            "yet supported. Wrap the call in `with torch.cuda.stream(s):` to "
            "dispatch onto a non-default stream."
        )
    if q_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q and V must be rank-4 BHSD; got Q={q_tensor.ndim}D V={v_tensor.ndim}D")

    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor(q_tensor)

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        q_tensor.dtype,
        k_tensor.dtype,
        v_tensor.dtype,
        is_causal,
        window_size_left,
        causal_bottom_right,
        scale_softmax,
        seq_kv_lens is not None,
        sinks is not None,
        q_tensor.device,
        k_tensor.device,
        v_tensor.device,
    )
    sdpa_fwd = _cache_of_objects.get(cache_key)
    if sdpa_fwd is None:
        _logger.debug("sdpa_fwd_wrapper_dsl: building new SdpaFwdDsl")
        sdpa_fwd = SdpaFwdDsl(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_lse=lse_tensor,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_size_left=window_size_left,
            scale_softmax=scale_softmax,
            seq_kv_lens_present=seq_kv_lens is not None,
            has_sink=sinks is not None,
        )
        sdpa_fwd.check_support()  # raises ValueError / NotImplementedError if unsupported
        sdpa_fwd.compile()
        _cache_of_objects[cache_key] = sdpa_fwd

    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        scale_softmax=scale_softmax,
        sinks=sinks,
        seq_kv_lens=seq_kv_lens,
        current_stream=current_stream,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)
