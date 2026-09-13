# -*- coding: utf-8 -*-
"""Query-side LoRA adapter: applied to questions only, never to documents.

The adapter maps a question into the *existing* document space, so the frozen
production collection stays valid and no re-indexing is ever required.  That
property only holds if documents are never encoded through it, which is why
this module exposes a query-shaped entry point and is called from exactly one
place -- the query embedding step in ``rag_gate.retrieve_with_trace``.  Ingest
does not import it and structurally cannot reach it.

Default off: ``RAG_QUERY_ADAPTER_DIR`` empty means the production path keeps
calling ``get_embeddings_batch`` unchanged.
"""

from __future__ import annotations

import os
import threading
from typing import Any


_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()


def adapter_path() -> str:
    return (os.environ.get("RAG_QUERY_ADAPTER_DIR", "") or "").strip()


def _load(path: str):
    """Rebuild the tower and graft the trained low-rank residuals onto it."""
    import torch
    from train_query_adapter import build_tower

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "lora" not in payload:
        raise RuntimeError(
            f"{path} is not a query-adapter checkpoint (expected a dict with "
            "'lora'); refusing to guess its geometry"
        )
    device = os.environ.get("OFFERCLAW_TORCH_DEVICE", "").strip() or "cpu"
    tokenizer, model, _wrapped, _trainable = build_tower(
        payload.get("base_model", "BAAI/bge-base-zh-v1.5"),
        int(payload["rank"]), float(payload["alpha"]), device,
    )
    missing = model.load_state_dict(payload["lora"], strict=False)
    # ``strict=False`` is required (the checkpoint holds only lora_ tensors),
    # so unexpected keys are the failure mode that would otherwise load a
    # silently untrained adapter and report it as a result.
    if getattr(missing, "unexpected_keys", None):
        raise RuntimeError(
            f"{path}: checkpoint carries keys this tower has no slot for: "
            f"{list(missing.unexpected_keys)[:3]}"
        )
    model.eval()
    return tokenizer, model, device


def embed_query(text: str) -> list[float] | None:
    """Return the adapted query vector, or ``None`` when the adapter is off."""
    path = adapter_path()
    if not path:
        return None
    with _LOCK:
        entry = _CACHE.get(path)
        if entry is None:
            entry = _load(path)
            _CACHE[path] = entry
        tokenizer, model, device = entry
        from train_query_adapter import encode

        vector = encode(model, tokenizer, [text], device, batch_size=1)
    return vector[0].detach().cpu().tolist()


__all__ = ["adapter_path", "embed_query"]
