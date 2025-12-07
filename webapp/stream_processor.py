import time
from typing import Callable, List, Optional, Dict, Any
from tokenizer_adapter import TokenizerAdapter

class StreamProcessor:
    """Simple pass-through stream processor.

    Emits each chunk yielded by the TextIteratorStreamer unchanged as a 'token' event.
    Supports cancellation via cancel_dict but performs no intro-stripping or stop-marker
    manipulation so output text is preserved exactly as produced by the tokenizer.
    """

    def __init__(
        self,
        tokenizer_adapter: TokenizerAdapter,
        stop_markers: Optional[List[str]] = None,
        cancel_dict: Optional[Dict[str, Any]] = None,
        intro_strip_regex: Optional[str] = None,
    ):
        self.tk = tokenizer_adapter
        self.stop_markers = stop_markers or []
        self.cancel_dict = cancel_dict
        # We intentionally ignore intro_strip_regex to avoid modifying output text

    def _maybe_cancel(self, req_id):
        try:
            if self.cancel_dict is not None and req_id is not None:
                return bool(self.cancel_dict.get(req_id))
        except Exception:
            return False
        return False

    def process(self, streamer, emit_event: Callable[[dict], None], req_id: str | None):
        """Iterate the streamer and emit each non-empty chunk unchanged.

        Emits events of the form:
          {"id": req_id, "event_type": "token", "payload": {"token": chunk, "token_ts": ts, "n_tokens": n}}

        If cancellation is detected via cancel_dict, emits a 'cancelled' event and returns.
        """
        for chunk in streamer:
            # Cancellation check
            if self._maybe_cancel(req_id):
                try:
                    emit_event({"id": req_id, "event_type": "cancelled", "payload": {"is_last": True, "token_ts": time.time()}})
                except Exception:
                    pass
                return

            if not chunk:
                continue

            ts = time.time()
            # naive token count by whitespace; keep simple and fast
            n = len((chunk or '').split())
            try:
                emit_event({"id": req_id, "event_type": "token", "payload": {"token": chunk, "token_ts": ts, "n_tokens": n}})
            except Exception:
                # swallow emit errors to avoid crashing the streamer loop
                pass
        # do not emit a final 'done' here; the caller (worker) handles end-of-stream
