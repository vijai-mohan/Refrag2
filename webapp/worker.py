import multiprocessing as mp
import time
import psutil

from model_config import get_model_config
from prompt_formatter import format_prompt
from tokenizer_adapter import TokenizerAdapter
from stream_processor import StreamProcessor


def _set_cpu_affinity(affinity):
    try:
        p = psutil.Process()
        p.cpu_affinity(affinity)
        return True
    except Exception:
        return False


def worker_main(request_queue: mp.Queue, response_queue: mp.Queue, model_name: str, cpu_affinity=None, cancel_dict=None):
    """Model-agnostic worker using declarative config and streaming."""
    # Name the process for easier identification in process listings
    try:
        mp.current_process().name = model_name
    except Exception:
        pass

    # Also name the main thread inside the process
    try:
        import threading as _threading
        _threading.current_thread().name = f"worker-main-{model_name}"
    except Exception:
        pass

    if cpu_affinity:
        _set_cpu_affinity(cpu_affinity)

    # Lazy imports
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
        import torch
        import threading
    except Exception as e:
        response_queue.put({"id": None, "event_type": "worker_error", "payload": f"missing deps: {e}"})
        return

    cfg = get_model_config(model_name) or {}

    # Device from config (default cpu)
    try:
        device_str = str(cfg.get('device', 'cpu'))
        device = torch.device(device_str)
    except Exception:
        device = torch.device('cpu')

    # Load model + tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if getattr(model.config, 'pad_token_id', None) is None and getattr(model.config, 'eos_token_id', None) is not None:
            model.config.pad_token_id = model.config.eos_token_id
        model.to(device).eval()
    except Exception as e:
        response_queue.put({"id": None, "event_type": "worker_error", "payload": f"load error: {e}"})
        return

    # Stop markers: union of config + tokenizer eos
    stop_markers = list(cfg.get('stop_markers') or [])
    try:
        if getattr(tokenizer, 'eos_token', None):
            stop_markers.append(tokenizer.eos_token)
    except Exception:
        pass

    tk = TokenizerAdapter(tokenizer)

    response_queue.put({"id": None, "event_type": "model_ready", "payload": {"model_name": model_name, "timestamp": time.time()}})

    # Generation defaults from config
    gen_defaults = dict(cfg.get('gen_defaults') or {})

    # Tokenizer decode defaults from config
    decode_kwargs = dict(((cfg.get('tokenizer') or {}).get('decode_kwargs') or {}))
    streamer_decode_kwargs = {"skip_special_tokens": True, **decode_kwargs}

    # Optional tokenizer chat-template flag
    use_template = bool(cfg.get('use_tokenizer_template', True))
    add_generation_prompt = bool(cfg.get('add_generation_prompt', False))

    # We intentionally do not perform any intro-strip or text mutation here; the StreamProcessor is a pass-through.

    while True:
        req = request_queue.get()
        if req is None:
            break
        req_id = req.get('id')
        prompt = req.get('prompt', '')
        history = req.get('history') if isinstance(req, dict) else None

        # Merge generation params: request overrides > config defaults
        max_new_tokens = int(req.get('max_new_tokens', gen_defaults.get('max_new_tokens', 128)))
        temperature = float(req.get('temperature', gen_defaults.get('temperature', 1.0)))
        top_p = float(req.get('top_p', gen_defaults.get('top_p', 1.0)))
        do_sample = bool(req.get('do_sample', gen_defaults.get('do_sample', True)))

        t0 = time.time()
        response_queue.put({"id": req_id, "event_type": "request_received", "payload": {"request_received": t0}})

        # Prompt building: either tokenizer chat template or declarative formatter
        prompt_text = prompt
        try:
            hist = history or []
            sys_prompt = cfg.get('system_prompt')
            if sys_prompt and not any((isinstance(h, dict) and (h.get('role') or '').lower().startswith('system')) for h in hist):
                hist = [{'role': 'system', 'text': sys_prompt}] + hist
            # Irrespective of having a history, treat the current prompt with the role when using tokenizer template
            if use_template and hasattr(tk.tokenizer, 'apply_chat_template'):
                chat = []
                for h in hist:
                    role = (h.get('role') or 'user')
                    content = h.get('text') or h.get('content') or ''
                    chat.append({"role": role, "content": content})
                chat.append({"role": "user", "content": prompt})
                prompt_text = tk.apply_chat_template(chat, add_generation_prompt=add_generation_prompt)
            else:
                prompt_text = format_prompt(model_name, hist, prompt) if hist else prompt
        except Exception:
            prompt_text = prompt

        # Tokenize
        try:
            inputs = tokenizer(prompt_text, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            response_queue.put({"id": req_id, "event_type": "worker_error", "payload": "tokenize_failed"})
            continue

        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p)
        if getattr(model.config, 'pad_token_id', None) is not None:
            gen_kwargs['pad_token_id'] = int(model.config.pad_token_id)
        try:
            cfg_eos = cfg.get('eos_token_id')
            if cfg_eos is not None:
                gen_kwargs['eos_token_id'] = int(cfg_eos)
        except Exception:
            pass

        try:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs=streamer_decode_kwargs, skip_special_tokens=True)
            th = threading.Thread(target=model.generate, kwargs={**inputs, **gen_kwargs, 'streamer': streamer}, daemon=True)
            # Name the generation thread with the model identifier for easier tracing
            try:
                th.name = f"gen-{model_name}"
            except Exception:
                pass
            th.start()

            # Use StreamProcessor (pass-through) to emit chunks unchanged.
            sp = StreamProcessor(tk, stop_markers, cancel_dict=cancel_dict, intro_strip_regex=None)
            sp.process(streamer, lambda ev: response_queue.put({**ev, 'id': req_id}), req_id)

            th.join()
            response_queue.put({"id": req_id, "event_type": "done", "payload": {"is_last": True, "token_ts": time.time()}})
        except Exception as e:
            # fallback: blocking generate
            try:
                outputs = model.generate(**{**inputs, **gen_kwargs})
                out_ids = outputs[0].tolist()
                in_len = inputs['input_ids'].shape[1]
                gen_ids = out_ids[in_len:]
                text = tk.decode(gen_ids, **streamer_decode_kwargs)
                response_queue.put({"id": req_id, "event_type": "token", "payload": {"token": text, "token_ts": time.time(), "n_tokens": len(gen_ids)}})
                response_queue.put({"id": req_id, "event_type": "done", "payload": {"is_last": True, "token_ts": time.time()}})
            except Exception as e2:
                response_queue.put({"id": req_id, "event_type": "worker_error", "payload": str(e2)})

    response_queue.put({"id": None, "event_type": "worker_exited", "payload": model_name})
