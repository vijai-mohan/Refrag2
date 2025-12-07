import os
import multiprocessing as mp

ROOT = os.path.dirname(__file__)

WORKERS = {}


def _worker_entry(req_q, res_q, model_name, cpu_aff):
    """Entry point used as the multiprocessing target.
    This function is defined at module top-level so it can be pickled by the 'spawn'
    start method. It imports worker.py by path and delegates to worker_main.
    """
    try:
        import importlib
        worker_mod = importlib.import_module('worker')
    except Exception:
        import importlib.util
        worker_file = os.path.join(ROOT, 'worker.py')
        spec = importlib.util.spec_from_file_location('worker', worker_file)
        worker_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker_mod)

    # Ensure the child process has a descriptive name (useful for debugging/logs)
    try:
        mp.current_process().name = model_name
    except Exception:
        pass

    # Delegate to worker_main
    worker_mod.worker_main(req_q, res_q, model_name, cpu_aff)


def start_workers(models_list):
    """Start worker processes for each model in models_list.
    Each worker gets a pair of queues (request_q, response_q) and is started via _worker_entry.
    """
    # Read affinities from env
    affinity_map = {}
    try:
        raw = os.environ.get('WORKER_CPU_AFFINITY')
        if raw:
            import json as _json
            affinity_map = _json.loads(raw)
    except Exception:
        affinity_map = {}

    for m in models_list:
        if m in WORKERS:
            continue
        req_q = mp.Queue()
        res_q = mp.Queue()
        cpu_aff = affinity_map.get(m)
        # Name the process with the model identifier to make it easy to identify
        p = mp.Process(target=_worker_entry, args=(req_q, res_q, m, cpu_aff), name=str(m), daemon=True)
        p.start()
        WORKERS[m] = {"proc": p, "req_q": req_q, "res_q": res_q}


def stop_workers():
    for m, info in list(WORKERS.items()):
        try:
            info['req_q'].put(None)
            info['proc'].join(timeout=2.0)
        except Exception:
            pass
    WORKERS.clear()
