# Torchrun Driver & Distributed Training Guide

## Overview

The torchrun driver enables distributed PyTorch training with all parameters configured via Hydra. This makes it easy to scale from single-process development to multi-GPU and multi-node training.

**Windows Note**: This driver automatically sets `USE_LIBUV=0` to avoid compatibility issues on Windows.

## Quick Start

### Run the Hello World Demo

Test with 2 processes on a single machine:
```bash
python run.py driver=torchrun env=local app=hello_dist
```

Test with 4 processes:
```bash
python run.py driver=torchrun env=local driver.nproc_per_node=4 app=hello_dist
```

Custom message:
```bash
python run.py driver=torchrun env=local app=hello_dist app.message="Greetings from rank"
```

## Configuration

### Driver Configuration (`conf/driver/torchrun.yaml`)

```yaml
_target_: refrag.framework.drivers.torchrun.TorchrunDriver

# Basic distributed settings
nproc_per_node: 2  # Number of GPUs/processes per node
nnodes: 1  # Number of nodes
node_rank: 0  # Rank of this node (0 for single node)

# Master node settings (only used if standalone=false)
master_addr: localhost
master_port: 29500

# Rendezvous settings (only used if standalone=false)
rdzv_backend: c10d
rdzv_endpoint: null
rdzv_id: null

# Restart and monitoring
max_restarts: 0
monitor_interval: 5.0

# Launch settings
start_method: spawn
standalone: true  # TRUE for single-node (simpler, recommended)
no_python: false
```

### Key Parameters

**Single Node (Multi-GPU) - RECOMMENDED FOR MOST CASES**
- `standalone: true` (default) - Simplest setup, no rendezvous needed
- `nproc_per_node`: Number of GPUs/processes (e.g., 2, 4, 8)
- No need to set master_addr, master_port, or rdzv settings

**Multi-Node Training** (Advanced)
- `standalone: false` - Enable multi-node coordination
- `nproc_per_node`: GPUs per node
- `nnodes`: Total number of nodes
- `node_rank`: 0 for master, 1, 2, ... for workers
- `master_addr`: IP or hostname of rank 0 node
- `master_port`: Port for communication (same on all nodes)
- `rdzv_backend`: c10d (recommended)

## How to Write a Distributed App

Your app needs two methods:

### 1. `build_command(cfg)` - Returns command to run

```python
def build_command(self, cfg: DictConfig):
    """Return command for distributed execution."""
    import sys
    return [sys.executable, __file__, "--arg1=value"]
```

### 2. `run(cfg)` or `__main__` - Actual distributed logic

```python
def run(self, cfg: DictConfig):
    import torch.distributed as dist
    
    # Initialize process group (torchrun sets env vars)
    dist.init_process_group(backend="nccl")  # or "gloo" for CPU
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Your distributed code here
    print(f"Hello from rank {rank}/{world_size}")
    
    # Cleanup
    dist.destroy_process_group()
```

## Example Output

```bash
$ python run.py driver=torchrun env=local driver.nproc_per_node=2 app=hello_dist

[torchrun] Launching distributed job:
  Command: torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 29500 --rdzv_backend c10d --monitor_interval 5.0 --start_method spawn python src/refrag/framework/apps/hello_dist.py --message='Hello from distributed PyTorch!'
  Processes per node: 2
  Number of nodes: 1
  Master: localhost:29500

[Rank 0/2] Hello from distributed PyTorch!
[Rank 0/2] Local rank: 0
[Rank 0/2] Process ID: 12345
[Rank 1/2] Hello from distributed PyTorch!
[Rank 1/2] Local rank: 1
[Rank 1/2] Process ID: 12346

[Rank 0] Sum of all ranks: 1
[Rank 0] Expected sum: 1
[Rank 0] ✓ Collective operation successful!

[Rank 0] All 2 processes completed successfully!
```

## Common Scenarios

### Development (Single Process)
```bash
python run.py env=local app=hello_dist
```

### Single Node, 4 GPUs
```bash
python run.py driver=torchrun env=local driver.nproc_per_node=4 app=hello_dist
```

### Two Nodes, 8 GPUs Each

**On master node (rank 0):**
```bash
python run.py driver=torchrun env=local \
  driver.nproc_per_node=8 \
  driver.nnodes=2 \
  driver.node_rank=0 \
  driver.master_addr=192.168.1.10 \
  driver.master_port=29500 \
  app=hello_dist
```

**On worker node (rank 1):**
```bash
python run.py driver=torchrun env=local \
  driver.nproc_per_node=8 \
  driver.nnodes=2 \
  driver.node_rank=1 \
  driver.master_addr=192.168.1.10 \
  driver.master_port=29500 \
  app=hello_dist
```

### Elastic Training (Dynamic Nodes)
```bash
python run.py driver=torchrun env=local \
  driver.rdzv_backend=c10d \
  driver.rdzv_endpoint=192.168.1.10:29500 \
  driver.rdzv_id=my_job_123 \
  driver.min_nodes=2 \
  driver.max_nodes=4 \
  app=hello_dist
```

## Backends

- **NCCL**: Best for NVIDIA GPUs (`backend="nccl"`)
- **Gloo**: CPU or mixed CPU/GPU (`backend="gloo"`)
- **MPI**: If you have MPI installed (`backend="mpi"`)

## Troubleshooting

**Problem**: "Address already in use"
- **Solution**: Change `driver.master_port=29501` (or another free port)

**Problem**: Processes hang at initialization
- **Solution**: Check firewall settings, ensure all nodes can reach `master_addr:master_port`

**Problem**: NCCL errors on CPU
- **Solution**: Use `backend="gloo"` for CPU training

**Problem**: Different number of processes on nodes
- **Solution**: Ensure `nproc_per_node` is same on all nodes

## Architecture

```
run.py
  ├─ Instantiate app (hello_dist)
  ├─ Call app.build_command(cfg) → returns [python, script.py, --args]
  ├─ Instantiate driver (torchrun)
  └─ Call driver.run(cfg)
       └─ Builds: torchrun --nproc_per_node=N ... python script.py --args
            └─ Spawns N processes, each runs script.py
                 └─ Each process: dist.init_process_group()
                      └─ Your distributed code runs
```

## Next Steps

- Adapt your training script to use `build_command()` pattern
- Test locally with `nproc_per_node=2`
- Scale to multiple GPUs
- Deploy to multi-node cluster via AWS Batch or similar
