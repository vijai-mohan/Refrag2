"""
Distributed Hello World App

This simple app shows:
- How to write a distributed PyTorch app
- How to get rank and world size
- How to run with the torchrun env via Hydra

Usage:
  python run.py env=torchrun app=hello_dist
  python run.py env=torchrun env.nproc_per_node=4 app=hello_dist
"""

from dataclasses import dataclass

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import os


@dataclass
class HelloDistApp:
    """A simple distributed hello world application."""

    message: str = "Hello from distributed process"



    def run(self, cfg: DictConfig):
        """Run in distributed mode - called by torchrun."""
        print(f"[run] starting run(), pid={os.getpid()}, env.RANK={os.environ.get('RANK')}, env.LOCAL_RANK={os.environ.get('LOCAL_RANK')}, env.WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
        import torch
        import torch.distributed as dist

        # Check if we're running in distributed mode
        if not dist.is_available():
            print("PyTorch distributed is not available!")
            return
        # Initialize process group if not already initialized
        if not dist.is_initialized():
            print("[run] initializing process group (dist.init_process_group)")
            # This is called by torchrun, which sets these env vars
            master = os.environ.get("MASTER_ADDR", "127.0.0.1")
            port = os.environ.get("MASTER_PORT", "29500")
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

            dist.init_process_group(backend="gloo", init_method=f"tcp://{master}:{port}", world_size=world_size,
                                    rank=rank)

            print("[run] process group initialized")

        # Get distributed info
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Print from each rank
        print(f"[Rank {rank}/{world_size}] {self.message}")
        print(f"[Rank {rank}/{world_size}] Local rank: {local_rank}")
        print(f"[Rank {rank}/{world_size}] Process ID: {os.getpid()}")

        # Demonstrate a simple collective operation
        # All processes send their rank, we sum them up
        print(f"[Rank {rank}] preparing rank tensor for all_reduce")
        rank_tensor = torch.tensor([rank], dtype=torch.int32)
        dist.all_reduce(rank_tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] all_reduce completed, value={rank_tensor.item()}")

        if rank == 0:
            print(f"\n[Rank 0] Sum of all ranks: {rank_tensor.item()}")
            print(f"[Rank 0] Expected sum: {sum(range(world_size))}")
            print(f"[Rank 0] ✓ Collective operation successful!")

        # Barrier to synchronize all processes
        print(f"[Rank {rank}] entering barrier()")
        dist.barrier()
        print(f"[Rank {rank}] passed barrier()")

        if rank == 0:
            print(f"\n[Rank 0] All {world_size} processes completed successfully!")

        # Cleanup
        print(f"[Rank {rank}] destroying process group")
        dist.destroy_process_group()
        print(f"[Rank {rank}] process group destroyed")

        return {"status": "success", "rank": rank, "world_size": world_size}


if __name__ == "__main__":
    """Entry point when run directly via torchrun."""
    import argparse
    import torch.distributed as dist

    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="Hello from distributed process")
    args = parser.parse_args()

    print(f"[__main__] parsed args: {args}")
    # Create app and run
    app = HelloDistApp(message=args.message)

    print("[__main__] calling app.run()")
    # When called by torchrun, just run directly (not via Hydra)
    from omegaconf import DictConfig
    app.run(DictConfig({}))
