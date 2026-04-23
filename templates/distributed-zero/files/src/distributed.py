import os
import torch
import torch.distributed as dist

from .config import ZERO_CONFIG


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", ZERO_CONFIG["world_size"]))

    dist.init_process_group(
        backend=ZERO_CONFIG["backend"],
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_rank():
    return not dist.is_initialized() or dist.get_rank() == 0
