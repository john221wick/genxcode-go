import os
import torch
import torch.distributed as dist

from .config import SP_CONFIG


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", SP_CONFIG["tp_size"]))

    dist.init_process_group(
        backend=SP_CONFIG["backend"],
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def create_tp_group(world_size, tp_size):
    tp_groups = []
    for i in range(0, world_size, tp_size):
        ranks = list(range(i, i + tp_size))
        group = dist.new_group(ranks)
        tp_groups.append((ranks, group))

    my_rank = dist.get_rank()
    for ranks, group in tp_groups:
        if my_rank in ranks:
            return group

    raise RuntimeError(f"Rank {my_rank} not found in any TP group")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_rank():
    return not dist.is_initialized() or dist.get_rank() == 0
