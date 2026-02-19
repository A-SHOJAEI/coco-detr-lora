from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def get_dist_info(requested_ddp: bool) -> DistInfo:
    if not requested_ddp:
        return DistInfo(enabled=False, rank=0, world_size=1, local_rank=0)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    return DistInfo(enabled=enabled, rank=rank, world_size=world_size, local_rank=local_rank)


def init_distributed(dist_info: DistInfo) -> None:
    if not dist_info.enabled:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=dist_info.rank, world_size=dist_info.world_size)


def destroy_distributed(dist_info: DistInfo) -> None:
    if dist_info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(dist_info: DistInfo) -> None:
    if dist_info.enabled and dist.is_initialized():
        dist.barrier()

