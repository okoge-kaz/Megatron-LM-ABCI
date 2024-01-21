import os
import torch
import torch.distributed as dist

rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))
local_world_size = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", 1))

os.environ["RANK"] = str(rank)
os.environ["LOCAL_RANK"] = str(local_rank)
os.environ["WORLD_SIZE"] = str(world_size)
os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

print(rank, 'entering barrier...')
torch.distributed.barrier()
print(rank, 'through barrier...')
