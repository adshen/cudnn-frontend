"""Distributed multi-GPU smoke test, launched via torchrun --nproc_per_node=N.

Each rank binds one GPU, joins a NCCL process group, verifies an all-reduce,
and executes a small cudnn matmul graph on its device. This proves the
SLURM/enroot CI path plus NCCL across the node's GPUs; real distributed
cudnn_frontend tests will replace it.
"""

import os

import torch
import torch.distributed as dist
import cudnn


def run_cudnn_matmul(dev):
    handle = cudnn.create_handle()
    try:
        b, m, n, k = 4, 32, 64, 128
        a_gpu = torch.randn(b, m, k, device=f"cuda:{dev}", dtype=torch.float16)
        b_gpu = torch.randn(b, k, n, device=f"cuda:{dev}", dtype=torch.float16)

        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=handle, stream=stream)

        graph = cudnn.pygraph(
            handle=handle,
            io_data_type=cudnn.data_type.HALF,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        a = graph.tensor_like(a_gpu)
        b_t = graph.tensor_like(b_gpu)
        c = graph.matmul(name="matmul", A=a, B=b_t)
        c.set_output(True).set_data_type(cudnn.data_type.HALF)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

        c_actual = torch.zeros(b, m, n, device=f"cuda:{dev}", dtype=torch.float16)
        workspace = torch.empty(graph.get_workspace_size(), device=f"cuda:{dev}", dtype=torch.uint8)
        graph.execute({a: a_gpu, b_t: b_gpu, c: c_actual}, workspace, handle=handle)
        torch.cuda.synchronize()

        c_expected = torch.matmul(a_gpu, b_gpu)
        torch.testing.assert_close(c_expected, c_actual, atol=5e-2, rtol=5e-2)
    finally:
        cudnn.destroy_handle(handle)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    if rank == 0:
        print(f"cudnn frontend version: {cudnn.__version__}")
        print(f"cudnn backend version: {cudnn.backend_version_string()}")
        print(f"torch {torch.__version__}, world size: {world}")

    # NCCL all-reduce across all ranks: sum of (rank + 1) == world*(world+1)/2
    t = torch.full((1024,), float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(t)
    expected = world * (world + 1) / 2
    torch.testing.assert_close(t, torch.full_like(t, expected))
    print(f"rank {rank}/{world} (cuda:{local_rank}): NCCL all_reduce OK")

    # A small cudnn graph on this rank's GPU
    run_cudnn_matmul(local_rank)
    print(f"rank {rank}: cudnn matmul graph OK on {torch.cuda.get_device_name(local_rank)}")

    dist.barrier()
    if rank == 0:
        print("MULTI-GPU DISTRIBUTED SMOKE PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
