import os
import random

import numpy as np
import torch
import torch.distributed as dist
from cruise.utilities.distributed import DIST_ENV

_MAX_DATA_DIM = 5


# ======================================== sequence parallel ========================================
def _check_data_types(keys, data, target_dtype):
    """Check that all the keys have the same target data type."""
    for key in keys:
        assert data[key].dtype == target_dtype, '{} has data type {} which ' 'is different than {}'.format(
            key, data[key].dtype, target_dtype
        )


def _build_key_size_numel_dictionaries(keys, data):
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    # Pack the sizes on rank zero.
    if get_sequence_parallel_rank() == 0:
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, 'you should increase MAX_DATA_DIM'
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    # Move to GPU and broadcast.
    sizes_cuda = torch.tensor(sizes, dtype=torch.long, device='cuda')
    torch.distributed.broadcast(
        sizes_cuda, get_sequence_parallel_src_rank(), group=DIST_ENV.distributed_sequence_parallel_group
    )

    # Move back to cpu and unpack.
    sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Args:
        keys: list of keys in the data disctionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
    """
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data)

    # Pack on rank zero.
    if get_sequence_parallel_rank() == 0:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        flatten_data = torch.cat([data[key].contiguous().view(-1) for key in keys], dim=0).cuda()
    else:
        flatten_data = torch.empty(total_numel, device=torch.cuda.current_device(), dtype=datatype)

    # Broadcast
    torch.distributed.broadcast(
        flatten_data, get_sequence_parallel_src_rank(), group=DIST_ENV.distributed_sequence_parallel_group
    )

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output


def broadcast_obj_across_sequence_parallel_group(objects: list, device=None):
    torch.distributed.broadcast_object_list(
        objects, get_sequence_parallel_src_rank(), group=DIST_ENV.distributed_sequence_parallel_group
    )
    if device is not None:
        for obj in objects:
            if isinstance(obj, torch.Tensor):
                obj.to(device=device)
    return objects


def enable_sequence_parallel() -> bool:
    return DIST_ENV.distributed_sequence_parallel_size > 1


def get_sequence_parallel_src_rank():
    global_rank = torch.distributed.get_rank()
    local_world_size = DIST_ENV.distributed_sequence_parallel_size
    return (global_rank // local_world_size) * local_world_size


def get_sequence_parallel_rank() -> int:
    if not enable_sequence_parallel():
        return 0
    return torch.distributed.get_group_rank(DIST_ENV.distributed_sequence_parallel_group, DIST_ENV.rank)


def clear_parallel_state():
    DIST_ENV.tensor_model_parallel_size = 1
    DIST_ENV.tensor_model_parallel_rank = 0
    DIST_ENV.pipeline_model_parallel_size = 1
    DIST_ENV.pipeline_model_parallel_rank = 0
    DIST_ENV.distributed_sequence_parallel_size = 1
    DIST_ENV.distributed_sequence_parallel_rank = 0
    DIST_ENV.bitwise_ckpt = False
    DIST_ENV.dataset_use_auto_len = False
    DIST_ENV.data_parallel_group = None


def initialize_parallel_state(
    data_parallel_size: int = 1,
    sequence_parallel_size: int = 1,
):
    assert torch.distributed.is_initialized()
    world_size = DIST_ENV.world_size
    assert world_size == data_parallel_size * sequence_parallel_size

    num_data_parallel_groups = world_size // data_parallel_size
    num_sequence_parallel_groups = world_size // sequence_parallel_size

    rank = DIST_ENV.rank
    all_sequence_group_ranks = []
    if sequence_parallel_size > 1:
        for i in range(num_sequence_parallel_groups):
            ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
            all_sequence_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                DIST_ENV.setup_distributed_sequence_parallel(
                    sequence_parallel_size, torch.distributed.get_group_rank(group, rank), group
                )
    else:
        all_sequence_group_ranks = [[i] for i in range(world_size)]

    all_data_group_ranks = []
    for i in range(num_data_parallel_groups):
        ranks = [sequence_group_ranks[i] for sequence_group_ranks in all_sequence_group_ranks]
        all_data_group_ranks.append(list(ranks))
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            DIST_ENV.setup_data_parallel(data_parallel_size, torch.distributed.get_group_rank(group, rank), group)


# ======================================== all to all ========================================
# 2024.04.10 copy from opensora https://github.com/hpcaitech/Open-Sora/blob/main/opensora/acceleration/communications.py


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.
    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        return _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _all_to_all(
                grad_output,
                ctx.world_size,
                ctx.process_group,
                ctx.gather_dim,
                ctx.scatter_dim,
            ),
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)


# =======================================================================================


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class _AllToAll_single_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group

        input = input.contiguous()
        output = torch.empty_like(input)

        torch.distributed.all_to_all_single(
            output,
            input,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            None,
            _AllToAll_single_impl.apply(ctx.group, *grad_output),
        )


def all_to_all_single_impl(group, input_):
    return _AllToAll_single_impl.apply(group, input_)


def all_to_all_s2h(input_):
    # [seq/sp_size, head_num, hidden] -> [seq, head_num/sp_size, hidden]
    sp_size = DIST_ENV.distributed_sequence_parallel_size
    sp_group = DIST_ENV.distributed_sequence_parallel_group
    head_num, hidden_size = input_.shape[1], input_.shape[2]
    input_ = input_.reshape(input_.shape[0], -1)
    split_tensors = torch.split(input_, split_size_or_sections=input_.shape[-1] // sp_size, dim=1)
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all_single_impl(sp_group, concat_tensor)
    output = output.reshape(-1, head_num // sp_size, hidden_size)
    return output


def all_to_all_h2s(input_):
    # [seq, head_num/sp_size, hidden] -> [seq/sp_size, head_num, hidden]
    sp_size = DIST_ENV.distributed_sequence_parallel_size
    sp_group = DIST_ENV.distributed_sequence_parallel_group

    seq_len, hidden_size = input_.shape[0], input_.shape[2]
    input_ = input_.reshape(input_.shape[0], -1)

    input_exchanged = all_to_all_single_impl(sp_group, input_)
    input_reshaped = input_exchanged
    split_tensors = torch.split(input_reshaped, split_size_or_sections=input_reshaped.shape[0] // sp_size, dim=0)
    output = torch.cat(split_tensors, dim=-1)
    output = output.reshape(seq_len // sp_size, -1, hidden_size)
    return output
