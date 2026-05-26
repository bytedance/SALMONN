import torch
import torch.nn.functional as F


def memory_bank_compress_MALLM(memory_bank: torch.Tensor, compression_size: torch.Tensor, sync: bool=False) -> tuple:
    """
    Compresses the memory bank if the current memory bank length is greater than the threshold.
    Compression_size is the number of frames that are compressed into each position.
    
    Args:
        memory_bank (torch.Tensor): The input memory bank to compress. Shape: (B, T, N, C)
        compression_size (torch.Tensor): The number of frames to compress into each position. Shape: (B, T, N)
    
    Returns:
        compressed_memory_bank (torch.Tensor): The compressed memory bank. Shape: (B, T-1, N, C)
        compressed_size (torch.Tensor): The number of frames compressed into each position. Shape: (B, T-1, N)
    """
    B, T, N, C = memory_bank.shape
    # Calculate the cosine similarity between adjacent frames
    similarity_matrix = F.cosine_similarity(memory_bank[:, :-1, :], memory_bank[:, 1:, :], dim=-1)
    if sync:
        similarity_matrix = similarity_matrix.mean(-1, keepdim=True).expand(-1, -1, N)
    # Select the frame indices with the top-1 similarity 
    _, max_similarity_indices = torch.max(similarity_matrix, dim=1, keepdim=True)

    # Calculate source and dst indices for compression
    src_indices = max_similarity_indices + 1
    dst_indices = torch.arange(T - 1).to(memory_bank.device)[None, :, None].repeat(B, 1, N)
    dst_indices[dst_indices > max_similarity_indices] += 1

    # Gather source and dst memory banks and sizes
    src_memory_bank = memory_bank.gather(dim=1, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    dst_memory_bank = memory_bank.gather(dim=1, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    src_size = compression_size.gather(dim=1, index=src_indices)
    dst_size = compression_size.gather(dim=1, index=dst_indices)

    # Multiply the memory banks by their corresponding sizes
    src_memory_bank *= src_size.unsqueeze(-1)
    dst_memory_bank *= dst_size.unsqueeze(-1)

    # Compress the memory bank by adding the source memory bank to the dst memory bank
    dst_memory_bank.scatter_add_(dim=1, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C), src=src_memory_bank)
    dst_size.scatter_add_(dim=1, index=max_similarity_indices, src=src_size)

    # Normalize the dst memory bank by its size
    compressed_memory_bank = dst_memory_bank / dst_size.unsqueeze(-1)
    return compressed_memory_bank, dst_size


def memory_bank_compress_MALLM_hard(memory_bank: torch.Tensor, sync: bool=False) -> tuple:
    """
    Compresses the memory bank if the current memory bank length is greater than the threshold.
    Different from MA-LLM, this method replace the tgt features with src ones
    
    Args:
        memory_bank (torch.Tensor): The input memory bank to compress. Shape: (B, T, N, C)
    
    Returns:
        compressed_memory_bank (torch.Tensor): The compressed memory bank. Shape: (B, T-1, N, C)
    """
    B, T, N, C = memory_bank.shape
    # Calculate the cosine similarity between adjacent frames
    similarity_matrix = F.cosine_similarity(memory_bank[:, :-1, :], memory_bank[:, 1:, :], dim=-1)
    if sync:
        similarity_matrix = similarity_matrix.mean(-1, keepdim=True).expand(-1, -1, N)
    # Select the frame indices with the top-1 similarity 
    _, max_similarity_indices = torch.max(similarity_matrix, dim=1, keepdim=True)

    # Calculate source and dst indices for compression
    src_indices = max_similarity_indices + 1
    dst_indices = torch.arange(T - 1).to(memory_bank.device)[None, :, None].repeat(B, 1, N)
    dst_indices[dst_indices > max_similarity_indices] += 1

    # Gather source and dst memory banks and sizes
    src_memory_bank = memory_bank.gather(dim=1, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    dst_memory_bank = memory_bank.gather(dim=1, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))

    # Compress the memory bank by adding the source memory bank to the dst memory bank
    dst_memory_bank.scatter_(dim=1, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C), src=src_memory_bank)

    # Normalize the dst memory bank by its size
    compressed_memory_bank = dst_memory_bank
    return compressed_memory_bank


def memory_bank_compress_keyframe(memory_bank: torch.Tensor, tgt_mem_len: int, window_size: int=3, sync: bool=True) -> tuple:
    """
    Compresses the memory bank if the current memory bank length is greater than the threshold.
    Different from MA-LLM, this method replace the tgt features with src ones
    
    Args:
        memory_bank (torch.Tensor): The input memory bank to compress. Shape: (B, T, N, C)
    
    Returns:
        compressed_memory_bank (torch.Tensor): The compressed memory bank. Shape: (B, T-1, N, C)
        keypatches_mask (torch.Tensor): The compressed memory bank. Shape: (T-1 * N)
    """
    B, T, N, C = memory_bank.shape
    # Calculate the cosine similarity between adjacent frames
    similarity_matrix = F.cosine_similarity(memory_bank[:, :-1, :], memory_bank[:, 1:, :], dim=-1)
    dis_matrix = 1 - similarity_matrix[0].type(torch.float)
    # dis-similarity of the (i)th frame with the (i-1)th frame
    dis_matrix = torch.cat([
        torch.ones_like(dis_matrix[:1]),
        dis_matrix
    ], dim=0) # [T, N]

    if sync:
        # Meanpool over spatial locations
        dis_matrix = dis_matrix.mean(1) # [T]
        keypatches_mask = torch.zeros_like(dis_matrix).bool()

        # Argrelmax
        try:
            if torch.npu.is_available():
                # F.max_pool1d_with_indices is not supported and returns wrong tensor
                device = dis_matrix.device
                dis_matrix = dis_matrix.cpu()
        except:
            pass
        window_maxima = F.max_pool1d_with_indices(dis_matrix[None,None,:], window_size, 1, padding=window_size//2)[1].squeeze() # [T]
        candidates = window_maxima.unique()
        peaks = candidates[(window_maxima[candidates]==candidates).nonzero()].squeeze()
        try:
            if torch.npu.is_available():
                dis_matrix = dis_matrix.to(device)
                peaks = peaks.to(device)
        except:
            pass

        # Fill remaining frames
        keypatches_mask[peaks] = True
        dis_matrix[peaks] += 2 # select from peaks first
        peaks = torch.topk(dis_matrix, k=tgt_mem_len, sorted=False)[1] # [t]
        peaks = peaks.sort()[0]

        # Get keyframe memory
        compressed_memory_bank = memory_bank[:,peaks] # [B, t, N, C]
        keypatches_mask = keypatches_mask[peaks] # [t]
        keypatches_mask = keypatches_mask[:, None].repeat(1, N) # [t, N]
    else:
        dis_matrix = dis_matrix.transpose(0, 1) # [N, T]
        keypatches_mask = torch.zeros_like(dis_matrix).bool()
        # Argrelmax
        try:
            if torch.npu.is_available():
                # F.max_pool1d_with_indices is not supported and returns wrong tensor
                device = dis_matrix.device
                dis_matrix = dis_matrix.cpu()
                keypatches_mask = keypatches_mask.cpu()
        except:
            pass
        window_maxima = F.max_pool1d_with_indices(dis_matrix[:,None,:], window_size, 1, padding=window_size//2)[1].squeeze() # [N, T]
        for p, window_maxima_patch in enumerate(window_maxima):
            candidates_patch = window_maxima_patch.unique()
            peaks_patch = candidates_patch[(window_maxima_patch[candidates_patch]==candidates_patch).nonzero()][:,0]

            # Fill remaining frames
            keypatches_mask[p, peaks_patch] = True
            dis_matrix[p, peaks_patch] += 2
        try:
            if torch.npu.is_available():
                dis_matrix = dis_matrix.to(device)
                keypatches_mask = keypatches_mask.to(device)
        except:
            pass
        peaks = torch.topk(dis_matrix, k=tgt_mem_len, sorted=False, dim=1)[1] # [N, t]
        peaks = peaks.sort(dim=1)[0]
        peaks = peaks.transpose(0, 1) # [t, N]
        keypatches_mask = keypatches_mask.transpose(0, 1) # [t, N]

        # Get keyframe memory
        compressed_memory_bank = memory_bank.gather(dim=1, index=peaks[None,:,:,None].expand(-1, -1, -1, C))
        # [B, t, N, C]
        keypatches_mask = keypatches_mask.gather(dim=0, index=peaks) # [t, N]

    return compressed_memory_bank, keypatches_mask.flatten()