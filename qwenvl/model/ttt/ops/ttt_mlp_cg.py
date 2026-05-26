import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from qwenvl.model.ttt.ops.utils import gelu_bwd, ln_fused_l2_bwd, ln_fwd
from qwenvl.model.ttt.utils import scan
from qwenvl.model.ttt.ops.cg_utils import *
import numpy as np

# @torch.compile
def compute_mini_batch(params_dict, inputs):
    """
    Notes:
    1. This function always takes one Timestep at a time (a block of multi-time-step vectors).
    2. All dimensions are provided in comments. B: Batch Size, H: # of Heads, N: # of Time steps (in this minibatch), D: Hidden Dimension (for each head).
    """
    W1_init = params_dict["W1_states"] # (B, H, D, D)
    b1_init = params_dict["b1_states"] # (B, H, 1, D)
    W2_init = params_dict["W2_states"] # (B, H, D, D)
    b2_init = params_dict["b2_states"] # (B, H, 1, D)

    ttt_norm_weight = params_dict["ttt_norm_weight"]
    ttt_norm_bias = params_dict["ttt_norm_bias"]

    cg_max_iter = params_dict["cg_max_iter"]

    XQ_mini_batch = inputs["XQ"] # (B, H, N, D)
    XV_mini_batch = inputs["XV"] # (B, H, N, D)
    XK_mini_batch = inputs["XK"] # (B, H, N, D)
    ema_factor = params_dict["ema_factor"] if "ema_factor" in params_dict else 0
    lag_loss_mask = inputs["lag_loss_mask"] if "lag_loss_mask" in inputs else None # (B, H, N, 1)

    eta_mini_batch = inputs["eta"] # (B, H, *N, N), the 3rd dimension is boradcasted (*N), meaning every column shares the same eta value.

    num_heads = XQ_mini_batch.size(1)
    head_dim = XQ_mini_batch.size(-1)
    ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
    ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)

    loss, grad_l_wrt_Z2, grad_l_wrt_Z1, gelu_bwd_Z1, X1, Z1, X2, Z2, Y, l2_target, ln_info = \
        mlp_fwd_optimised(
            XK_mini_batch,
            XV_mini_batch,
            W1_init,
            W2_init,
            bias=(b1_init, b2_init),
            ln_params=(ln_weight, ln_bias),
            return_loss=False,
            ema_factor=ema_factor,
            lag_loss_mask=lag_loss_mask,
        )

    # compute gradients w.r.t. W1, W2, b1, b2
    last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
    gW1 = X1.transpose(-1, -2) @ (last_eta_mini_batch * grad_l_wrt_Z1) # eta now applies to grad_l_wrt_Z1
    gb1 = torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
    gW2 = X2.transpose(-1, -2) @ (last_eta_mini_batch * grad_l_wrt_Z2) # eta now applies to grad_l_wrt_Z2
    gb2 = torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)

    # do CG
    def matvec(v):
        out = mlp_Gvp(W1_init, W2_init, X1, X2, Z1, Z2, gelu_bwd_Z1, v, \
                    bias=(b1_init, b2_init), ln_params=(ln_weight, ln_bias), ln_mode="none", ln_info=ln_info)
        return out
    if cg_max_iter > 0:
        dcgW1, dcgW2, dcgb1, dcgb2 = cg_mlp_ES_optimised(matvec, [gW1, gW2, gb1, gb2], max_iter=cg_max_iter, verbose=False)
        # dcgW1n, dcgW2n, dcgb1n, dcgb2n = cg_mlp_ES(matvec, [gW1, gW2, gb1, gb2], max_iter=cg_max_iter, verbose=False)
    else:
        dcgW1, dcgW2, dcgb1, dcgb2 = gW1, gW2, gb1, gb2

    # update new weights
    W1_last = W1_init - dcgW1
    W2_last = W2_init - dcgW2
    b1_last = b1_init - dcgb1
    b2_last = b2_init - dcgb2

    XQW_mini_batch = mlp_fwd_no_backward(XQ_mini_batch, W1_last, W2_last, (b1_last, b2_last), (ln_weight, ln_bias))

    if "losses" in params_dict:
        with torch.no_grad():
            loss = mlp_fwd_getloss(
                XK_mini_batch,
                XV_mini_batch,
                W1_last,
                W2_last,
                (b1_last, b2_last),
                (ln_weight, ln_bias),
                lag_loss_mask=lag_loss_mask,
                ema_factor=ema_factor,
            )
            params_dict["losses"].append(loss)

    if "losses" not in params_dict:
        last_param_dict = {
            "W1_states": W1_last,
            "b1_states": b1_last,
            "W2_states": W2_last,
            "b2_states": b2_last,
            "ttt_norm_weight": ttt_norm_weight,
            "ttt_norm_bias": ttt_norm_bias,
            "cg_max_iter": cg_max_iter,
            "time_step": params_dict["time_step"] + 1,
            "ema_factor": ema_factor,
        }
    else:
        last_param_dict = {
            "W1_states": W1_last,
            "b1_states": b1_last,
            "W2_states": W2_last,
            "b2_states": b2_last,
            "ttt_norm_weight": ttt_norm_weight,
            "ttt_norm_bias": ttt_norm_bias,
            "cg_max_iter": cg_max_iter,
            "time_step": params_dict["time_step"] + 1,
            "ema_factor": ema_factor,
            "losses": params_dict["losses"],
        }

    return last_param_dict, XQW_mini_batch


def ttt_mlp_cg(
    XK,
    XQ,
    XV,
    eta,
    ttt_norm_weight,
    ttt_norm_bias,
    W1_init,
    b1_init,
    W2_init,
    b2_init,
    checkpoint_group_size,
    cg_max_iter=4,
    ema_factor=0,
    lag_loss_mask=None,
    return_loss=False,
):
    init_params_dict = {
        "W1_states": W1_init,
        "b1_states": b1_init,
        "W2_states": W2_init,
        "b2_states": b2_init,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
        "cg_max_iter": cg_max_iter,
        "time_step": 0,
        "ema_factor": ema_factor,
    }
    if return_loss:
        init_params_dict["losses"] = []

    inputs = {
        "XK": XK,
        "XQ": XQ,
        "XV": XV,
        "eta": eta,
    }
    if lag_loss_mask is not None:
        inputs["lag_loss_mask"] = lag_loss_mask

    # Reorder such that mini-batch is first dimension for iteration
    inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

    XQW_batch = torch.empty_like(inputs["XK"])

    params_dict, XQW_batch = scan(
        compute_mini_batch,  # Function to iterate over
        init_params_dict,
        inputs,
        checkpoint_group_size,
    )

    return params_dict, XQW_batch.permute(1, 0, 3, 2, 4)