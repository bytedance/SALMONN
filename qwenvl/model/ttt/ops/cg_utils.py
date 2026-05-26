import torch
from qwenvl.model.ttt.ops.utils import gelu_bwd, ln_fused_l2_bwd, ln_fwd
from line_profiler import profile
import torch.nn.functional as F


def norm(a):
    if a is None:
        return None
    elif type(a) is list:
        anorm = 0
        for i in a:
            anorm += (i**2).sum().item()
        return anorm**0.5
    else:
        return (a**2).sum().item()**0.5
    
def dp_list_batch(a, b):
    out = 0
    for ai, bi in zip(a, b):
        out += (ai * bi).sum(dim=(-2, -1))
    return out

def cosine_similarity(a, b):
    if a is None or b is None:
        return None, norm(a), norm(b)
    abdp = 0
    anorm = 0
    bnorm = 0
    if type(a) in (list, tuple) and type(b) in (list, tuple):
        for i, j in zip(a, b):
            abdp += (i * j).sum().item()
            anorm += (i**2).sum().item()
            bnorm += (j**2).sum().item()
    else:
        abdp += (a * b).sum().item()
        anorm += (a**2).sum().item()
        bnorm += (b**2).sum().item()
    if anorm == 0 or bnorm == 0:
        return 0, anorm, bnorm
    anorm = anorm**0.5
    bnorm = bnorm**0.5
    return abdp / anorm / bnorm, anorm, bnorm

class GVP(object):
    """
    Gauss-Newton Vector Product (GVP) for a model with parameters `params`.
    """
    def __init__(self, model_outputs, params):
        self.model_outputs = model_outputs
        self.params = params

        # initialise double differentiation graph
        self.t = torch.ones_like(model_outputs, requires_grad=True)
        self.Jt = torch.autograd.grad(model_outputs, params, self.t, create_graph=True)

    def Jvp(self, v):
        # compute Jv
        Jv = torch.autograd.grad(self.Jt, self.t, v, retain_graph=True)[0]
        return Jv

    def Gvp(self, v):
        # compute J^T J v
        Gv = torch.autograd.grad(self.model_outputs, self.params, self.Jvp(v), retain_graph=True)
        return Gv

# @torch.compile
def cg_mlp(matvec, b, max_iter=8, verbose=True):
    """
    Assume 'b' to be a list of vectors,
    Assume 'b' and matvec both handles additional batch dimensions (apart from the last two dimensions).
    """
    def dp(a, b):
        out = 0
        for ai, bi in zip(a, b):
            out += (ai * bi).sum(dim=(-1, -2), keepdim=True)
        return out
    def add(a, b, scalar_vec):
        return [ai + bi*scalar_vec for ai, bi in zip(a, b)]
    x  = [torch.zeros_like(b_i) for b_i in b]
    r  = [torch.clone(b_i) for b_i in b]
    p  = [torch.clone(b_i) for b_i in b]
    rs = dp(r, r)
    if verbose:
        print(rs.sum().item())
    for _ in range(max_iter):
        Ap     = matvec(p)
        alpha  = rs / dp(p, Ap)
        x      = add(x, p, alpha)
        r      = add(r, Ap, -alpha)
        new_rs = dp(r, r)
        beta   = new_rs / rs
        p      = add(r, p, beta)
        rs     = new_rs
        if verbose:
            print(rs.sum().item())
    return x

def cg_mlp_ES_optimised(matvec, b, max_iter=8, verbose=True, stop_criterion=0.75):
    """
    Assume 'b' to be a list of tensors,
    Assume 'b' and matvec both handles additional batch dimensions (apart from the last two dimensions).
    """
    # allow projection to and from flat vector
    def _make_flat_ops(tensors):
        """Return flat_tensor, flat(list->flat), unflat(flat->list)."""
        batch_shape = tensors[0].shape[:-2]
        shapes      = [t.shape for t in tensors]
        lens        = [int(torch.prod(torch.tensor(s[-2:]))) for s in shapes]

        flat_init   = torch.cat([t.reshape(*batch_shape, -1) for t in tensors], dim=-1)

        def flat(lst):          # list[Tensor] -> Tensor[..., D]
            return torch.cat([v.reshape(*batch_shape, -1) for v in lst], dim=-1)

        def unflat(v_flat):     # Tensor[..., D] -> list[Tensor]
            pieces = torch.split(v_flat, lens, dim=-1)
            return [p.reshape(s) for p, s in zip(pieces, shapes)]

        return flat_init, flat, unflat
    b, flat, unflat = _make_flat_ops(b)
    # project matvec to flat vector input/output
    def matvec_flat(v):
        return flat(matvec(unflat(v)))
    # prepare key function and variables
    x  = torch.zeros_like(b)
    r  = torch.clone(b)
    p  = torch.clone(b)
    rs = (r * r).sum(dim=-1, keepdim=True)
    # prepare for safety check
    # if any of the task is considered to break criteria, stop CG for that task
    # per_task_mask will be set to 0 for that task.
    per_task_xg = None
    per_task_pAp = None
    per_task_indicator = None
    per_task_mask = torch.ones_like(rs).detach()
    disabled_tasks = 0
    if verbose:
        total_pAp = 0
    # start CG iterations
    for _ in range(max_iter):
        # compute to-be-updated vector to x
        Ap     = matvec_flat(p)
        pAp    = (p * Ap).sum(dim=-1, keepdim=True)
        alpha  = rs / pAp

        # compute xg/√pAp to check for monotonicity. If monotonicity is broken, stop CG for that task.
        d_xg = torch.nan_to_num((p * b).sum(dim=-1, keepdim=True) * alpha, nan=0, posinf=0, neginf=0)
        d_pAp = torch.nan_to_num(rs.clone().detach(), nan=0, posinf=0, neginf=0)
        if per_task_xg is None:
            per_task_xg = d_xg
            per_task_pAp = d_pAp
            per_task_indicator = per_task_xg / (per_task_pAp**0.5)
        else:
            per_task_xg += d_xg * per_task_mask
            per_task_pAp += d_pAp * per_task_mask
            # check monotonicity
            this_per_task_indicator = per_task_xg / (per_task_pAp**0.5)
            monotonicity_check = this_per_task_indicator > per_task_indicator
            per_task_mask = (per_task_mask * monotonicity_check).detach() # stop CG for task that breaks monotonicity
            per_task_indicator = this_per_task_indicator
            # output task disabled information
            if verbose:
                cur_disabled_tasks = (per_task_mask == 0).sum().item()
                if cur_disabled_tasks > disabled_tasks:
                    disabled_tasks = cur_disabled_tasks
                    if verbose:
                        print(f"Warning, {disabled_tasks} tasks are disabled due to monotonicity break.")
        # if (per_task_mask==0).any():
        # make avoid NaN and Inf in key variables
        p = torch.nan_to_num(p, nan=0, posinf=0, neginf=0)
        alpha = torch.nan_to_num(alpha, nan=0, posinf=0, neginf=0)
        alpha = alpha * per_task_mask  # disallow update for tasks with per_task_mask == 0

        # update x, and move on with the iterations.
        x      = x + p * alpha
        r      = r - Ap * alpha
        new_rs = (r * r).sum(dim=-1, keepdim=True)
        beta   = new_rs / rs
        p      = r + p * beta
        rs     = new_rs

        # output information when required
        if verbose:
            this_pAp = pAp.clone().detach()
            this_alpha = alpha.clone().detach()
            this_pAp[per_task_mask==0] = 0
            this_alpha[per_task_mask==0] = 0
            total_pAp += (this_pAp*this_alpha).sum().item()
            xg = (x * b).sum()
            x_norm = (x * x).sum().item()**0.5
            indicator = xg/total_pAp**0.5
            print(f"CG iter: {_+1}, xg/√pAp: {indicator:.14f}, rs: {rs.sum().item():.5f}, √pAp: {total_pAp**0.5:.5f}, x norm: {x_norm:.8f}")

        # stop CG when the majority of tasks are disabled
        if (per_task_mask == 0).float().mean() >= stop_criterion:
            if verbose:
                print(f"Majority tasks are disabled, stopping CG at iteration {_+1}.")
            break
    if verbose:
        print(f"CG finished")
    return unflat(x)

def cg_mlp_ES(matvec, b, max_iter=8, verbose=True, stop_criterion=0.75):
    """
    Assume 'b' to be a list of vectors,
    Assume 'b' and matvec both handles additional batch dimensions (apart from the last two dimensions).
    """
    # prepare key function and variables
    def dp(a, b):
        out = 0
        for ai, bi in zip(a, b):
            out += (ai * bi).sum(dim=(-1, -2), keepdim=True)
        return out
    def add(a, b, scalar_vec):
        # disallow addition for tasks with per_task_mask == 0
        scalar_vec = scalar_vec * per_task_mask
        return [ai + bi * scalar_vec for ai, bi in zip(a, b)]
    x  = [torch.zeros_like(b_i) for b_i in b]
    r  = [torch.clone(b_i) for b_i in b]
    p  = [torch.clone(b_i) for b_i in b]
    rs = dp(r, r) 
    # prepare for safety check
    # if any of the task is considered to break criteria, stop CG for that task
    # per_task_mask will be set to 0 for that task.
    per_task_xg = None
    per_task_pAp = None
    per_task_indicator = None
    per_task_mask = torch.ones_like(rs).detach()
    disabled_tasks = 0
    if verbose:
        total_pAp = 0
    # start CG iterations
    for _ in range(max_iter):
        # compute to-be-updated vector to x
        Ap     = matvec(p)
        pAp    = dp(p, Ap)
        alpha  = rs / pAp

        # compute xg/√pAp to check for monotonicity. If monotonicity is broken, stop CG for that task.
        d_xg = torch.nan_to_num((dp(p, b) * alpha).detach(), nan=0, posinf=0, neginf=0)
        d_pAp = torch.nan_to_num(rs.clone().detach(), nan=0, posinf=0, neginf=0)
        if per_task_xg is None:
            per_task_xg = d_xg
            per_task_pAp = d_pAp
            per_task_indicator = per_task_xg / (per_task_pAp**0.5)
        else:
            per_task_xg += d_xg * per_task_mask
            per_task_pAp += d_pAp * per_task_mask
            # check monotonicity
            this_per_task_indicator = per_task_xg / (per_task_pAp**0.5)
            monotonicity_check = this_per_task_indicator > per_task_indicator
            per_task_mask = (per_task_mask * monotonicity_check).detach() # stop CG for task that breaks monotonicity
            per_task_indicator = this_per_task_indicator
            # update when a new task is disabled
            cur_disabled_tasks = (per_task_mask == 0).sum().item()
            if cur_disabled_tasks > disabled_tasks:
                # Broadcast per_task_mask to match p's shape and set masked elements to zero
                disabled_tasks = cur_disabled_tasks
                if verbose:
                    print(f"Warning, {disabled_tasks} tasks are disabled due to monotonicity break.")
        if disabled_tasks > 0:
            # make avoid NaN and Inf in key variables
            p = [torch.nan_to_num(pi, nan=0, posinf=0, neginf=0) for pi in p]
            alpha = torch.nan_to_num(alpha, nan=0, posinf=0, neginf=0)

        # update x, and move on with the iterations.
        x      = add(x, p, alpha)
        r      = add(r, Ap, -alpha)
        new_rs = dp(r, r)
        beta   = new_rs / rs
        p      = add(r, p, beta)
        rs     = new_rs

        # output information when required
        if verbose:
            this_pAp = pAp.clone().detach()
            this_alpha = alpha.clone().detach()
            this_pAp[per_task_mask==0] = 0
            this_alpha[per_task_mask==0] = 0
            total_pAp += (this_pAp*this_alpha).sum().item()
            xg = dp(x, b).sum().item()
            x_norm = dp(x, x).sum().item()**0.5
            indicator = xg/total_pAp**0.5
            print(f"CG iter: {_+1}, xg/√pAp: {indicator:.14f}, rs: {rs.sum().item():.5f}, √pAp: {total_pAp**0.5:.5f}, x norm: {x_norm:.8f}")

        # stop CG when the majority of tasks are disabled
        if disabled_tasks >= per_task_mask.numel() * stop_criterion:
            if verbose:
                print(f"Majority tasks are disabled, stopping CG at iteration {_+1}.")
            break
    if verbose:
        print(f"CG finished")
    return x

def ln_fused_l2_bwd_info(x, l2_target, gamma, beta, eps=1e-8, return_loss=True):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    rstd = torch.rsqrt(var + eps)
    x_hat = (x - mu) * rstd

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        * rstd
    )

    loss = (0.5 * torch.square(grad_output) if return_loss else None)

    return z, loss, y, (mu, var, rstd, x_hat)

def mlp_fwd(XK, XV, W1, W2, bias=None, ln_params=None, return_loss=True):
    X1 = XK
    # do forward pass
    if bias is None:
        Z1 = X1 @ W1
        X2 = torch.nn.functional.gelu(Z1, approximate="tanh")
        Z2 = X2 @ W2
    else:
        b1, b2 = bias
        Z1 = X1 @ W1 + b1
        X2 = torch.nn.functional.gelu(Z1, approximate="tanh")
        Z2 = X2 @ W2 + b2
    # compute loss
    if ln_params is not None:
        ln_weight, ln_bias = ln_params
    else:
        ln_weight, ln_bias = 1.0, 0.0
    reconstruction_target = XV - XK
    grad_l_wrt_Z2, loss, Y, ln_info = ln_fused_l2_bwd_info(Z2, reconstruction_target, ln_weight, ln_bias, return_loss=return_loss)
    # compute gradients
    gelu_bwd_Z1 = gelu_bwd(Z1)
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2.transpose(-2, -1) * gelu_bwd_Z1
    # return loss if requested
    if return_loss:
        loss = loss.sum(dim=(-2, -1))
    return loss, grad_l_wrt_Z2, grad_l_wrt_Z1, gelu_bwd_Z1, X1, Z1, X2, Z2, Y, reconstruction_target, ln_info

def ln_fused_l2_bwd_info_optimised(
    x,
    l2_target,
    gamma,
    beta,
    eps=1e-8,
    return_loss=True,
    ema_factor=0,
    lag_loss_mask=None,
):
    "Batch backward for LayerNorm fused with L2 loss."
    # simutaneously compute mean and variance
    var, mu = torch.var_mean(x, dim=-1, keepdim=True, unbiased=False)

    # Normalization
    rstd = torch.rsqrt(var + eps)
    x_hat = (x - mu) * rstd

    # Scale and shift
    y = gamma * x_hat + beta

    # compute gradients
    if lag_loss_mask is not None:
        grad_output = (y - l2_target) * lag_loss_mask
    else:
        grad_output = y - l2_target
    grad_x_hat = grad_output * gamma

    # faster backward formula
    g_mean   = grad_x_hat.mean(dim=-1, keepdim=True)
    gx_mean  = (grad_x_hat * x_hat).mean(dim=-1, keepdim=True)
    z = (grad_x_hat - g_mean - x_hat * gx_mean) * rstd

    loss = (0.5 * torch.square(grad_output) if return_loss else None)

    return z, loss, y, (mu, var, rstd, x_hat)

def mlp_fwd_optimised(XK, XV, W1, W2, bias, ln_params, return_loss=True, eta=None, ema_factor=0, lag_loss_mask=None):
    # 50% speedup compared to mlp_fwd
    X1 = XK
    # do forward pass
    b1, b2 = bias
    Z1 = X1 @ W1 + b1
    X2 = torch.nn.functional.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2 + b2
    # compute loss
    ln_weight, ln_bias = ln_params
    reconstruction_target = XV - XK
    reconstruction_target_lag = None
    # faster gelu backward
    gelu_bwd_Z1 = torch.ops.aten.gelu_backward(torch.ones_like(Z1), Z1, approximate="tanh")
    # compute gradient backprop
    grad_l_wrt_Z2, loss, Y, ln_info = ln_fused_l2_bwd_info_optimised(
        Z2,
        reconstruction_target,
        ln_weight,
        ln_bias,
        return_loss=return_loss,
        ema_factor=ema_factor,
        lag_loss_mask=lag_loss_mask,
    )
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2.transpose(-2, -1) * gelu_bwd_Z1

    # scale loss
    if return_loss and eta is None:
        loss = loss.sum(dim=(-2, -1))
    elif return_loss and eta is not None:
        # scale loss with eta
        last_eta_mini_batch = eta[:, :, -1, :, None]
        loss = (last_eta_mini_batch * loss).sum(dim=(-2, -1))

    return loss, grad_l_wrt_Z2, grad_l_wrt_Z1, gelu_bwd_Z1, X1, Z1, X2, Z2, Y, reconstruction_target, ln_info

def mlp_fwd_getloss(XK, XV, W1, W2, bias, ln_params, XV_lag=None, XK_lag=None, lag_loss_mask=None, ema_factor=0.0):
    # 50% speedup compared to mlp_fwd
    X1 = XK
    # do forward pass
    b1, b2 = bias
    Z1 = X1 @ W1 + b1
    X2 = torch.nn.functional.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2 + b2
    # compute loss
    ln_weight, ln_bias = ln_params
    reconstruction_target = XV - XK
    # faster gelu backward
    gelu_bwd_Z1 = torch.ops.aten.gelu_backward(torch.ones_like(Z1), Z1, approximate="tanh")
    # compute gradient backprop
    grad_l_wrt_Z2, loss, Y, ln_info = ln_fused_l2_bwd_info_optimised(
        Z2,
        reconstruction_target,
        ln_weight,
        ln_bias,
        return_loss=True,
        ema_factor=ema_factor,
        lag_loss_mask=lag_loss_mask,
    )

    # scale loss
    loss = loss.mean(dim=(-3, -1))
    return loss

def mlp_fwd_no_backward(X, W1, W2, bias, ln_params):
    # trivial way of outputing ZQ (do not use special tricks to compute the new Z2_bar)
    X1_bar = X
    Z1_bar = X1_bar @ W1 + bias[0]
    X2_bar = F.gelu(Z1_bar, approximate="tanh")
    Z2_bar = X2_bar @ W2 + bias[1]

    # do layer norm and residual connection
    Z2_bar = ln_fwd(Z2_bar, ln_params[0], ln_params[1])
    Z_out = Z2_bar + X
    return Z_out

def mlp_Jv(W1, W2, X1, X2, Z1, Z2, gelu_bwd_Z1, v, bias=None):
    if bias is not None:
        dW1, dW2, db1, db2 = v
        dZ1 = X1 @ dW1 + db1
        dX2 = dZ1 * gelu_bwd_Z1
        dZ2 = dX2 @ W2 + X2 @ dW2 + db2
    else:
        dW1, dW2 = v
        dZ1 = X1 @ dW1
        dX2 = dZ1 * gelu_bwd_Z1
        dZ2 = dX2 @ W2 + X2 @ dW2
    return dZ2

def mlp_Gvp(W1, W2, X1, X2, Z1, Z2, gelu_bwd_Z1, v, ln_info=None, bias=None, ln_params=None, ln_mode="none"):
    # ln_mode can be "none", "GN", or "H"
    dZ2 = mlp_Jv(W1, W2, X1, X2, Z1, Z2, gelu_bwd_Z1, v, bias=bias)
    if ln_mode == "none":
        g_Z2 = dZ2
    elif ln_mode == "GN":
        g_Z2 = ln_Gvp(Z2, dZ2, ln_info, ln_params=ln_params)
    elif ln_mode == "H":
        # g_Z2 = ln_Hvp(Z2, dZ2, l2_target, ln_params=ln_params)
        raise NotImplementedError("GGN involving Hessian for LayerNorm is not PSD, so it cannot be used with CG. Use 'GN' mode instead.")
    g_Z1 = g_Z2 @ W2.transpose(-2, -1) * gelu_bwd_Z1
    dW1 = X1.transpose(-1, -2) @ g_Z1
    dW2 = X2.transpose(-1, -2) @ g_Z2
    if bias is None:
        return [dW1, dW2]
    else:
        db1 = g_Z1.sum(dim=-2, keepdim=True)
        db2 = g_Z2.sum(dim=-2, keepdim=True)
        return [dW1, dW2, db1, db2]

def ln_Gvp(x, v, ln_info, ln_params=None):
    if ln_params is not None:
        ln_weight, ln_bias = ln_params
    else:
        ln_weight, ln_bias = 1, 0
    # unpack ln_info
    mu, var, rstd, xhat = ln_info

    # forward Jvp
    z   = x - mu
    dv  = v - v.mean(dim=-1, keepdim=True)

    # ---- J · v --------------------------------------------------------
    inner = (z * dv).mean(dim=-1, keepdim=True)        #  ⟨z,dv⟩ / D
    jv = (dv - xhat * inner * rstd**2)*rstd           #  correct Jv

    # ---------- Jᵀ · (gamma ⊙ Jv)  ------------------------------------
    gw   = ln_weight * jv                               # γ ⊙ Jv
    g_mean  = gw.mean(dim=-1, keepdim=True)
    gx_mean = (gw * xhat).mean(dim=-1, keepdim=True)
    dx = (gw - g_mean - xhat * gx_mean) * rstd      # Jᵗ…

    return dx

def ln_Hvp(x, dx, l2_target, eps=1e-8, ln_params=None):
    """
    Plain (non-autograd) Hessian-vector product of the fused
    layer-norm-plus-L2-loss with respect to `x`.

    Note that this Hessian is not PSD, so it cannot work with CG

    Parameters
    ----------
    x         : (..., D)   point at which the HVP is evaluated
    v         : (..., D)   vector to be multiplied by the Hessian
    l2_target : (..., D)
    gamma     : (..., D)
    beta      : (..., D)
    eps       : float      (same ε as in forward)

    Returns
    -------
    hv        : (..., D)   (∂²L/∂x²)·v   with exactly the same shape as x
    """
    if ln_params is not None:
        gamma, beta = ln_params
    else:
        gamma, beta = 1, 0

    # =========================== forward pass =========================== #
    D = x.shape[-1]                               # feature dimension
    mu   = x.mean(dim=-1, keepdim=True)           # (...,1)
    z    = x - mu                                 # centred
    var  = (z**2).mean(dim=-1, keepdim=True)      # (...,1)
    std  = torch.sqrt(var + eps)                  # (...,1)  σ
    inv_std = 1.0 / std                           # (...,1)  1/σ
    xhat = z * inv_std                            # (...,D)  x̂

    y = gamma * xhat + beta                       # (...,D)
    e = y - l2_target                             # (...,D)  residual
    e_gamma = e * gamma                           # (...,D)

    # Scalars that appear in the gradient formula (keep last-axis dims!)
    sum_e_gamma       = e_gamma.sum(dim=-1, keepdim=True)          # (...,1)
    sum_xhat_e_gamma  = (xhat * e_gamma).sum(dim=-1, keepdim=True) # (...,1)

    # ====================== directional derivatives ===================== #
    # Tangent of μ, z, σ, 1/σ, x̂ along the direction v
    dmu      = dx.mean(dim=-1, keepdim=True)                     # (...,1)
    dz       = dx - dmu                                          # (...,D)
    inner    = (z * dz).sum(dim=-1, keepdim=True)               # zᵀ·dz  (...,1)

    dvar     = (2.0 / D) * inner                                # (...,1)
    dstd     = 0.5 * dvar / std                                 # (...,1)
    dinv_std = -dstd / (std**2)                                 # (...,1)

    dxhat    = dz * inv_std + z * dinv_std                      # (...,D)

    de       = gamma * dxhat                                    # (...,D)
    de_gamma = de * gamma                                       # (...,D)

    d_sum_e_gamma      = de_gamma.sum(dim=-1, keepdim=True)     # (...,1)
    d_sum_xhat_e_gamma = (
        dxhat * e_gamma + xhat * de_gamma
    ).sum(dim=-1, keepdim=True)                                 # (...,1)

    # ========================== Hessian·v =============================== #
    A  = D * e_gamma - sum_e_gamma - xhat * sum_xhat_e_gamma
    dA = (D * de_gamma
          - d_sum_e_gamma
          - dxhat * sum_xhat_e_gamma
          - xhat * d_sum_xhat_e_gamma)

    hv = (dinv_std * A + inv_std * dA) / D                      # (...,D)
    return hv
