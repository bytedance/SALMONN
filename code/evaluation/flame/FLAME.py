"""Zero-identity FLAME geometry using a separately obtained FLAME2020 asset.

Linear blend skinning comes from the installed smplx dependency.
"""
import os
from pathlib import Path
import torch
from torch import nn
from smplx.lbs import lbs

class FLAMEModel(nn.Module):
    def __init__(self, n_shape=300, n_exp=100):
        super().__init__()
        self.n_shape = n_shape
        asset = torch.load(Path(os.environ["FLAME_ASSET_DIR"]) / "flame_2020.pt", weights_only=True)
        self.register_buffer("v_template", asset["v_template"].float())
        dirs = asset["shapedirs"]
        self.register_buffer("shapedirs", torch.cat([dirs[:, :, :n_shape], dirs[:, :, 300:300+n_exp]], dim=-1).float())
        basis = asset["posedirs"]
        self.register_buffer("posedirs", basis.reshape(-1, basis.shape[-1]).T.float())
        self.register_buffer("J_regressor", asset["J_regressor"].float())
        parents = asset["kintree_table"][0].clone().long(); parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", asset["weights"].float())

    def forward(self, expression, jaw_pose):
        batch = expression.shape[0]
        betas = torch.cat([expression.new_zeros(batch, self.n_shape), expression], dim=-1)
        pose = torch.cat([jaw_pose.new_zeros(batch, 6), jaw_pose, jaw_pose.new_zeros(batch, 6)], dim=-1)
        vertices, _ = lbs(betas, pose, self.v_template, self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents, self.lbs_weights)
        return vertices
