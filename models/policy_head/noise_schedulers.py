# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/baaivision/UniVLA. The original license is located at 'third-party-license/UniVLA.txt'.

import torch
from torch.distributions import Beta, Uniform

class FlowMatchingScheduler:
    def __init__(self, sample_method="beta", s=0.999):
        """
        Initialize the scheduler.

        Args:
            sample_method (str): The sampling method. Must be "uniform" or "beta".
            s (float): Threshold for timesteps.
        """
        assert sample_method in ["uniform", "beta"], "Invalid sampling method"
        self.sample_method = sample_method
        self.s = s  # The threshold for timesteps

        if self.sample_method == "beta":
            # Beta(1.5, 1.0) distribution
            self.distribution = Beta(torch.tensor([1.5]), torch.tensor([1.0]))
        elif self.sample_method == "uniform":
            # Uniform distribution from [0, s]
            self.distribution = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

    def sample_t(self, num_samples):
        """
        Sample timesteps using the specified distribution.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Timesteps sampled and scaled to [0, s].
        """
        if self.sample_method == "beta":
            # Sample from the Beta distribution and scale to [0, s]
            samples = self.distribution.sample((num_samples,))
            timesteps = self.s * (1 - samples)  # Scale to [0, s]
        else:
            # Sample uniformly in [0, s]
            timesteps = self.distribution.sample((num_samples,)) * self.s

        return timesteps

    def add_noise(self, original_samples, noise, timesteps):
        """
        Add noise to the original samples based on the sampled timesteps.

        Args:
            original_samples (torch.FloatTensor): Original sample data.
            noise (torch.FloatTensor): Noise to add to the samples.
            timesteps (torch.FloatTensor): Timesteps sampled from the distribution.

        Returns:
            torch.FloatTensor: Noisy samples.
        """
        # Expand timesteps to match the shape of noise
        while len(timesteps.shape) < len(noise.shape):
            timesteps = timesteps.unsqueeze(-1)
        timesteps = timesteps.expand_as(noise)

        # Add noise directly using sampled timesteps (τ)
        return (1 - timesteps) * original_samples + timesteps * noise
    

# Example usage
if __name__ == "__main__":
    scheduler = FlowMatchingScheduler(sample_method="beta", s=0.999)

    # Example inputs
    num_samples = 10  # Batch size
    original_samples = torch.randn((num_samples, 8, 7))  # Original action samples (B, 8, 7)
    noise = torch.randn((num_samples, 8, 7))  # Noise tensor with the same shape
    timesteps = scheduler.sample_t(num_samples)  # Sample timesteps for the batch

    # Add noise to the original samples
    noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)

    print("Timesteps shape:", timesteps.shape)  # Ensure shape matches (B,)
    print("Timesteps:", timesteps)
    print("Noisy Samples shape:", noisy_samples.shape)  # Ensure shape matches (B, 8, 7)
    print("Noisy Samples:", noisy_samples)
