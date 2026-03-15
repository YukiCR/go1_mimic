from __future__ import annotations

import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierBase, ModifierCfg


def depth_one_col(data: torch.Tensor) -> torch.Tensor:
    """
    get only one column of the depth image, this is a testing func to see if modifier works for mdp.image
    Args:
        data: the input depth image, shape is (batch_size, 64, 64, 1)
    the output is the same depth image but only with the first column, shape is (batch_size, 64)
    """
    return data[:, :, 0, 0]


class DepthAutoencoderModifier(ModifierBase):
    """Modifier that encodes depth images into latent vectors using a pre-trained autoencoder.

    This modifier takes depth images of shape [B, H, W, C], normalizes them, and passes them
    through a JIT-compiled autoencoder encoder to produce latent vectors of shape [B, 64].

    The modifier handles:
    - Input normalization (configurable mean and std, defaults: mean=1.77, std=2.58)
    - Tensor reshaping from [B, H, W, C] to [B, C, H, W]
    - Batch processing (max batch size of 64)
    - Device placement
    """

    def __init__(self, cfg: DepthAutoencoderModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        """Initializes the DepthAutoencoderModifier.

        Args:
            cfg: Configuration parameters for the modifier.
            data_dim: The dimensions of the data to be modified. First element is the batch size.
            device: The device to run the modifier on.
        """
        super().__init__(cfg, data_dim, device)

        # Normalization parameters from config
        self.mean = torch.tensor([cfg.mean], device=device)
        self.std = torch.tensor([cfg.std], device=device)

        # Maximum batch size for processing
        self.max_batch_size = cfg.max_batch_size

        # Load the JIT model
        checkpoint_path = cfg.checkpoint_path

        # Resolve the checkpoint path if relative
        if not os.path.isabs(checkpoint_path):
            current_dir = os.getcwd()
            checkpoint_path = os.path.join(current_dir, checkpoint_path)

        self.model = torch.jit.load(checkpoint_path, map_location=device)
        self.model.eval()

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the modifier state.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        # No stateful operations in this modifier, nothing to reset
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Encodes depth images into latent vectors.

        Args:
            data: The depth image data of shape [B, 64, 64, 1] where B is the batch size.

        Returns:
            Latent vectors of shape [B, 64].
        """
        batch_size = data.shape[0]

        # Permute from [B, H, W, C] to [B, C, H, W]
        # Input is [B, 64, 64, 1], we need [B, 1, 64, 64]
        data = data.permute(0, 3, 1, 2)

        # Normalize: (input - mean) / std
        data = (data - self.mean) / self.std

        # Process in batches if needed
        if batch_size <= self.max_batch_size:
            # Process all at once
            with torch.no_grad():
                latent = self.model(data)
            return latent
        else:
            # Process batch-by-batch
            latents = []
            for i in range(0, batch_size, self.max_batch_size):
                end_idx = min(i + self.max_batch_size, batch_size)
                batch = data[i:end_idx]
                with torch.no_grad():
                    batch_latent = self.model(batch)
                latents.append(batch_latent)
            return torch.cat(latents, dim=0)


@configclass
class DepthAutoencoderModifierCfg(ModifierCfg):
    """Configuration parameters for DepthAutoencoderModifier.

    For more information, please check the :class:`DepthAutoencoderModifier` class.
    """

    func: type[DepthAutoencoderModifier] = DepthAutoencoderModifier
    """The depth autoencoder modifier class to be instantiated."""

    checkpoint_path: str = MISSING
    """Path to the JIT-compiled autoencoder encoder checkpoint."""

    max_batch_size: int = MISSING
    """Maximum batch size for processing. If input batch is larger, it will be processed in chunks."""

    mean: float = MISSING
    """Mean value for input normalization (depth in meters)."""

    std: float = MISSING
    """Standard deviation value for input normalization (depth in meters)."""