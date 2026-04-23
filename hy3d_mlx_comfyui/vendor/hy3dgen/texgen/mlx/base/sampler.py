# DDIM scheduler for MLX, matching diffusers DDIMScheduler behavior.
# Used by Hunyuan3D-Paint with v_prediction and rescaled zero-SNR betas.

import numpy as np

import mlx.core as mx

from .config import DDIMConfig


def _rescale_zero_terminal_snr(betas):
    """Rescale betas to have zero terminal SNR (from Common Diffusion Noise Schedules)."""
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)

    # Rescale so that the last alpha_cumprod is near zero
    alphas_bar_sqrt = np.sqrt(alphas_cumprod)
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].copy()

    alphas_bar_sqrt = (alphas_bar_sqrt - alphas_bar_sqrt_T) / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    alphas_cumprod = alphas_bar_sqrt ** 2
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = np.clip(alphas, a_min=0.0, a_max=1.0)
    alphas = np.concatenate([alphas_cumprod[:1], alphas])
    betas = 1.0 - alphas
    return betas


class DDIMSampler:
    """DDIM scheduler matching diffusers DDIMScheduler behavior.

    Supports v_prediction and rescaled zero-SNR betas as used by Hunyuan3D-Paint.
    """

    def __init__(self, config: DDIMConfig):
        self.config = config
        self.prediction_type = config.prediction_type

        # Compute beta schedule
        if config.beta_schedule == "scaled_linear":
            betas = np.linspace(
                config.beta_start ** 0.5,
                config.beta_end ** 0.5,
                config.num_train_steps,
            ) ** 2
        elif config.beta_schedule == "linear":
            betas = np.linspace(
                config.beta_start, config.beta_end, config.num_train_steps
            )
        else:
            raise NotImplementedError(f"{config.beta_schedule} is not implemented.")

        if config.rescale_betas_zero_snr:
            betas = _rescale_zero_terminal_snr(betas)

        alphas = 1.0 - betas
        self.alphas_cumprod = mx.array(np.cumprod(alphas).astype(np.float32))
        self.final_alpha_cumprod = mx.array(1.0)

        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int):
        """Set the discrete timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        # Trailing timestep spacing (as used by Hunyuan3D-Paint)
        step_ratio = self.config.num_train_steps // num_inference_steps
        timesteps = np.round(
            np.arange(self.config.num_train_steps, 0, -step_ratio)
        ).astype(np.int64) - 1

        self.timesteps = mx.array(timesteps)

    def scale_model_input(self, sample, timestep):
        """DDIM does not scale the model input."""
        return sample

    def step(self, model_output, timestep, sample, eta=0.0):
        """Perform one DDIM denoising step: x_t → x_{t-1}.

        Args:
            model_output: predicted noise or v-prediction from the UNet
            timestep: current timestep t
            sample: current noisy sample x_t
            eta: DDIM stochasticity (0.0 = deterministic)

        Returns:
            Denoised sample x_{t-1}
        """
        # Get the timestep index for prev
        if self.timesteps is not None:
            t_idx = mx.argmax(self.timesteps == timestep)
            if t_idx + 1 < len(self.timesteps):
                prev_timestep = self.timesteps[t_idx + 1]
            else:
                prev_timestep = mx.array(0)
        else:
            prev_timestep = mx.maximum(
                timestep - self.config.num_train_steps // self.num_inference_steps,
                mx.array(0),
            )

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Convert model output to predicted x_0
        if self.prediction_type == "v_prediction":
            pred_original_sample = (
                mx.sqrt(alpha_prod_t) * sample - mx.sqrt(beta_prod_t) * model_output
            )
            pred_epsilon = (
                mx.sqrt(alpha_prod_t) * model_output + mx.sqrt(beta_prod_t) * sample
            )
        elif self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - mx.sqrt(beta_prod_t) * model_output
            ) / mx.sqrt(alpha_prod_t)
            pred_epsilon = model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Compute the previous sample x_{t-1}
        pred_sample_direction = mx.sqrt(beta_prod_t_prev) * pred_epsilon
        prev_sample = (
            mx.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        # Add noise for eta > 0 (stochastic DDIM)
        if eta > 0:
            variance = (
                (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            )
            std = mx.sqrt(variance) * eta
            noise = mx.random.normal(prev_sample.shape).astype(prev_sample.dtype)
            prev_sample = prev_sample + std * noise

        return prev_sample
