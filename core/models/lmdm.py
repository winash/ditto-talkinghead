import numpy as np
import torch
from ..utils.load_model import load_model


def make_beta(n_timestep, cosine_s=8e-3):
    timesteps = (
        torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
    )
    alphas = timesteps / (1 + cosine_s) * np.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = np.clip(betas, a_min=0, a_max=0.999)
    return betas.numpy()


class LMDM:
    def __init__(self, model_path, device="cuda", **kwargs):
        kwargs["module_name"] = "LMDM"

        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

        self.motion_feat_dim = kwargs.get("motion_feat_dim", 265)
        self.audio_feat_dim = kwargs.get("audio_feat_dim", 1024+35)
        self.seq_frames = kwargs.get("seq_frames", 80)

        if self.model_type == "pytorch":
            pass
        else:
            self._init_np()

    def setup(self, sampling_timesteps, noise_guidance=0.25):
        if self.model_type == "pytorch":
            self.model.setup(sampling_timesteps)
        else:
            self._setup_np(sampling_timesteps, noise_guidance)

    def _init_np(self):
        self.sampling_timesteps = None
        self.n_timestep = 1000

        betas = torch.Tensor(make_beta(n_timestep=self.n_timestep))
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).cpu().numpy()

    def _setup_np(self, sampling_timesteps=50, noise_guidance=0.25):
        """
        Enhanced diffusion sampling setup with added noise guidance parameter
        for increased expressiveness in the generated motion
        """
        if self.sampling_timesteps == sampling_timesteps:
            return
        
        self.sampling_timesteps = sampling_timesteps
        self.noise_guidance = noise_guidance  # Control parameter for noise characteristics

        total_timesteps = self.n_timestep
        # Variable eta for better stochasticity control (was fixed at 1.0)
        eta = 0.9  # Slightly reduced for more precise expressions
        shape = (1, self.seq_frames, self.motion_feat_dim)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        self.time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        self.time_cond_list = []
        self.alpha_next_sqrt_list = []
        self.sigma_list = []
        self.c_list = []
        self.noise_list = []

        for time, time_next in self.time_pairs:
            time_cond = np.full((1,), time, dtype=np.int64)
            self.time_cond_list.append(time_cond)
            if time_next < 0:
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * np.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            c = np.sqrt(1 - alpha_next - sigma ** 2)
            
            # Generate noise with more expression-focused characteristics
            base_noise = np.random.randn(*shape).astype(np.float32)
            
            # Add bias to facial expression-related dimensions (197-260)
            # This encourages more expressive motion in these dimensions
            expression_noise = base_noise.copy()
            exp_start, exp_end = 197, 260  # Expression dimensions
            
            # Apply expression-biased noise
            expression_bias = np.random.randn(1, 1, exp_end-exp_start) * self.noise_guidance
            expression_noise[:, :, exp_start:exp_end] += expression_bias
            
            # Blend the base noise with expression-biased noise
            noise = base_noise * 0.7 + expression_noise * 0.3
            
            self.alpha_next_sqrt_list.append(np.sqrt(alpha_next))
            self.sigma_list.append(sigma)
            self.c_list.append(c)
            self.noise_list.append(noise)

    def _one_step(self, x, cond_frame, cond, time_cond):
        if self.model_type == "onnx":
            pred = self.model.run(None, {"x": x, "cond_frame": cond_frame, "cond": cond, "time_cond": time_cond})
            pred_noise, x_start = pred[0], pred[1]
        elif self.model_type == "tensorrt":
            self.model.setup({"x": x, "cond_frame": cond_frame, "cond": cond, "time_cond": time_cond})
            self.model.infer()
            pred_noise, x_start = self.model.buffer["pred_noise"][0], self.model.buffer["x_start"][0]
        elif self.model_type == "pytorch":
            with torch.no_grad():
                pred_noise, x_start = self.model(x, cond_frame, cond, time_cond)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return pred_noise, x_start

    def _call_np(self, kp_cond, aud_cond, sampling_timesteps):
        self._setup_np(sampling_timesteps)

        cond_frame = kp_cond
        cond = aud_cond

        x = np.random.randn(1, self.seq_frames, self.motion_feat_dim).astype(np.float32)

        x_start = None
        i = 0
        for _, time_next in self.time_pairs:
            time_cond = self.time_cond_list[i]
            pred_noise, x_start = self._one_step(x, cond_frame, cond, time_cond)
            if time_next < 0:
                x = x_start
                continue

            alpha_next_sqrt = self.alpha_next_sqrt_list[i]
            c = self.c_list[i]
            sigma = self.sigma_list[i]
            noise = self.noise_list[i]
            x = x_start * alpha_next_sqrt + c * pred_noise + sigma * noise

            i += 1

        return x

    def __call__(self, kp_cond, aud_cond, sampling_timesteps):
        if self.model_type == "pytorch":
            pred_kp_seq = self.model.ddim_sample(
                torch.from_numpy(kp_cond).to(self.device), 
                torch.from_numpy(aud_cond).to(self.device), 
                sampling_timesteps,
            ).cpu().numpy()
        else:
            pred_kp_seq = self._call_np(kp_cond, aud_cond, sampling_timesteps)
        return pred_kp_seq


