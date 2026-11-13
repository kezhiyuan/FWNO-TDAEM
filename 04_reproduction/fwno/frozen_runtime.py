from pathlib import Path
import numpy as np
import torch
import HTUFNO_AEM_v6_main as fwno
from package_common import CHECKPOINT, CHECKPOINT_SHA256, sha256

class FrozenFWNO:

    def __init__(self, checkpoint=CHECKPOINT, device="cpu", verify_archived=True):
        self.checkpoint_path = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha256(self.checkpoint_path)
        if verify_archived and self.checkpoint_sha256 != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint does not match the archived epoch-82 weights.")
        self.device = torch.device(device)
        checkpoint = fwno.safe_torch_load(str(self.checkpoint_path), map_location="cpu")
        self.config, self.meta = checkpoint["config"], checkpoint["meta"]
        self.epoch = int(checkpoint["epoch"])
        assert self.config["nz_earth"] == 20 and self.config["input_transform"] == "none"
        assert self.config["target_transform"] == "log_abs"
        self.times_ms = np.asarray(torch.as_tensor(self.meta["times"]), dtype=np.float64)
        self.heights_m = np.asarray(torch.as_tensor(self.meta["heights"]), dtype=np.float64)
        assert self.times_ms.shape == (16,) and self.heights_m.shape == (7,)
        self.x_normalizer = checkpoint["x_normalizer"]
        self.y_normalizer = fwno.move_normalizer(checkpoint["y_normalizer"], self.device)
        self.model = fwno.build_model_from_config(
            self.config, self.meta, self.config["n_out"], self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict_log10(self, raw_log10_sigma):
        raw = np.asarray(raw_log10_sigma, dtype=np.float32)
        if raw.ndim != 4 or raw.shape[1:] != (48, 48, 20):
            raise ValueError("Expected raw log10 conductivity with shape [B,48,48,20].")
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite model values.")

        x = self.x_normalizer.encode(torch.from_numpy(raw[..., None])).to(self.device)
        with fwno.amp_autocast_context(self.config, self.device):
            prediction = self.model(x)
        decoded = fwno.decode_normalized(self.y_normalizer, prediction)
        return decoded.reshape(len(raw), 7, 16).float().cpu().numpy()

    def predict_amplitude(self, raw_log10_sigma):
        return np.power(10.0, self.predict_log10(raw_log10_sigma))
