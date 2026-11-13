import csv
import inspect
import json
import os
import sys
from contextlib import nullcontext
from timeit import default_timer

import numpy as np
import torch
import torch.nn.functional as F
import yaml
try:
    from torchinfo import summary
except ImportError:
    summary = None

from utilities import *
import HTUFNO_AEM_v6 as v5_model_module


HeightCurveNormalizer = v5_model_module.HeightCurveNormalizer


torch.manual_seed(0)
np.random.seed(0)


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def get_path_list(config, key="DATA_PATHS", legacy_prefix="DATA_PATH"):
    if key in config:
        return _as_list(config[key])

    paths = []
    i = 1
    while f"{legacy_prefix}{i}" in config:
        paths.append(config[f"{legacy_prefix}{i}"])
        i += 1


    if not paths and "TRAIN_PATHS" in config:
        paths = _as_list(config["TRAIN_PATHS"])
    if not paths:
        i = 1
        while f"TRAIN_PATH{i}" in config:
            paths.append(config[f"TRAIN_PATH{i}"])
            i += 1
    return paths


def to_1d_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.reshape(-1).float()


def to_tensor_float(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.float()


def to_coordinate_vector(x, name):

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim == 1:
        pass
    elif x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    else:
        raise ValueError(
            f"{name} must be a vector or singleton row/column; got shape {tuple(x.shape)}. "
            "Select the physical coordinate column explicitly instead of flattening it."
        )
    x = x.reshape(-1).float()
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return x


def validate_coordinate_vector(values, name, expected_length=None, positive=False):
    values = to_coordinate_vector(values, name)
    if expected_length is not None and values.numel() != int(expected_length):
        raise ValueError(
            f"{name} length {values.numel()} does not match expected length {int(expected_length)}."
        )
    if positive and torch.any(values <= 0):
        raise ValueError(f"All {name} values must be positive.")
    if values.numel() > 1:
        delta = values[1:] - values[:-1]
        if not torch.all(delta > 0):
            raise ValueError(
                f"{name} must be strictly increasing; got {values.tolist()}. "
                "Reorder the response axes and coordinate vector together before training."
            )
    return values


def try_read_field(reader, keys):
    last_exc = None
    for k in keys:
        if k is None:
            continue
        try:
            return reader.read_field(k), k
        except Exception as exc:
            last_exc = exc
    return None, None


def make_indices(n_total, stride=1, count=None, name="index"):
    idx = torch.arange(0, int(n_total), int(stride), dtype=torch.long)
    if count is not None and int(count) > 0:
        idx = idx[:int(count)]
    if idx.numel() == 0:
        raise ValueError(f"Empty {name} index. Check stride/count settings.")
    return idx


def normalize_log_time(times, logt_min, logt_max):
    times = to_1d_tensor(times)
    if torch.any(times <= 0):
        raise ValueError("All time samples must be positive for log10(t) input.")
    logt = torch.log10(times)
    denom = float(logt_max) - float(logt_min)
    if abs(denom) < 1e-12:
        raise ValueError("logt_max and logt_min are equal; time normalization is undefined.")
    return ((logt - float(logt_min)) / denom).float()


def normalize_height(heights, h_min, h_max):
    h = to_1d_tensor(heights)
    denom = float(h_max) - float(h_min)
    if abs(denom) < 1e-12:
        raise ValueError("height_max and height_min are equal; height normalization is undefined.")
    return ((h - float(h_min)) / denom).float()


def build_query(times, heights, logt_min, logt_max, h_min, h_max):

    tau = normalize_log_time(times, logt_min, logt_max)
    hn = normalize_height(heights, h_min, h_max)
    hh, tt = torch.meshgrid(hn, tau, indexing="ij")
    query = torch.stack([tt.reshape(-1), hh.reshape(-1)], dim=-1).float()
    return query


def read_time_vector(reader, config, n_time_raw):

    source = None
    t_full = None

    if config.get("time_values", None) is not None:
        t_full = to_coordinate_vector(config["time_values"], "config.time_values")
        source = "config.time_values"
    elif config.get("TIMES", None) is not None:
        t_full = to_coordinate_vector(config["TIMES"], "config.TIMES")
        source = "config.TIMES"
    else:

        keys = [config.get("time_key", None), "time", "Time", "times", "TIMES", "t"]
        keys = [k for k in keys if k is not None]
        seen = set()
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            try:
                cand = to_coordinate_vector(reader.read_field(k), f"MAT time variable {k!r}")
            except Exception:
                continue
            if cand.numel() >= int(n_time_raw):
                t_full = cand
                source = k
                break
            else:
                print(
                    f"Warning: MAT variable '{k}' has length {cand.numel()}, "
                    f"but response time dimension is {n_time_raw}; skip it as time vector."
                )

        if t_full is None:
            if bool(config.get("allow_time_index_fallback", False)):
                t_full = torch.arange(1, int(n_time_raw) + 1, dtype=torch.float32)
                source = "fallback_index_1_to_T"
                print("Warning: no valid time vector found. Using 1..T as pseudo-time for debugging.")
            else:
                raise KeyError(
                    f"Cannot find a valid time vector with at least {int(n_time_raw)} values. "
                    "Set time_key/time_values/TIMES correctly, or set allow_time_index_fallback: true for debugging."
                )

    if t_full.numel() < int(n_time_raw):
        raise ValueError(
            f"Time vector from {source} has length {t_full.numel()}, "
            f"but response time dimension is {n_time_raw}."
        )

    r_time = int(config.get("r_time", config.get("r", config.get("r_train", [1, 1, 1, 1]))[3]))
    n_time = config.get("n_time", config.get("s_time", config.get("s", config.get("s_train", [47, 47, 20, n_time_raw]))[3]))
    idx = make_indices(n_time_raw, stride=r_time, count=n_time, name="time")
    times = validate_coordinate_vector(
        t_full[idx], "selected time", expected_length=len(idx), positive=True
    )
    print(f"Using time vector from {source}; selected n_time={len(times)}")
    return times, idx


def read_height_vector(reader, config, n_height_raw):

    if config.get("height_values", None) is not None:
        h_full = to_coordinate_vector(config["height_values"], "config.height_values")
        source = "config.height_values"
    elif config.get("HEIGHTS", None) is not None:
        h_full = to_coordinate_vector(config["HEIGHTS"], "config.HEIGHTS")
        source = "config.HEIGHTS"
    else:
        keys = [config.get("height_key", None), "obs", "OBS", "height", "Height", "heights", "Heights", "HEIGHTS", "H", "h", "altitude", "Altitude", "flight_height"]
        h_raw, source = try_read_field(reader, keys)
        if h_raw is None:
            raise KeyError("Cannot find height vector. Set height_values/HEIGHTS in config or set height_key.")
        h_full = to_coordinate_vector(h_raw, f"MAT height variable {source!r}")

    if h_full.numel() < int(n_height_raw):
        raise ValueError(
            f"Height vector from {source} has length {h_full.numel()}, "
            f"but response height dimension is {n_height_raw}."
        )


    if config.get("keep_height_indices", None) is not None:
        idx = torch.tensor(config["keep_height_indices"], dtype=torch.long).reshape(-1)
        if int(config.get("height_indices_base", 0)) == 1:
            idx = idx - 1
        if torch.any(idx < 0) or torch.any(idx >= h_full.numel()):
            raise IndexError("keep_height_indices contains out-of-range indices.")
    elif config.get("keep_heights", None) is not None:
        keep = to_1d_tensor(config["keep_heights"])
        idx_list = []
        tol = float(config.get("height_match_tol", 1e-5))
        for h in keep:
            diff = torch.abs(h_full - h)
            j = int(torch.argmin(diff))
            if float(diff[j]) > tol:
                print(f"Warning: requested height {float(h):.6g}; using nearest {float(h_full[j]):.6g}.")
            idx_list.append(j)
        idx = torch.tensor(idx_list, dtype=torch.long)
    else:
        r_height = int(config.get("r_height", 1))
        s_height = config.get("s_height", None)
        idx = make_indices(n_height_raw, stride=r_height, count=s_height, name="height")

    heights = validate_coordinate_vector(
        h_full[idx], "selected height", expected_length=len(idx), positive=False
    )
    print(f"Using height vector from {source}; selected heights={heights.tolist()}")
    return heights, idx


def transform_input_model(x, input_transform="none"):
    input_transform = str(input_transform).lower()
    if input_transform in ["none", "identity"]:
        return x
    if input_transform in ["log10", "log_conductivity", "conductivity_log10"]:
        if torch.any(x <= 0):
            raise ValueError(
                "input_transform='log10' requires strictly positive conductivity values."
            )
        return torch.log10(x)
    if input_transform in ["negate", "logrho_to_logsigma", "log_resistivity_to_log_conductivity"]:
        return -x
    if input_transform == "abs":
        return torch.abs(x)
    raise ValueError(
        "input_transform must be 'none', 'log10', 'negate'/'logrho_to_logsigma', or 'abs'."
    )


def transform_target(y, target_transform="log_abs", log_eps=1e-30, signed_scale=1.0):

    target_transform = str(target_transform).lower()
    if target_transform in ["none", "identity"]:
        return y
    if target_transform == "abs":
        return torch.abs(y)
    if target_transform == "log_abs":
        return torch.log10(torch.clamp(torch.abs(y), min=float(log_eps)))
    if target_transform == "log":
        return torch.log10(torch.clamp(y, min=float(log_eps)))
    if target_transform == "signed_log":
        return torch.sign(y) * torch.log10(1.0 + torch.abs(y) / float(signed_scale))
    raise ValueError("target_transform must be none, abs, log_abs, log, or signed_log.")


def read_models_from_reader(reader, config):
    sig_key = config.get("sig_key", "sig")
    r = config.get("r", config.get("r_train", [1, 1, 1, 1]))
    s = config.get("s", config.get("s_train", [47, 47, 20, 34]))
    r_xyz = r[:3]
    s_xyz = s[:3]
    nz_earth = config.get("nz_earth", None)
    input_transform = config.get("input_transform", "none")
    n_model_per_file = config.get("n_model_per_file", config.get("n_model", None))

    x = reader.read_field(sig_key)
    x = to_tensor_float(x)
    n_avail = int(x.shape[0])
    n_use = n_avail if n_model_per_file is None or int(n_model_per_file) <= 0 else min(int(n_model_per_file), n_avail)

    x = x[:n_use, ::r_xyz[0], ::r_xyz[1], ::r_xyz[2]]
    x = x[:, :s_xyz[0], :s_xyz[1], :s_xyz[2]]
    if nz_earth is not None:
        x = x[:, :, :, :int(nz_earth)]
    x = transform_input_model(x.float(), input_transform=input_transform)
    x = x.reshape(n_use, x.shape[1], x.shape[2], x.shape[3], 1).float()
    return x, n_use


def _choose_center_loc(n_loc, center_loc_index=None, obs=None):
    if center_loc_index is not None:
        center_loc_index = int(center_loc_index)
        if center_loc_index < 0:
            center_loc_index = n_loc + center_loc_index
        if center_loc_index < 0 or center_loc_index >= n_loc:
            raise IndexError(f"center_loc_index={center_loc_index} is outside [0, {n_loc - 1}]")
        return center_loc_index
    if obs is not None:
        obs = torch.tensor(obs).float()
        if obs.ndim >= 2 and obs.shape[0] == n_loc and obs.shape[-1] >= 2:
            dist = obs[:, 0] ** 2 + obs[:, 1] ** 2
            return int(torch.argmin(dist))
        if obs.ndim == 1 and obs.numel() == n_loc:
            return int(torch.argmin(torch.abs(obs)))
    return int(n_loc // 2)


def infer_response_layout(y, n_model, n_time_raw=None, n_height_raw=None):

    shape = tuple(y.shape)
    if y.dim() == 4:
        if shape[0] >= n_model and shape[-1] == 1:
            if n_time_raw is not None and n_height_raw is not None:
                if shape[1] == n_time_raw and shape[2] == n_height_raw:
                    return "model_time_height_component"
                if shape[1] == n_height_raw and shape[2] == n_time_raw:
                    return "model_height_time_component"

            if shape[1] > shape[2]:
                return "model_time_height_component"
            return "model_height_time_component"
        if shape[0] != n_model and shape[0] > 1:

            if n_time_raw is not None and n_height_raw is not None:
                if shape[0] == n_time_raw and shape[1] == n_height_raw:
                    return "time_height_loc_model"
                if shape[0] == n_height_raw and shape[1] == n_time_raw:
                    return "height_time_loc_model"
    elif y.dim() == 3:
        if shape[0] >= n_model:
            if n_time_raw is not None and n_height_raw is not None:
                if shape[1] == n_time_raw and shape[2] == n_height_raw:
                    return "model_time_height"
                if shape[1] == n_height_raw and shape[2] == n_time_raw:
                    return "model_height_time"
            if shape[1] > shape[2]:
                return "model_time_height"
            return "model_height_time"
        if n_time_raw is not None and n_height_raw is not None:
            if shape[0] == n_time_raw and shape[1] == n_height_raw:
                return "time_height_model"
            if shape[0] == n_height_raw and shape[1] == n_time_raw:
                return "height_time_model"
    raise ValueError(f"Cannot infer response_layout for response shape {shape}. Please set response_layout.")


def read_response_one_key(reader, response_key, n_model, time_indices, height_indices, config,
                          n_time_raw=None, n_height_raw=None):

    y = reader.read_field(response_key)
    y = to_tensor_float(y)
    response_layout = str(config.get("response_layout", "auto")).lower()
    if response_layout == "auto":
        response_layout = infer_response_layout(y, n_model, n_time_raw=n_time_raw, n_height_raw=n_height_raw)
        print(f"Auto inferred response_layout='{response_layout}' for {response_key}.shape={tuple(y.shape)}")

    t_idx = time_indices.to(dtype=torch.long)
    h_idx = height_indices.to(dtype=torch.long)
    m_idx = torch.arange(int(n_model), dtype=torch.long)

    if response_layout == "model_time_height_component":

        y = y.index_select(0, m_idx).index_select(1, t_idx).index_select(2, h_idx)
        if y.shape[-1] != 1:
            raise ValueError(f"Expected singleton component axis, got shape {tuple(y.shape)}")
        y = y[..., 0].permute(0, 2, 1).contiguous()

    elif response_layout == "model_height_time_component":

        y = y.index_select(0, m_idx).index_select(1, h_idx).index_select(2, t_idx)
        if y.shape[-1] != 1:
            raise ValueError(f"Expected singleton component axis, got shape {tuple(y.shape)}")
        y = y[..., 0].contiguous()

    elif response_layout == "model_time_height":

        y = y.index_select(0, m_idx).index_select(1, t_idx).index_select(2, h_idx)
        y = y.permute(0, 2, 1).contiguous()

    elif response_layout == "model_height_time":

        y = y.index_select(0, m_idx).index_select(1, h_idx).index_select(2, t_idx).contiguous()

    elif response_layout == "time_height_model":

        y = y.index_select(0, t_idx).index_select(1, h_idx).index_select(2, m_idx)
        y = y.permute(2, 1, 0).contiguous()

    elif response_layout == "height_time_model":

        y = y.index_select(0, h_idx).index_select(1, t_idx).index_select(2, m_idx)
        y = y.permute(2, 0, 1).contiguous()

    elif response_layout in ["time_height_loc_model", "time_loc_height_model", "height_time_loc_model", "model_height_time_loc"]:
        obs = None
        try:
            obs = reader.read_field(config.get("obs_key", "obs"))
        except Exception:
            obs = None

        if response_layout == "time_height_loc_model":

            y = y.index_select(0, t_idx).index_select(1, h_idx).index_select(3, m_idx)
            loc_idx = _choose_center_loc(y.shape[2], config.get("center_loc_index", None), obs=obs)
            y = y[:, :, loc_idx, :].permute(2, 1, 0).contiguous()
        elif response_layout == "time_loc_height_model":

            y = y.index_select(0, t_idx).index_select(2, h_idx).index_select(3, m_idx)
            loc_idx = _choose_center_loc(y.shape[1], config.get("center_loc_index", None), obs=obs)
            y = y[:, loc_idx, :, :].permute(2, 1, 0).contiguous()
        elif response_layout == "height_time_loc_model":

            y = y.index_select(0, h_idx).index_select(1, t_idx).index_select(3, m_idx)
            loc_idx = _choose_center_loc(y.shape[2], config.get("center_loc_index", None), obs=obs)
            y = y[:, :, loc_idx, :].permute(2, 0, 1).contiguous()
        else:

            y = y.index_select(0, m_idx).index_select(1, h_idx).index_select(2, t_idx)
            loc_idx = _choose_center_loc(y.shape[3], config.get("center_loc_index", None), obs=obs)
            y = y[:, :, :, loc_idx].contiguous()
    else:
        raise ValueError(
            f"Unsupported response_layout='{response_layout}'. Use auto or one of the documented model/time/height layouts."
        )

    if y.shape != (int(n_model), int(len(height_indices)), int(len(time_indices))):
        raise RuntimeError(
            f"Internal response shape error: got {tuple(y.shape)}, "
            f"expected {(int(n_model), int(len(height_indices)), int(len(time_indices)))}"
        )
    return y.float()


def load_all_models_from_multiheight_mats(config, n_out):
    paths = get_path_list(config)
    if len(paths) == 0:
        raise ValueError("Please set DATA_PATHS in config.")

    response_keys = list(config.get("response_keys", ["dBzdt"]))[:int(n_out)]
    if len(response_keys) == 0:
        raise ValueError("response_keys cannot be empty.")

    x_list, y_list = [], []
    times_ref, heights_ref = None, None
    time_indices_ref, height_indices_ref = None, None

    for ip, path in enumerate(paths):
        print(f"\nReading file {ip + 1}/{len(paths)}: {path}")
        reader = MatReader(path)
        x, n_use = read_models_from_reader(reader, config)


        y0 = to_tensor_float(reader.read_field(response_keys[0]))
        layout_hint = str(config.get("response_layout", "auto")).lower()
        detected_layout = None



        h_cfg = config.get("height_values", config.get("HEIGHTS", None))
        if h_cfg is not None:
            h_cfg_len = len(h_cfg)
        else:
            h_cfg_len = None
            h_key_candidates = [config.get("height_key", None), "obs", "OBS", "height", "Height", "heights", "Heights"]
            seen_h = set()
            for hk in h_key_candidates:
                if hk is None or hk in seen_h:
                    continue
                seen_h.add(hk)
                try:
                    h_tmp = to_coordinate_vector(
                        reader.read_field(hk), f"MAT height variable {hk!r}"
                    )
                    h_cfg_len = int(h_tmp.numel())
                    print(f"Detected height vector candidate '{hk}' length = {h_cfg_len}")
                    break
                except Exception:
                    pass

        s_cfg = config.get("s", config.get("s_train", [47, 47, 20, None]))
        t_cfg_len = int(s_cfg[3]) if len(s_cfg) > 3 and s_cfg[3] is not None else None
        if t_cfg_len is None:
            t_key_candidates = [config.get("time_key", None), "time", "Time", "times", "TIMES", "t"]
            seen_t = set()
            for tk in t_key_candidates:
                if tk is None or tk in seen_t:
                    continue
                seen_t.add(tk)
                try:
                    t_tmp = to_coordinate_vector(
                        reader.read_field(tk), f"MAT time variable {tk!r}"
                    )
                    t_cfg_len = int(t_tmp.numel())
                    print(f"Detected time vector candidate '{tk}' length = {t_cfg_len}")
                    break
                except Exception:
                    pass


        if bool(config.get("force_layout_adapter", True)):
            if h_cfg_len is not None and t_cfg_len is not None:
                shape3 = tuple(int(v) for v in y0.shape[:3])
                if y0.dim() == 3:
                    candidates = {
                        (int(n_use), int(h_cfg_len), int(t_cfg_len)): "model_height_time",
                        (int(n_use), int(t_cfg_len), int(h_cfg_len)): "model_time_height",
                        (int(t_cfg_len), int(h_cfg_len), int(n_use)): "time_height_model",
                        (int(h_cfg_len), int(t_cfg_len), int(n_use)): "height_time_model",
                    }
                    detected_layout = candidates.get(shape3)
                elif y0.dim() == 4 and int(y0.shape[-1]) == 1:
                    candidates = {
                        (int(n_use), int(h_cfg_len), int(t_cfg_len)): "model_height_time_component",
                        (int(n_use), int(t_cfg_len), int(h_cfg_len)): "model_time_height_component",
                    }
                    detected_layout = candidates.get(shape3)

                if detected_layout is not None:
                    if detected_layout in ["model_height_time", "model_height_time_component"]:
                        n_height_raw, n_time_raw = int(y0.shape[1]), int(y0.shape[2])
                    elif detected_layout in ["model_time_height", "model_time_height_component"]:
                        n_time_raw, n_height_raw = int(y0.shape[1]), int(y0.shape[2])
                    elif detected_layout == "time_height_model":
                        n_time_raw, n_height_raw = int(y0.shape[0]), int(y0.shape[1])
                    elif detected_layout == "height_time_model":
                        n_height_raw, n_time_raw = int(y0.shape[0]), int(y0.shape[1])
                    print(
                        "Axis-role adapter matched response layout "
                        f"'{detected_layout}' from shape={tuple(y0.shape)}, "
                        f"n_model={n_use}, n_height={h_cfg_len}, n_time={t_cfg_len}"
                    )

        if detected_layout is None:
            if layout_hint == "model_time_height_component":
                n_time_raw, n_height_raw = int(y0.shape[1]), int(y0.shape[2])
                detected_layout = layout_hint
            elif layout_hint == "model_height_time_component":
                n_height_raw, n_time_raw = int(y0.shape[1]), int(y0.shape[2])
                detected_layout = layout_hint
            elif layout_hint == "model_time_height":
                n_time_raw, n_height_raw = int(y0.shape[1]), int(y0.shape[2])
                detected_layout = layout_hint
            elif layout_hint == "model_height_time":
                n_height_raw, n_time_raw = int(y0.shape[1]), int(y0.shape[2])
                detected_layout = layout_hint
            elif layout_hint == "time_height_model":
                n_time_raw, n_height_raw = int(y0.shape[0]), int(y0.shape[1])
                detected_layout = layout_hint
            elif layout_hint == "height_time_model":
                n_height_raw, n_time_raw = int(y0.shape[0]), int(y0.shape[1])
                detected_layout = layout_hint
            else:
                detected_layout = infer_response_layout(y0, n_use)
                print(f"Initial auto layout guess for raw dimensions: {detected_layout}")
                if detected_layout in ["model_time_height_component", "model_time_height"]:
                    n_time_raw, n_height_raw = int(y0.shape[1]), int(y0.shape[2])
                elif detected_layout in ["model_height_time_component", "model_height_time"]:
                    n_height_raw, n_time_raw = int(y0.shape[1]), int(y0.shape[2])
                elif detected_layout == "time_height_model":
                    n_time_raw, n_height_raw = int(y0.shape[0]), int(y0.shape[1])
                elif detected_layout == "height_time_model":
                    n_height_raw, n_time_raw = int(y0.shape[0]), int(y0.shape[1])
                else:
                    n_time_raw, n_height_raw = int(y0.shape[1]), int(y0.shape[2])

        print(
            f"Detected raw response shape {tuple(y0.shape)} as layout='{detected_layout}', "
            f"n_time_raw={n_time_raw}, n_height_raw={n_height_raw}"
        )
        config_for_response = dict(config)
        config_for_response["response_layout"] = detected_layout

        times, time_indices = read_time_vector(reader, config, n_time_raw=n_time_raw)
        heights, height_indices = read_height_vector(reader, config, n_height_raw=n_height_raw)

        if times_ref is None:
            times_ref = times
            heights_ref = heights
            time_indices_ref = time_indices
            height_indices_ref = height_indices
        else:
            if times.shape != times_ref.shape or not torch.allclose(times, times_ref, rtol=0, atol=1e-8):
                raise ValueError("Selected time samples are not identical across .mat files.")
            if heights.shape != heights_ref.shape or not torch.allclose(heights, heights_ref, rtol=0, atol=1e-6):
                raise ValueError("Selected flight heights are not identical across .mat files.")

        y_channels = []
        for key in response_keys:
            y_one = read_response_one_key(
                reader, key, n_use, time_indices_ref, height_indices_ref, config_for_response,
                n_time_raw=n_time_raw, n_height_raw=n_height_raw,
            )
            y_channels.append(y_one)

        y = torch.stack(y_channels, dim=-1)
        y = y.reshape(n_use, len(heights_ref) * len(times_ref), 1, len(response_keys))
        y = transform_target(
            y,
            target_transform=config.get("target_transform", "log_abs"),
            log_eps=float(config.get("log_eps", 1e-30)),
            signed_scale=float(config.get("signed_scale", 1.0)),
        )

        print(f"x.shape after read = {tuple(x.shape)}")
        print(f"y.shape after reshape = {tuple(y.shape)}  # [M,H*T,1,nout]")
        x_list.append(x.float())
        y_list.append(y.float())

    x_all = torch.cat(x_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    return x_all, y_all, times_ref.float(), heights_ref.float()


def split_train_val_test(
    x_all,
    y_all,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=1234,
):
    n_total = int(x_all.shape[0])
    ratios = np.asarray(
        [float(train_ratio), float(val_ratio), float(test_ratio)], dtype=np.float64
    )
    if np.any(ratios <= 0.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must all be positive.")
    if not np.isclose(float(ratios.sum()), 1.0, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"train/val/test ratios must sum to 1, got {ratios.tolist()}.")
    if n_total < 3:
        raise ValueError("At least three models are required for train/val/test splitting.")
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=gen)
    ntrain = int(round(train_ratio * n_total))
    nval = int(round(val_ratio * n_total))
    ntrain = min(max(ntrain, 1), n_total - 2)
    nval = min(max(nval, 1), n_total - ntrain - 1)
    train_idx = perm[:ntrain]
    val_idx = perm[ntrain:ntrain + nval]
    test_idx = perm[ntrain + nval:]
    return (
        x_all[train_idx], y_all[train_idx],
        x_all[val_idx], y_all[val_idx],
        x_all[test_idx], y_all[test_idx],
        train_idx, val_idx, test_idx,
    )


def make_compatible_normalizer(normalizer_cls, data, name="normalizer", preferred_flag="Point"):

    flag = str(preferred_flag)
    if flag not in ["Point", "Unit"]:
        print(f"Warning: {name} flag={flag!r} is invalid; use 'Point' instead.")
        flag = "Point"

    try:
        print(f"Using {name}: {normalizer_cls.__name__}(data, flag={flag!r})")
        return normalizer_cls(data, flag=flag)
    except TypeError:

        print(f"Using {name}: {normalizer_cls.__name__}(data, {flag!r})")
        return normalizer_cls(data, flag)

def get_batch_data(config, batch_size, n_out):
    print("===============================")
    print("Start to read data for HT-UFNO-AEM-v2 training")
    print("===============================")
    t_read0 = default_timer()

    x_all, y_all, times, heights = load_all_models_from_multiheight_mats(config, n_out)
    print("\nCombined data:")
    print("x_all.shape =", tuple(x_all.shape))
    print("y_all.shape =", tuple(y_all.shape), "# [N,H*T,1,nout]")

    split_ratios = config.get('split_ratios', None)
    if split_ratios is None:
        split_ratios = [
            float(config.get('train_ratio', 0.8)),
            float(config.get('val_ratio', 0.1)),
            float(config.get('test_ratio', 0.1)),
        ]
    if len(split_ratios) != 3:
        raise ValueError('split_ratios must contain [train, val, test].')
    (
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        train_idx, val_idx, test_idx,
    ) = split_train_val_test(
        x_all, y_all,
        train_ratio=float(split_ratios[0]),
        val_ratio=float(split_ratios[1]),
        test_ratio=float(split_ratios[2]),
        seed=int(config.get("split_seed", 1234)),
    )
    config["ntrain"] = int(x_train.shape[0])
    config["nval"] = int(x_val.shape[0])
    config["ntest"] = int(x_test.shape[0])

    print(
        "\nSplit:", f"ntrain={x_train.shape[0]}",
        f"nval={x_val.shape[0]}", f"ntest={x_test.shape[0]}"
    )
    print("x_train.shape =", tuple(x_train.shape))
    print("y_train.shape =", tuple(y_train.shape))
    print("x_val.shape   =", tuple(x_val.shape))
    print("y_val.shape   =", tuple(y_val.shape))
    print("x_test.shape  =", tuple(x_test.shape))
    print("y_test.shape  =", tuple(y_test.shape))

    logt_min = float(config.get("logt_min", torch.log10(times).min().item()))
    logt_max = float(config.get("logt_max", torch.log10(times).max().item()))
    h_min = float(config.get("height_min", heights.min().item()))
    h_max = float(config.get("height_max", heights.max().item()))
    query = build_query(times, heights, logt_min, logt_max, h_min, h_max)
    print("query.shape  =", tuple(query.shape), "# [H*T,2]")

    x_normalizer = make_compatible_normalizer(GaussianNormalizer, x_train, name='x_normalizer', preferred_flag=config.get('x_normalizer_flag', 'Point'))
    x_train = x_normalizer.encode(x_train)
    x_val = x_normalizer.encode(x_val)
    x_test = x_normalizer.encode(x_test)

    y_normalizer_type = str(config.get('y_normalizer_type', 'height_curve')).lower()
    if y_normalizer_type in ['height_curve', 'curve', 'heightcurve']:
        print('Using y_normalizer: HeightCurveNormalizer(mean curve per height-time, one scale per height)')
        y_normalizer = HeightCurveNormalizer(
            y_train, n_height=int(len(heights)), n_time=int(len(times)),
            eps=float(config.get('y_normalizer_eps', 1.0e-6)),
        )
    elif y_normalizer_type in ['point', 'gaussian_point']:
        y_normalizer = make_compatible_normalizer(
            GaussianNormalizer_out, y_train, name='y_normalizer',
            preferred_flag=config.get('y_normalizer_flag', 'Point')
        )
    else:
        raise ValueError(
            f"Unsupported y_normalizer_type={y_normalizer_type!r}; use 'height_curve' or 'point'."
        )
    y_train = y_normalizer.encode(y_train)
    y_val = y_normalizer.encode(y_val)
    y_test = y_normalizer.encode(y_test)

    eval_batch_size = int(config.get('eval_batch_size', batch_size))
    if eval_batch_size <= 0:
        raise ValueError('eval_batch_size must be positive.')
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=int(batch_size), shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val),
        batch_size=eval_batch_size, shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=eval_batch_size, shuffle=False,
    )

    meta = {
        "logt_min": logt_min,
        "logt_max": logt_max,
        "height_min": h_min,
        "height_max": h_max,
        "times": times,
        "heights": heights,
        "n_height": int(len(heights)),
        "n_time": int(len(times)),
        "target_transform": config.get("target_transform", "log_abs"),
        "log_eps": float(config.get("log_eps", 1e-30)),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "split_seed": int(config.get("split_seed", 1234)),
    }

    t_read1 = default_timer()
    print("===============================")
    print(f"all data has been read in {t_read1 - t_read0:.3f} seconds")
    print("===============================")
    return (
        train_loader, val_loader, test_loader, query,
        x_normalizer, y_normalizer, meta,
    )


import copy
import math
from torch.optim import AdamW


def print_model(model, flag=True):
    if flag:
        print(model)


def move_normalizer(normalizer, device):
    if hasattr(normalizer, 'to'):
        normalizer.to(device)
    elif hasattr(normalizer, 'mean') and hasattr(normalizer, 'std'):
        normalizer.mean = normalizer.mean.to(device)
        normalizer.std = normalizer.std.to(device)
    return normalizer


def decode_normalized(normalizer, y):

    return normalizer.decode(y.clone())


def _parse_float_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace('[', '').replace(']', '')
        return [float(v.strip()) for v in value.split(',') if v.strip()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def make_height_time_loss_weights(config, meta, device):
    n_height, n_time = int(meta['n_height']), int(meta['n_time'])
    w = torch.ones(n_height, n_time, dtype=torch.float32)
    if not bool(config.get('use_height_time_weight', True)):
        return w.to(device)

    h_list = _parse_float_list(config.get('height_loss_weights', None))
    if h_list is None:
        h_list = torch.linspace(
            float(config.get('low_height_weight', 1.3)),
            float(config.get('high_height_weight', 1.0)),
            n_height,
        ).tolist()
    if len(h_list) != n_height:
        raise ValueError(f'height_loss_weights length {len(h_list)} != n_height {n_height}')
    h_w = torch.tensor(h_list, dtype=torch.float32).reshape(n_height, 1)

    t_list = _parse_float_list(config.get('time_loss_weights', None))
    if t_list is not None:
        if len(t_list) != n_time:
            raise ValueError(f'time_loss_weights length {len(t_list)} != n_time {n_time}')
        t_w = torch.tensor(t_list, dtype=torch.float32).reshape(1, n_time)
    else:
        t_w = torch.ones(1, n_time, dtype=torch.float32)
        mid_start = int(config.get('time_mid_start', max(0, n_time // 3)))
        late_start = int(config.get('time_late_start', max(0, n_time - 8)))
        if mid_start < n_time:
            t_w[:, mid_start:] = float(config.get('time_mid_weight', 1.0))
        if late_start < n_time:
            t_w[:, late_start:] = float(config.get('time_late_weight', 1.2))

    w = h_w * t_w
    if bool(config.get('normalize_loss_weights', True)):
        w = w / torch.clamp(w.mean(), min=1.0e-12)
    return w.to(device)


def _reshape_ht(y, n_height, n_time):
    return y.reshape(y.shape[0], int(n_height), int(n_time), 1, y.shape[-1])


def _weighted_mean(elem, weight_ht=None):
    if weight_ht is None:
        return elem.mean()

    w = weight_ht
    if elem.shape[2] == w.shape[1] - 1:
        w = 0.5 * (w[:, :-1] + w[:, 1:])
    elif elem.shape[2] == w.shape[1] - 2:
        w = (w[:, :-2] + w[:, 1:-1] + w[:, 2:]) / 3.0
    w = w.reshape(1, w.shape[0], w.shape[1], 1, 1).to(device=elem.device, dtype=elem.dtype)
    return (elem * w).mean()


def _robust_zero_loss(err, kind='smoothl1', beta=0.02):
    kind = str(kind).lower()
    if kind == 'l1':
        return torch.abs(err)
    if kind == 'mse':
        return err * err
    return F.smooth_l1_loss(err, torch.zeros_like(err), reduction='none', beta=float(beta))


def make_time_coordinate(times, transform='log10', device='cpu'):
    t = torch.as_tensor(times, dtype=torch.float32, device=device).reshape(-1)
    transform = str(transform).lower()
    if transform in ['log10', 'log']:
        if torch.any(t <= 0):
            raise ValueError('Positive times are required for log-time slope loss.')
        u = torch.log10(t)
    elif transform in ['ln', 'loge']:
        if torch.any(t <= 0):
            raise ValueError('Positive times are required for log-time slope loss.')
        u = torch.log(t)
    elif transform in ['linear', 'none', 'identity']:
        u = t
    else:
        raise ValueError(f'Unsupported slope_time_transform={transform!r}')
    u = (u - u.min()) / torch.clamp(u.max() - u.min(), min=1.0e-12)
    return u


def stage_loss_weights(config, stage):
    stage = int(stage)
    if stage not in [1, 2, 3]:
        raise ValueError(f'Unsupported training stage: {stage}')
    prefix = f'stage{stage}_'
    defaults_by_stage = {
        1: {
            'point_log_weight': 1.0,
            'relative_weight': 0.75,
            'amplitude_weight': 0.2,
            'slope_weight': 0.02,
            'curvature_weight': 0.005,
            'threshold5_weight': 0.5,
            'threshold10_weight': 0.2,
            'cvar_weight': 0.1,
            'height_consistency_weight': 0.005,
            'monotonic_weight': 0.0,
        },
        2: {
            'point_log_weight': 1.0,
            'relative_weight': 1.0,
            'amplitude_weight': 0.15,
            'slope_weight': 0.02,
            'curvature_weight': 0.0,
            'threshold5_weight': 1.0,
            'threshold10_weight': 0.25,
            'cvar_weight': 0.15,
            'height_consistency_weight': 0.005,
            'monotonic_weight': 0.0,
        },
        3: {
            'point_log_weight': 1.0,
            'relative_weight': 1.0,
            'amplitude_weight': 0.1,
            'slope_weight': 0.02,
            'curvature_weight': 0.0,
            'threshold5_weight': 1.0,
            'threshold10_weight': 0.25,
            'cvar_weight': 0.2,
            'height_consistency_weight': 0.005,
            'monotonic_weight': 0.0,
        },
    }
    defaults = defaults_by_stage[stage]
    result = {
        key: float(config.get(prefix + key, config.get(key, default)))
        for key, default in defaults.items()
    }

    result['cvar_weight'] = float(config.get(
        prefix + 'tail_weight',
        config.get(
            prefix + 'cvar_weight',
            config.get('tail_weight', config.get('cvar_weight', result['cvar_weight'])),
        ),
    ))
    return result


def raw_relative_error_from_log_error(log_error):

    error64 = log_error.detach().double()
    return torch.abs(torch.expm1(math.log(10.0) * error64))


def smooth_training_relative_error(log_error, config):

    clip = float(config.get(
        'training_relative_log_error_clip',
        config.get('relative_log_error_clip', 0.5),
    ))
    if clip <= 0.0:
        robust_error = log_error
    else:
        robust_error = clip * torch.tanh(log_error / clip)
    return torch.abs(torch.expm1(math.log(10.0) * robust_error))


def asymmetric_log_threshold_violation(log_error, tau):

    tau = float(tau)
    if not 0.0 < tau < 1.0:
        raise ValueError(f'tau must lie in (0, 1), got {tau}.')
    lower = math.log10(1.0 - tau)
    upper = math.log10(1.0 + tau)
    return (
        torch.relu(log_error - upper) / upper
        + torch.relu(lower - log_error) / (-lower)
    )


def asymmetric_log_threshold_loss(log_error, tau, weight_ht, config):

    violation = asymmetric_log_threshold_violation(log_error, tau)
    configured_power = config.get('threshold_loss_power', None)
    if configured_power is not None:
        power = float(configured_power)
        if power <= 0.0:
            raise ValueError(f'threshold_loss_power must be positive, got {power}.')
        elem = violation.pow(power)
    else:
        elem = _robust_zero_loss(
            violation,
            kind=config.get('threshold_loss_type', 'smoothl1'),
            beta=float(config.get('threshold_smoothl1_beta', 0.25)),
        )
    threshold_weight = None if bool(
        config.get('threshold_loss_unweighted', True)
    ) else weight_ht
    return _weighted_mean(elem, threshold_weight)


def cvar_top_tail_loss(tail_score, config):

    fraction = float(config.get(
        'tail_fraction', config.get('cvar_fraction', config.get('top_tail_fraction', 0.10))
    ))
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f'cvar_fraction must lie in (0, 1], got {fraction}.')
    flat = tail_score.reshape(tail_score.shape[0], -1)
    k = max(1, int(math.ceil(fraction * flat.shape[1])))
    return torch.topk(flat, k=k, dim=1, largest=True, sorted=False).values.mean()


def physical_loss(
    pred_norm,
    true_norm,
    y_normalizer,
    times,
    heights,
    weight_ht,
    config,
    stage=1,
):

    pred_log = decode_normalized(y_normalizer, pred_norm)
    true_log = decode_normalized(y_normalizer, true_norm)
    h, t = int(len(heights)), int(len(times))
    p = _reshape_ht(pred_log, h, t)
    y = _reshape_ht(true_log, h, t)
    err = p - y

    weights = stage_loss_weights(config, stage)
    loss_kind = config.get('physical_point_loss_type', 'smoothl1')
    beta = float(config.get('physical_smoothl1_beta', 0.02))

    point_elem = _robust_zero_loss(err, kind=loss_kind, beta=beta)
    point = _weighted_mean(point_elem, weight_ht)


    rel_train = smooth_training_relative_error(err, config)
    rel_beta = float(config.get('relative_smoothl1_beta', 0.05))
    relative = _weighted_mean(
        F.smooth_l1_loss(
            rel_train, torch.zeros_like(rel_train), reduction='none', beta=rel_beta
        ),
        weight_ht,
    )

    violation5 = asymmetric_log_threshold_violation(err, 0.05)
    threshold5 = asymmetric_log_threshold_loss(err, 0.05, weight_ht, config)
    threshold10 = asymmetric_log_threshold_loss(err, 0.10, weight_ht, config)
    tail_source = str(config.get('tail_source', 'threshold5_violation')).lower()
    tail_score = rel_train if tail_source in ['relative', 'smooth_relative'] else violation5
    cvar = cvar_top_tail_loss(tail_score, config)

    anchor_gates = max(1, min(int(config.get('amplitude_anchor_gates', 3)), t))
    amp_err = p[:, :, :anchor_gates].mean(dim=2, keepdim=True) - y[:, :, :anchor_gates].mean(dim=2, keepdim=True)
    h_w = weight_ht[:, :anchor_gates].mean(dim=1).reshape(1, h, 1, 1, 1)
    amplitude = (_robust_zero_loss(amp_err, kind=loss_kind, beta=beta) * h_w).mean()

    u = make_time_coordinate(times, config.get('slope_time_transform', 'log10'), device=pred_norm.device)
    dt = u[1:] - u[:-1]
    dt_floor = float(config.get('slope_dt_floor_ratio', 0.25)) * torch.clamp(dt.mean(), min=1.0e-6)
    dt_eff = torch.clamp(dt, min=dt_floor).reshape(1, 1, -1, 1, 1)
    p_slope = (p[:, :, 1:] - p[:, :, :-1]) / dt_eff
    y_slope = (y[:, :, 1:] - y[:, :, :-1]) / dt_eff
    slope = _weighted_mean(
        _robust_zero_loss(p_slope - y_slope, kind='smoothl1', beta=float(config.get('slope_smoothl1_beta', 0.1))),
        weight_ht,
    )

    if t >= 3:
        du = 0.5 * (dt_eff[:, :, 1:] + dt_eff[:, :, :-1])
        p_curv = (p_slope[:, :, 1:] - p_slope[:, :, :-1]) / du
        y_curv = (y_slope[:, :, 1:] - y_slope[:, :, :-1]) / du
        curvature = _weighted_mean(
            _robust_zero_loss(p_curv - y_curv, kind='smoothl1', beta=float(config.get('curvature_smoothl1_beta', 0.2))),
            weight_ht,
        )
    else:
        curvature = torch.zeros((), device=pred_norm.device)

    hh = torch.as_tensor(heights, dtype=pred_norm.dtype, device=pred_norm.device).reshape(-1)
    dh = hh[1:] - hh[:-1]
    dh = dh / torch.clamp(dh.mean(), min=1.0e-6)
    dh = dh.reshape(1, -1, 1, 1, 1)
    p_hdiff = (p[:, 1:] - p[:, :-1]) / dh
    y_hdiff = (y[:, 1:] - y[:, :-1]) / dh
    hpair_w = 0.5 * (weight_ht[1:] + weight_ht[:-1])
    height_consistency = _weighted_mean(
        _robust_zero_loss(p_hdiff - y_hdiff, kind='smoothl1', beta=beta),
        hpair_w,
    )


    p_delta = p[:, :, 1:] - p[:, :, :-1]
    y_delta = y[:, :, 1:] - y[:, :, :-1]
    decreasing_mask = (y_delta < 0).to(dtype=p_delta.dtype)
    monotonic = _weighted_mean(torch.relu(p_delta) ** 2 * decreasing_mask, weight_ht)

    total = (
        weights['point_log_weight'] * point
        + weights['relative_weight'] * relative
        + weights['amplitude_weight'] * amplitude
        + weights['slope_weight'] * slope
        + weights['curvature_weight'] * curvature
        + weights['threshold5_weight'] * threshold5
        + weights['threshold10_weight'] * threshold10
        + weights['cvar_weight'] * cvar
        + weights['height_consistency_weight'] * height_consistency
        + weights['monotonic_weight'] * monotonic
    )

    rel_raw = raw_relative_error_from_log_error(err)
    p5 = (rel_raw < 0.05).double().mean()
    p10 = (rel_raw < 0.10).double().mean()

    metrics = {
        'total': total,
        'point_log': point,
        'relative': relative,
        'amplitude': amplitude,
        'slope': slope,
        'curvature': curvature,
        'threshold5': threshold5,
        'threshold10': threshold10,
        'cvar': cvar,
        'height': height_consistency,
        'monotonic': monotonic,
        'mean_rel': rel_raw.mean(),
        'p5': p5,
        'p10': p10,
        'under5': p5,
        'under10': p10,
    }
    return total, metrics


def _accumulate(acc, metrics, batch_size):
    for k, v in metrics.items():
        acc[k] = acc.get(k, 0.0) + float(v.detach().cpu()) * int(batch_size)


def _average_acc(acc, n):
    return {k: v / max(1, int(n)) for k, v in acc.items()}


def _amp_enabled(config, device):
    return bool(config.get('use_amp', False)) and device.type == 'cuda'


def _amp_dtype(config):
    name = str(config.get('amp_dtype', 'float16')).lower()
    return torch.bfloat16 if name in ['bfloat16', 'bf16'] else torch.float16


def amp_autocast_context(config, device):

    if not _amp_enabled(config, device):
        return nullcontext()
    dtype = _amp_dtype(config)
    if device.type == 'cpu' and dtype == torch.float16:
        dtype = torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def make_grad_scaler(config, device):
    enabled = _amp_enabled(config, device) and _amp_dtype(config) == torch.float16
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def batch_train(
    model, loader, optimizer, scaler, device,
    y_normalizer, meta, config, stage, weight_ht,
):
    model.train()
    acc = {}
    grad_clip = float(config.get('grad_clip', 1.0))
    accumulation_steps = max(1, int(config.get('gradient_accumulation_steps', 1)))
    n_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, n_batches - group_start)
        with amp_autocast_context(config, device):
            out = model(x)
            loss, metrics = physical_loss(
                out, y, y_normalizer, meta['times'], meta['heights'],
                weight_ht, config, stage=stage,
            )
            scaled_loss = loss / float(group_size)
        scaler.scale(scaled_loss).backward()
        should_step = ((batch_index + 1) % accumulation_steps == 0) or (
            batch_index + 1 == n_batches
        )
        if should_step:

            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        _accumulate(acc, metrics, x.shape[0])
    return _average_acc(acc, len(loader.dataset))


def batch_validate(model, loader, device, y_normalizer, meta, config, stage, weight_ht):
    model.eval()
    acc = {}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with amp_autocast_context(config, device):
                out = model(x)
                _, metrics = physical_loss(
                    out, y, y_normalizer, meta['times'], meta['heights'],
                    weight_ht, config, stage=stage,
                )
            _accumulate(acc, metrics, x.shape[0])
    return _average_acc(acc, len(loader.dataset))


def _normalizer_to_cpu(normalizer):
    copied = copy.deepcopy(normalizer)
    if hasattr(copied, 'cpu'):
        return copied.cpu()
    for name in ['mean', 'std', 'mean_curve', 'std_height']:
        if hasattr(copied, name):
            value = getattr(copied, name)
            if torch.is_tensor(value):
                setattr(copied, name, value.detach().cpu())
    return copied


def save_checkpoint(
    path,
    model,
    optimizer,
    config,
    x_normalizer,
    y_normalizer,
    meta,
    epoch,
    stage,
    metrics=None,
    save_optimizer=True,
):

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    state = {
        'model': {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        'optimizer': (
            optimizer.state_dict()
            if optimizer is not None and bool(save_optimizer)
            else None
        ),
        'config': copy.deepcopy(config),
        'meta': copy.deepcopy(meta),
        'x_normalizer': _normalizer_to_cpu(x_normalizer),
        'y_normalizer': _normalizer_to_cpu(y_normalizer),
        'epoch': int(epoch),
        'stage': int(stage),
        'metrics': copy.deepcopy(metrics or {}),
    }

    temp_path = path + '.writing'
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        torch.save(state, temp_path)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def safe_torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_matching_weights(model, checkpoint_path):

    ckpt = safe_torch_load(checkpoint_path, map_location='cpu')
    source = ckpt.get('model', ckpt)
    target = model.state_dict()
    matched = {
        k: v for k, v in source.items()
        if k in target and tuple(v.shape) == tuple(target[k].shape)
    }
    missing_before = len(target) - len(matched)
    model.load_state_dict(matched, strict=False)
    print(
        f'Warm start: loaded {len(matched)}/{len(target)} tensors from '
        f'{checkpoint_path}; unmatched={missing_before}'
    )


def selection_value(metrics, name='composite', config=None):
    name = str(name).lower()
    config = config or {}
    if name in ['composite', 'p5_p10_meanrel', 'target_composite']:
        p5_weight = float(config.get('composite_p5_weight', 1.0))
        p10_weight = float(config.get('composite_p10_weight', 0.25))
        mean_weight = float(config.get('composite_mean_rel_weight', 0.01))
        return (
            p5_weight * (1.0 - float(metrics['p5']))
            + p10_weight * (1.0 - float(metrics['p10']))
            + mean_weight * float(metrics['mean_rel'])
        )
    if name in ['mean_rel', 'relative_error']:
        return float(metrics['mean_rel'])
    if name in ['p5', 'under5']:
        return 1.0 - float(metrics['p5'])
    if name in ['p10', 'under10']:
        return 1.0 - float(metrics['p10'])
    if name in ['point', 'point_log']:
        return float(metrics['point_log'])
    if name in ['threshold', 'threshold5']:
        return float(metrics['threshold5'])
    if name == 'threshold10':
        return float(metrics['threshold10'])
    if name in ['total', 'loss']:
        return float(metrics['total'])
    raise ValueError(f'Unsupported best_metric={name!r}')


def _make_scheduler(optimizer, config, stage):
    stage = int(stage)
    if stage == 1:
        factor = float(config.get('lr_plateau_factor', 0.5))
        patience = int(config.get('lr_plateau_patience', 3))
    else:
        factor = float(config.get(f'stage{stage}_lr_plateau_factor', 0.5))
        patience = int(config.get(f'stage{stage}_lr_plateau_patience', 4))
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        min_lr=float(config.get('min_learning_rate', 1.0e-7)),
    )


def _unique_parameters(parameters):
    seen = set()
    out = []
    for p in parameters:
        if id(p) not in seen:
            out.append(p)
            seen.add(id(p))
    return out


def _stage_group_lr(config, stage, group_name):
    stage = int(stage)
    common = float(config.get('learning_rate', 1.0e-4))
    stage_common = float(config.get(f'stage{stage}_learning_rate', common))
    return float(config.get(f'stage{stage}_{group_name}_learning_rate', stage_common))


def build_optimizer(model, config):

    encoder_params = _unique_parameters(model.encoder_parameters())
    residual_params = _unique_parameters(model.residual_parameters())
    excluded = {id(p) for p in encoder_params + residual_params}
    base_params = [
        p for p in _unique_parameters(model.base_decoder_parameters())
        if id(p) not in excluded
    ]
    groups = []
    if encoder_params:
        groups.append({
            'params': encoder_params,
            'lr': _stage_group_lr(config, 1, 'encoder'),
            'name': 'encoder',
        })
    if base_params:
        groups.append({
            'params': base_params,
            'lr': _stage_group_lr(config, 1, 'base'),
            'name': 'base',
        })
    if residual_params:
        groups.append({
            'params': residual_params,
            'lr': _stage_group_lr(config, 1, 'residual'),
            'name': 'residual',
        })
    if not groups:
        raise ValueError('Model exposes no parameters for AdamW.')
    return AdamW(
        groups,
        weight_decay=float(config.get('weight_decay', 1.0e-5)),
        betas=tuple(config.get('adam_betas', [0.9, 0.999])),
        eps=float(config.get('adam_eps', 1.0e-8)),
    )


def configure_training_stage(model, optimizer, config, stage):

    stage = int(stage)
    for parameter in model.parameters():
        parameter.requires_grad = True
    freeze_encoder = bool(config.get(f'stage{stage}_freeze_spatial_encoder', False))
    model.set_spatial_encoder_trainable(not freeze_encoder)
    for group in optimizer.param_groups:
        name = str(group.get('name', 'base'))
        group['lr'] = _stage_group_lr(config, stage, name)
    return freeze_encoder


def _optimizer_lr_text(optimizer):
    parts = []
    for i, group in enumerate(optimizer.param_groups):
        name = group.get('name', str(i))
        parts.append(f'{name}:{group["lr"]:.3e}')
    return ','.join(parts)


def _prune_old_checkpoints(paths, keep_last=3):
    keep_last = max(1, int(keep_last))
    while len(paths) > keep_last:
        old = paths.pop(0)
        if os.path.exists(old):
            try:
                os.remove(old)
            except OSError:
                pass


def run_train(
    model,
    train_loader,
    val_loader,
    x_normalizer,
    y_normalizer,
    meta,
    config,
    device,
    log_file,
):
    epochs = int(config['epochs'])
    save_step = int(config.get('save_step', 5))
    keep_last = int(config.get('save_keep_last', 3))
    patience = int(config.get('patience', 15))
    thre_epoch = int(config.get('thre_epoch', 0))
    stage2_start = int(config.get('stage2_start_epoch', epochs + 1))
    stage3_start = int(config.get('stage3_start_epoch', epochs + 1))
    if stage2_start <= epochs and stage3_start <= stage2_start:
        raise ValueError('stage3_start_epoch must be greater than stage2_start_epoch.')

    best_metric_name = config.get('best_metric', 'composite')
    scheduler_metric_name = config.get('scheduler_metric', best_metric_name)
    output_dir = config['_resolved_output_dir']
    model_path = os.path.join(
        output_dir, 'model', f"{config['name']}_{config['n_out']}"
    )
    temp_path = os.path.join(
        output_dir, 'temp_model', f"{config['name']}_{config['n_out']}"
    )

    y_normalizer = move_normalizer(y_normalizer, device)
    weight_ht = make_height_time_loss_weights(config, meta, device)
    print('Height-time weight: mean=%.4f min=%.4f max=%.4f' % (
        float(weight_ht.mean()),
        float(weight_ht.min()),
        float(weight_ht.max()),
    ))
    print('Stage-1 loss weights:', stage_loss_weights(config, 1))
    print('Stage-2 loss weights:', stage_loss_weights(config, 2))
    print('Stage-3 loss weights:', stage_loss_weights(config, 3))

    current_stage = 1
    optimizer = build_optimizer(model, config)
    configure_training_stage(model, optimizer, config, current_stage)
    scheduler = _make_scheduler(optimizer, config, current_stage)
    scaler = make_grad_scaler(config, device)

    best_value = float('inf')
    best_path = model_path + '_best.pkl'
    best_epoch = None
    stop_counter = 0
    rolling_paths = []
    min_delta = float(config.get('early_stopping_min_delta', 0.0))

    for ep in range(1, epochs + 1):
        if ep >= stage3_start:
            stage = 3
        elif ep >= stage2_start:
            stage = 2
        else:
            stage = 1

        if stage != current_stage:
            current_stage = stage
            frozen = configure_training_stage(model, optimizer, config, stage)
            scheduler = _make_scheduler(optimizer, config, stage)
            stop_counter = 0
            print(
                f'Entering stage {stage} at epoch {ep}: '
                f'spatial encoder frozen={frozen}; '
                f'lr={_optimizer_lr_text(optimizer)}'
            )

        t0 = default_timer()
        lr_used = _optimizer_lr_text(optimizer)
        train_m = batch_train(
            model, train_loader, optimizer, scaler, device,
            y_normalizer, meta, config, stage, weight_ht
        )
        val_m = batch_validate(
            model, val_loader, device,
            y_normalizer, meta, config, stage, weight_ht
        )
        select = selection_value(val_m, best_metric_name, config)
        scheduler_select = selection_value(val_m, scheduler_metric_name, config)
        scheduler.step(scheduler_select)

        if ep % save_step == 0:
            temp_file = temp_path + f'_epoch_{ep}.pkl'
            save_checkpoint(
                temp_file, model, optimizer, config,
                x_normalizer, y_normalizer, meta,
                ep, stage, val_m, save_optimizer=True
            )
            rolling_paths.append(temp_file)
            _prune_old_checkpoints(rolling_paths, keep_last=keep_last)



        improved = select < best_value - min_delta
        if improved:
            best_value = select
            best_epoch = ep
            stop_counter = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                config,
                x_normalizer,
                y_normalizer,
                meta,
                ep,
                stage,
                val_m,
                save_optimizer=False,
            )
        elif ep > thre_epoch:
            stop_counter += 1

        elapsed = default_timer() - t0
        msg = (
            f"{ep} {elapsed:.3f} stage={stage} "
            f"lr_used={lr_used} next_lr={_optimizer_lr_text(optimizer)} "
            f"train={train_m['total']:.8e} "
            f"trainRawMeanRel={100.0 * train_m['mean_rel']:.3f}% "
            f"trainP5={100.0 * train_m['p5']:.2f}% "
            f"trainP10={100.0 * train_m['p10']:.2f}% "
            f"val={val_m['total']:.8e} "
            f"valRawMeanRel={100.0 * val_m['mean_rel']:.3f}% "
            f"valP5={100.0 * val_m['p5']:.2f}% "
            f"valP10={100.0 * val_m['p10']:.2f}% "
            f"valComposite={select:.8e} "
            f"valPoint={val_m['point_log']:.6e} "
            f"valAmp={val_m['amplitude']:.6e} "
            f"valSlope={val_m['slope']:.6e} "
            f"valCVaR={val_m['cvar']:.6e}"
        )
        print(msg)
        print(msg, file=log_file)
        log_file.flush()

        if ep > thre_epoch and stop_counter >= patience:
            print(
                f'Early stop at epoch {ep}; '
                f'best {best_metric_name}={best_value:.8e}'
            )
            break

    if best_epoch is None or not os.path.exists(best_path):
        raise RuntimeError('Training finished without producing a best checkpoint.')
    return best_path, best_epoch, best_value, optimizer


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_split_indices(meta, output_dir):
    payload = {
        'split_seed': int(meta['split_seed']),
        'train_idx': torch.as_tensor(meta['train_idx']).cpu().tolist(),
        'val_idx': torch.as_tensor(meta['val_idx']).cpu().tolist(),
        'test_idx': torch.as_tensor(meta['test_idx']).cpu().tolist(),
    }
    path = os.path.join(output_dir, 'split_indices.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def validated_eval_quantiles(config):
    raw = config.get('eval_quantiles', [0.50, 0.90, 0.95, 0.99])
    if isinstance(raw, (int, float)):
        raw = [raw]
    quantiles = sorted({float(value) for value in raw})
    for quantile in quantiles:
        if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError(
                f'eval_quantiles entries must be finite and lie in [0,1], got {quantile}.'
            )
    return quantiles


def relative_quantile_fields(relative_values, quantiles):
    values = torch.as_tensor(relative_values, dtype=torch.float64).reshape(-1).cpu()
    if values.numel() == 0:
        raise ValueError('Cannot compute relative-error quantiles from an empty tensor.')
    fields = {}
    for quantile in quantiles:
        percentile = 100.0 * float(quantile)
        if math.isclose(percentile, round(percentile), rel_tol=0.0, abs_tol=1.0e-10):
            suffix = str(int(round(percentile)))
        else:
            suffix = ('%.6g' % percentile).replace('.', '_')
        key = f'rel_q{suffix}'
        value = float(torch.quantile(values, float(quantile)))
        fields[key] = value
        fields[key + '_percent'] = 100.0 * value
    return fields


def evaluate_test_and_write_diagnostics(
    model, test_loader, y_normalizer, meta, config, device, best_checkpoint,
):

    checkpoint = safe_torch_load(best_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    y_normalizer = move_normalizer(y_normalizer, device)
    n_height, n_time = int(meta['n_height']), int(meta['n_time'])
    sum_rel_ht = torch.zeros(n_height, n_time, dtype=torch.float64)
    count_ht = torch.zeros(n_height, n_time, dtype=torch.float64)
    under5_ht = torch.zeros(n_height, n_time, dtype=torch.float64)
    under10_ht = torch.zeros(n_height, n_time, dtype=torch.float64)
    relative_chunks = []

    with torch.no_grad():
        for x, true_norm in test_loader:
            x = x.to(device)
            true_norm = true_norm.to(device)
            with amp_autocast_context(config, device):
                pred_norm = model(x)
            pred_log = _reshape_ht(
                decode_normalized(y_normalizer, pred_norm), n_height, n_time
            )
            true_log = _reshape_ht(
                decode_normalized(y_normalizer, true_norm), n_height, n_time
            )
            rel = raw_relative_error_from_log_error(pred_log - true_log).cpu()
            relative_chunks.append(rel)

            reduce_dims = (0, 3, 4)
            sum_rel_ht += rel.sum(dim=reduce_dims)
            under5_ht += (rel < 0.05).double().sum(dim=reduce_dims)
            under10_ht += (rel < 0.10).double().sum(dim=reduce_dims)
            points_per_ht = rel.shape[0] * rel.shape[3] * rel.shape[4]
            count_ht += float(points_per_ht)

    if torch.any(count_ht <= 0):
        raise RuntimeError('Test diagnostics encountered an empty height-time cell.')
    mean_ht = sum_rel_ht / count_ht
    p5_ht = under5_ht / count_ht
    p10_ht = under10_ht / count_ht
    total_count = float(count_ht.sum())
    overall = {
        'checkpoint': os.path.abspath(best_checkpoint),
        'checkpoint_epoch': int(checkpoint.get('epoch', -1)),
        'n_test_models': int(len(test_loader.dataset)),
        'n_test_points': int(total_count),
        'raw_mean_relative_error': float(sum_rel_ht.sum() / total_count),
        'raw_mean_relative_error_percent': float(100.0 * sum_rel_ht.sum() / total_count),
        'p5': float(under5_ht.sum() / total_count),
        'p5_percent': float(100.0 * under5_ht.sum() / total_count),
        'p10': float(under10_ht.sum() / total_count),
        'p10_percent': float(100.0 * under10_ht.sum() / total_count),
    }
    all_relative_ht = torch.cat(relative_chunks, dim=0)
    quantiles = validated_eval_quantiles(config)
    overall.update(relative_quantile_fields(all_relative_ht, quantiles))

    heights = torch.as_tensor(meta['heights']).reshape(-1).cpu().tolist()
    times = torch.as_tensor(meta['times']).reshape(-1).cpu().tolist()
    by_height = []
    for ih, height in enumerate(heights):
        count = float(count_ht[ih].sum())
        row = {
            'height_index': ih,
            'height': float(height),
            'n_points': int(count),
            'raw_mean_relative_error': float(sum_rel_ht[ih].sum() / count),
            'raw_mean_relative_error_percent': float(100.0 * sum_rel_ht[ih].sum() / count),
            'p5': float(under5_ht[ih].sum() / count),
            'p5_percent': float(100.0 * under5_ht[ih].sum() / count),
            'p10': float(under10_ht[ih].sum() / count),
            'p10_percent': float(100.0 * under10_ht[ih].sum() / count),
        }
        row.update(relative_quantile_fields(all_relative_ht[:, ih], quantiles))
        by_height.append(row)
    by_time = []
    for it, time_value in enumerate(times):
        count = float(count_ht[:, it].sum())
        row = {
            'time_index': it,
            'time': float(time_value),
            'n_points': int(count),
            'raw_mean_relative_error': float(sum_rel_ht[:, it].sum() / count),
            'raw_mean_relative_error_percent': float(100.0 * sum_rel_ht[:, it].sum() / count),
            'p5': float(under5_ht[:, it].sum() / count),
            'p5_percent': float(100.0 * under5_ht[:, it].sum() / count),
            'p10': float(under10_ht[:, it].sum() / count),
            'p10_percent': float(100.0 * under10_ht[:, it].sum() / count),
        }
        row.update(relative_quantile_fields(all_relative_ht[:, :, it], quantiles))
        by_time.append(row)

    output_dir = config['_resolved_output_dir']
    os.makedirs(output_dir, exist_ok=True)
    overall_path = os.path.join(output_dir, 'test_overall.json')
    diagnostics_path = os.path.join(output_dir, 'test_diagnostics.json')
    height_path = os.path.join(output_dir, 'test_by_height.csv')
    time_path = os.path.join(output_dir, 'test_by_time.csv')
    with open(overall_path, 'w', encoding='utf-8') as handle:
        json.dump(overall, handle, ensure_ascii=False, indent=2)
    with open(diagnostics_path, 'w', encoding='utf-8') as handle:
        json.dump(
            {'overall': overall, 'by_height': by_height, 'by_time': by_time},
            handle, ensure_ascii=False, indent=2,
        )
    _write_csv(height_path, by_height, list(by_height[0].keys()))
    _write_csv(time_path, by_time, list(by_time[0].keys()))
    print(
        'Final untouched test: RawMeanRel=%.6f%% P5=%.4f%% P10=%.4f%%' % (
            overall['raw_mean_relative_error_percent'],
            overall['p5_percent'], overall['p10_percent'],
        )
    )
    print('Test diagnostics:', diagnostics_path, height_path, time_path)
    return overall


def load_yaml_config(config_file):
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030']
    last_error = None
    for enc in encodings:
        try:
            with open(config_file, 'r', encoding=enc) as f:
                print(f'Using YAML encoding: {enc}')
                return yaml.full_load(f)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error

def _resolve_model_class():
    for class_name in [
        'HTUFNO_AEM_v5_Hybrid',
        'HTUFNO_AEM_v5_corrected',
        'HTUFNO_AEM_v4_Hybrid',
    ]:
        candidate = getattr(v5_model_module, class_name, None)
        if candidate is not None:
            return candidate
    raise ImportError(
        'HTUFNO_AEM_v5_corrected.py must export HTUFNO_AEM_v5_Hybrid.'
    )


def _modes_xyz(config):
    value = config.get('modes_xyz', config.get('modes', 18))
    if isinstance(value, (int, float)):
        result = (int(value), int(value), int(value))
    else:
        result = tuple(int(v) for v in value)
    if len(result) != 3 or any(v <= 0 for v in result):
        raise ValueError(f'modes_xyz must contain three positive integers, got {result}.')
    return result


def build_model_from_config(config, meta, n_out, device):
    model_class = _resolve_model_class()
    modes_xyz = _modes_xyz(config)
    kwargs = {
        'modes1': modes_xyz[0],
        'modes2': modes_xyz[1],
        'modes3': modes_xyz[2],
        'modes_xyz': modes_xyz,
        'width': int(config.get('width', 42)),
        'nout': int(n_out),
        'n_height': int(meta['n_height']),
        'n_time': int(meta['n_time']),
        'init_func': config.get('init_func', 'xavier_normal'),
        'layer_fno': int(config.get('layer_fno', 3)),
        'layer_ufno': int(config.get('layer_ufno', 3)),
        'act_func': config.get('act_fno', 'gelu'),
        'padding': config.get(
            'padding', 0 if 'spatial_padding' in config else 8
        ),
        'spatial_padding': config.get('spatial_padding', config.get('padding', 8)),
        'use_kan_lift': bool(config.get('use_kan_lift', True)),
        'use_coord_channels': bool(config.get('use_coord_channels', True)),
        'use_wavelet_branch': bool(config.get('use_wavelet_branch', True)),
        'wavelet_layers': int(config.get('wavelet_layers', 3)),
        'wavelet_hidden_multiplier': float(config.get('wavelet_hidden_multiplier', 1.0)),
        'wavelet_scale_init': float(config.get('wavelet_scale_init', 1.0)),
        'wavelet_dropout': float(config.get('wavelet_dropout', 0.0)),
        'latent_dim': int(config.get('latent_dim', 256)),
        'summary_hidden': int(config.get('summary_hidden', 512)),
        'summary_dropout': float(config.get('summary_dropout', 0.0)),
        'summary_global_grid': int(config.get('summary_global_grid', 3)),
        'summary_global_z_bins': int(config.get('summary_global_z_bins', 8)),
        'summary_center_windows': config.get('summary_center_windows', [3, 7, 15]),
        'summary_center_z_bins': int(config.get('summary_center_z_bins', 8)),
        'spatial_token_grid': config.get('spatial_token_grid', 3),
        'use_height_interaction': bool(config.get('use_height_interaction', True)),
        'height_interaction_layers': int(config.get('height_interaction_layers', 2)),
        'height_interaction_heads': int(config.get('height_interaction_heads', 4)),
        'height_interaction_hidden': int(config.get('height_interaction_hidden', 512)),
        'temporal_width': int(config.get('temporal_width', 128)),
        'temporal_modes': int(config.get('temporal_modes', 12)),
        'temporal_pad_to': int(config.get('temporal_pad_to', 68)),
        'temporal_depth': int(config.get('temporal_depth', 2)),
        'temporal_hidden': int(config.get('temporal_hidden', 256)),
        'temporal_dropout': float(config.get('temporal_dropout', 0.0)),
        'height_specific': bool(config.get('height_specific', False)),
        'output_parameterization': config.get('output_parameterization', 'amplitude_shape'),
        'amplitude_anchor_gates': int(config.get('amplitude_anchor_gates', 3)),
        'time_values': meta['times'],
        'height_values': meta['heights'],
        'temporal_time_transform': config.get(
            'temporal_time_transform', config.get('slope_time_transform', 'log10')
        ),
        'temporal_center_padding': bool(config.get('temporal_center_padding', True)),
        'use_residual_corrector': bool(config.get('use_residual_corrector', True)),
        'depth_token_bins': int(config.get('depth_token_bins', 16)),
        'depth_token_windows': config.get('depth_token_windows', [3, 7, 15]),
        'depth_token_hidden': int(config.get('depth_token_hidden', 384)),
        'depth_token_layers': int(config.get('depth_token_layers', 2)),
        'residual_heads': int(config.get('residual_heads', 4)),
        'residual_hidden': int(config.get('residual_hidden', 384)),
        'residual_temporal_modes': int(config.get('residual_temporal_modes', 10)),
        'residual_temporal_layers': int(config.get('residual_temporal_layers', 2)),
        'residual_temporal_operator': config.get('residual_temporal_operator', 'attention'),
        'residual_temporal_attention_heads': config.get(
            'residual_temporal_attention_heads', config.get('residual_heads', 4)
        ),
        'residual_height_layers': int(config.get('residual_height_layers', 1)),
        'residual_dropout': float(config.get('residual_dropout', 0.0)),
        'residual_init_scale': float(config.get('residual_init_scale', 0.05)),
    }
    signature = inspect.signature(model_class.__init__)
    explicit = set(signature.parameters) - {'self'}
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    passed = kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in explicit}
    skipped = sorted(set(kwargs) - set(passed))
    if skipped:
        print('Model constructor ignored unsupported compatibility keys:', skipped)
    print(f'Building model class {model_class.__name__} with modes_xyz={modes_xyz}')
    return model_class(**passed).to(device)


def _resolve_relative_path(path, base_dir):
    if path is None:
        return None
    path = os.fspath(path)
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(base_dir, path))


def main(item=None, config_file=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if config_file is None:
        config_file = os.path.join(script_dir, 'config_HTUFNO_AEM_v6.yml')
    else:
        config_file = _resolve_relative_path(config_file, script_dir)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f'Cannot find {config_file}')
    config_all = load_yaml_config(config_file)
    if item is None:
        preferred = 'HTUFNO_AEM_v5_corrected_config'
        item = preferred if preferred in config_all else next(iter(config_all))
    if item not in config_all:
        raise KeyError(f'Cannot find config item {item!r} in {config_file}')
    config = dict(config_all[item])

    output_setting = config.get('output_dir', os.path.join('outputs', config.get('name', item)))
    output_dir = _resolve_relative_path(output_setting, script_dir)
    config['_resolved_output_dir'] = output_dir
    os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'temp_model'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'log'), exist_ok=True)

    device = torch.device(
        f"cuda:{config.get('cuda_id', 0)}" if torch.cuda.is_available() else 'cpu'
    )
    print('Using device:', device)
    n_out = int(config['n_out'])
    (
        train_loader, val_loader, test_loader, _,
        x_normalizer, y_normalizer, meta,
    ) = get_batch_data(config, int(config['batch_size']), n_out)
    config['n_height'] = int(meta['n_height'])
    config['n_time'] = int(meta['n_time'])
    split_path = save_split_indices(meta, output_dir)
    print('Saved split indices:', split_path)

    model = build_model_from_config(config, meta, n_out, device)
    print_model(model, bool(config.get('print_model_flag', False)))
    print(
        'Model setting: Wavelet=%s DepthResidual=%s KANLift(actual)=%s' % (
            getattr(model, 'use_wavelet_branch', 'unknown'),
            getattr(model, 'use_residual_corrector', 'unknown'),
            getattr(model, 'using_kan_lift', 'unknown'),
        )
    )

    warm_start = config.get('warm_start_checkpoint', None)
    if warm_start:
        warm_start = _resolve_relative_path(warm_start, script_dir)
        if not os.path.exists(warm_start):
            raise FileNotFoundError(f'warm_start_checkpoint does not exist: {warm_start}')
        load_matching_weights(model, warm_start)

    log_path = os.path.join(output_dir, 'log', f"{config['name']}_{n_out}.log")
    with open(log_path, 'a+', encoding='utf-8') as log_file:
        print('=======================================')
        print('Now begin to train HTUFNO_AEM_v5_corrected')
        print('=======================================')
        best_path, best_epoch, best_value, _ = run_train(
            model, train_loader, val_loader, x_normalizer, y_normalizer,
            meta, config, device, log_file,
        )
    print(
        f'Best validation checkpoint: epoch={best_epoch} '
        f'score={best_value:.8e} path={best_path}'
    )
    return evaluate_test_and_write_diagnostics(
        model, test_loader, y_normalizer, meta, config, device, best_path,
    )


if __name__ == '__main__':
    selected_item = sys.argv[1] if len(sys.argv) > 1 else None
    selected_config = sys.argv[2] if len(sys.argv) > 2 else None
    main(selected_item, selected_config)
