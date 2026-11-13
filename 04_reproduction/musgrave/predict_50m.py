import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from metrics import sha256
from paths import CHECKPOINT, DATA, FWNO_SOURCE, OUTPUT


def main():
    parser = argparse.ArgumentParser(description='Optional CUDA prediction using the packaged frozen FWNO checkpoint.')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--windows', type=Path, default=DATA / 'south_australia_windows.npz')
    parser.add_argument('--output', type=Path, default=OUTPUT / 'prediction_rerun')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.batch_size < 1 or args.repetitions < 1:
        parser.error('batch-size and repetitions must be positive')
    import torch
    sys.path.insert(0, str(FWNO_SOURCE))
    import HTUFNO_AEM_v6_main as fwno
    archived = json.loads((DATA / 'prediction_manifest.json').read_text(encoding='utf-8'))
    assert sha256(args.checkpoint) == archived['checkpoint_sha256'], 'Expected the archived epoch-82 checkpoint'
    assert sha256(args.windows) == archived['windows_sha256'], 'Expected the archived Musgrave window data'
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this optional prediction/timing protocol. CPU metrics use reproduce.py.')
    torch.set_num_threads(6)
    with np.load(args.windows, allow_pickle=False) as saved:
        raw = saved['sig'].astype(np.float32, copy=False)
        ids, centers, depth = saved['model_ids'], saved['centers_local_m'], saved['depth_edges_m']
    assert raw.shape == (396, 48, 48, 20)
    ckpt = fwno.safe_torch_load(str(args.checkpoint), map_location='cpu')
    config, meta = ckpt['config'], ckpt['meta']
    assert int(ckpt['epoch']) == 82 and config['nz_earth'] == 20
    assert config['input_transform'] == 'none' and config['target_transform'] == 'log_abs'
    times = np.asarray(torch.as_tensor(meta['times']), dtype=np.float64)
    heights = np.asarray(torch.as_tensor(meta['heights']), dtype=np.float64)
    h_index = np.flatnonzero(heights == 50.0).item()
    x_norm = ckpt['x_normalizer']
    y_norm = fwno.move_normalizer(ckpt['y_normalizer'], device)
    model = fwno.build_model_from_config(config, meta, config['n_out'], device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    del ckpt

    def infer(batch):
        normalized = x_norm.encode(torch.from_numpy(batch[..., None])).to(device)
        with fwno.amp_autocast_context(config, device):
            result = model(normalized)
        log_amp = fwno.decode_normalized(y_norm, result).reshape(len(batch), len(heights), len(times))
        return torch.pow(10.0, log_amp[:, h_index, :].float()).cpu().numpy()

    elapsed = []
    with torch.inference_mode():
        for _ in range(3):
            infer(raw[:args.batch_size])
        for repeat in range(args.repetitions):
            torch.cuda.synchronize(device)
            started = perf_counter()
            amplitudes = np.concatenate([infer(raw[i:i+args.batch_size]) for i in range(0, len(raw), args.batch_size)])
            torch.cuda.synchronize(device)
            elapsed.append(perf_counter() - started)
            if repeat == 0:
                first = amplitudes.copy()
            else:
                np.testing.assert_allclose(amplitudes, first, rtol=1e-5, atol=1e-10)
            print(f'Pass {repeat+1}: {elapsed[-1]:.6f} s', flush=True)
    with np.load(DATA / 'fwno_predictions_50m.npz', allow_pickle=False) as saved:
        difference = np.abs(amplitudes.astype(float) / saved['amplitude_nT_per_s'] - 1.0)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'fwno_predictions_50m.npz', model_ids=ids,
                        centers_local_m=centers, height_m=50., times_ms=times,
                        amplitude_nT_per_s=amplitudes, depth_edges_m=depth)
    with (args.output / 'fwno_predictions_50m.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model_id', 'x_local_m', 'y_local_m', 'height_m', 'gate', 'time_ms', 'fwno_amplitude_nT_per_s'])
        for i, model_id in enumerate(ids):
            for j, time in enumerate(times):
                writer.writerow([int(model_id), *centers[i], 50., j+1, f'{time:.6f}', f'{amplitudes[i,j]:.9g}'])
    record = {'checkpoint_sha256': sha256(args.checkpoint), 'checkpoint_epoch': 82,
              'windows_sha256': sha256(args.windows), 'batch_size': args.batch_size,
              'device': torch.cuda.get_device_name(device), 'torch': torch.__version__,
              'pass_seconds': elapsed, 'median_seconds': float(np.median(elapsed)),
              'max_relative_difference_from_archived_prediction': float(difference.max()),
              'timing_scope': archived['timing_scope'], 'window_count': 396,
              'height_m': 50., 'predicts_joint_height_count': 7, 'response_values': 6336,
              'time_gate_opening_ms': times.tolist()}
    (args.output / 'prediction_manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
