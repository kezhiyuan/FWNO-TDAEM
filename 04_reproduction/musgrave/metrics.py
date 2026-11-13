import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from paths import DATA, OUTPUT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_archive():
    manifest = json.loads((DATA / 'package_manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path = DATA / record['path']
        if not path.is_file() or sha256(path) != record['sha256']:
            raise ValueError(f'Archive checksum mismatch: {record["path"]}')
    return len(manifest['files'])


def run(output_dir=OUTPUT, verify_files=True):
    output_dir = Path(output_dir)
    verified = verify_archive() if verify_files else 0
    expected = json.loads((DATA / 'comparison_metrics_50m.json').read_text(encoding='utf-8'))
    with np.load(DATA / 'fwno_predictions_50m.npz', allow_pickle=False) as archive:
        ids = archive['model_ids']
        centers = archive['centers_local_m']
        times = archive['times_ms']
        height = float(archive['height_m'])
        prediction = archive['amplitude_nT_per_s'].astype(np.float64)
    np.testing.assert_array_equal(ids, np.arange(1, 397))
    assert height == 50.0 and prediction.shape == (396, 16) and centers.shape == (396, 2)
    with (DATA / 'time_gates_50m.csv').open(encoding='ascii', newline='') as stream:
        gates = list(csv.DictReader(stream))
    assert [int(row['gate']) for row in gates] == list(range(1, 17))
    assert all(float(row['flight_height_m']) == 50 and row['amplitude_unit'] == 'nT/s' for row in gates)
    np.testing.assert_array_equal([float(row['opening_time_ms']) for row in gates], times.round(6))
    directory = DATA / 'FD_predictions_50m'
    expected_names = {f'dBzdt_predict_all_{i}.txt' for i in ids}
    assert {p.name for p in directory.glob('*.txt')} == expected_names
    reference, hashes = [], {}
    for model_id in ids:
        path = directory / f'dBzdt_predict_all_{model_id}.txt'
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        assert len(lines) == 16 and all(len(line.split()) == 1 for line in lines)
        reference.append([float(line) for line in lines])
        hashes[path.name] = sha256(path)
    reference = np.asarray(reference, dtype=np.float64)
    assert hashes == expected['reference_provenance']['files_sha256']
    assert np.isfinite(reference).all() and (reference > 0).all()
    assert np.isfinite(prediction).all() and (prediction > 0).all()
    residual = prediction - reference
    error = 100.0 * np.abs(residual / reference)
    worst_window, worst_gate = np.unravel_index(error.argmax(), error.shape)
    maxima = error.max(axis=1)
    result = {
        'height_m': height, 'window_count': len(ids), 'gate_count': len(times), 'pair_count': int(error.size),
        'mean_relative_error_percent': float(error.mean()),
        'median_relative_error_percent': float(np.median(error)),
        'p95_relative_error_percent': float(np.quantile(error, .95)),
        'max_relative_error_percent': float(error.max()),
        'mean_signed_relative_difference_percent': float((100 * residual / reference).mean()),
        'rmse_nT_per_s': float(np.sqrt(np.mean(residual**2))),
        'global_relative_l2_percent': float(100 * np.linalg.norm(residual) / np.linalg.norm(reference)),
        'point_count_below_5_percent': int((error < 5).sum()),
        'point_count_below_10_percent': int((error < 10).sum()),
        'window_count_all_gates_below_5_percent': int((maxima < 5).sum()),
        'window_count_all_gates_below_10_percent': int((maxima < 10).sum()),
        'worst_pair': {'window_id': int(ids[worst_window]), 'gate': int(worst_gate + 1),
                       'opening_time_ms': float(times[worst_gate]),
                       'reference_nT_per_s': float(reference[worst_window, worst_gate]),
                       'fwno_nT_per_s': float(prediction[worst_window, worst_gate])},
        'gate_mean_relative_error_percent': error.mean(axis=0).tolist(),
        'illustrated_windows': {str(i): {'mean_relative_error_percent': float(error[i-1].mean()),
                                         'max_relative_error_percent': float(maxima[i-1])}
                                for i in (67, 199, 331)},
    }
    for key, value in result.items():
        if isinstance(value, (int, float, list)):
            np.testing.assert_allclose(value, expected[key], rtol=1e-12, atol=1e-12, err_msg=key)
        else:
            assert value == expected[key], key
    result['verification'] = {'archive_files_sha256_verified': verified,
                              'fd_files_sha256_verified': len(hashes),
                              'recorded_metrics_reproduced': True,
                              'forward_solver_executed': False}
    result['reference_provenance'] = {**expected['reference_provenance'], 'directory': 'FD_predictions_50m'}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'comparison_metrics_50m.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (output_dir / 'paired_responses_50m.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model_id', 'x_local_m', 'y_local_m', 'flight_height_m', 'gate', 'opening_time_ms',
                         'reference_amplitude_nT_per_s', 'fwno_amplitude_nT_per_s', 'relative_error_percent'])
        for row, (model_id, (x, y)) in enumerate(zip(ids, centers)):
            for col, time in enumerate(times):
                writer.writerow([model_id, x, y, height, col + 1, f'{time:.6f}', f'{reference[row,col]:.12g}',
                                 f'{prediction[row,col]:.9g}', f'{error[row,col]:.9g}'])
    assert (output_dir / 'paired_responses_50m.csv').read_bytes() == (DATA / 'paired_responses_50m.csv').read_bytes()
    print(f'Verified {len(ids)} windows / {error.size} pairs: MRE {error.mean():.12f}%, max {error.max():.12f}%')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recompute the archived Musgrave 50 m comparison without running a solver.')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--skip-archive-hashes', action='store_true', help='FD hashes are always checked.')
    args = parser.parse_args()
    run(args.output, not args.skip_archive_hashes)
