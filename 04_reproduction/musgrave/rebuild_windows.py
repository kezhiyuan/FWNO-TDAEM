import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from paths import DATA, OUTPUT
DZ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 100])
Z = np.r_[0, np.cumsum(DZ)]
CELL = 30
WINDOW = 48

def read_modem(path):
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    nx, ny, nz, _, scale = lines[1].split()
    nx, ny, nz = int(nx), int(ny), int(nz)
    if scale.upper() != 'LOGE':
        raise ValueError(f'Expected LOGE resistivity in {path}')
    widths = tuple(np.fromstring(line, sep=' ') for line in lines[2:5])
    assert tuple(map(len, widths)) == (nx, ny, nz)
    assert all(np.all(w > 0) for w in widths)
    values = np.fromstring(' '.join(lines[5:]), sep=' ')
    count = nx * ny * nz
    assert values.size == count + 4 and np.isfinite(values).all()

    ln_rho = values[:count].reshape(nz, ny, nx).transpose(2, 1, 0)[::-1, :, :]
    return ln_rho, widths, values[-4:-1], values[-1]

def edges(widths):
    return np.r_[0., np.cumsum(widths)]

def remap_weights(source_edges, target_edges):
    lower = np.maximum(target_edges[:-1, None], source_edges[None, :-1])
    upper = np.minimum(target_edges[1:, None], source_edges[None, 1:])
    weights = np.maximum(upper - lower, 0) / np.diff(target_edges)[:, None]
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    return weights

def build_windows(path, output):
    ln_rho, widths, header_origin, rotation = read_modem(path)
    full_rho, full_widths, original_origin, original_rotation = read_modem(path.with_name('iso.rho'))
    np.testing.assert_array_equal(ln_rho, full_rho[4:-4, 4:-4, :])
    for trimmed, full in zip(widths, (full_widths[0][4:-4], full_widths[1][4:-4], full_widths[2])):
        np.testing.assert_array_equal(trimmed, full)
    assert rotation == original_rotation == 0
    source = tuple(edges(w) for w in widths)
    offset = np.array([full_widths[0][:4].sum(), full_widths[1][:4].sum(), 0.])
    nx, ny = (int(axis[-1] // CELL) for axis in source[:2])
    x = np.arange(nx + 1) * CELL + (source[0][-1] - nx * CELL) / 2
    y = np.arange(ny + 1) * CELL + (source[1][-1] - ny * CELL) / 2

    wx, wy, wz = (remap_weights(old, new) for old, new in zip(source, (x, y, Z)))
    log_rho = np.einsum('ia,jb,kc,abc->ijk', wx, wy, wz, ln_rho / np.log(10.), optimize=True)
    centers = np.array([((x[i] + x[i+WINDOW]) / 2, (y[j] + y[j+WINDOW]) / 2)
                        for i in range(nx-WINDOW+1) for j in range(ny-WINDOW+1)])
    sig = np.stack([-log_rho[i:i+WINDOW, j:j+WINDOW]
                    for i in range(nx-WINDOW+1) for j in range(ny-WINDOW+1)]).astype(np.float32)
    assert sig.shape == (len(centers), 48, 48, 20) and np.isfinite(sig).all()
    np.savez_compressed(output, sig=sig, model_ids=np.arange(1,len(sig)+1),
                        centers_local_m=centers, grid_x_edges_local_m=x, grid_y_edges_local_m=y,
                        depth_edges_m=Z, corrected_origin_modem_m=original_origin+offset)
    grid = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_shape': list(ln_rho.shape), 'source_extent_m': [float(q[-1]) for q in source],
            'original_rho_header_origin_m': original_origin.tolist(), 'cropped_header_origin_m': header_origin.tolist(),
            'corrected_crop_origin_m': (original_origin+offset).tolist(),
            'resampled_shape': list(log_rho.shape), 'window_shape': list(sig.shape[1:]),
            'window_stride_m': CELL, 'window_depth_edges_m': Z.tolist(),
            'remap': 'Volume-overlap averaging of log10 resistivity with full target cell coverage'}
    return sig, centers, log_rho, x, y, grid

def main():
    parser = argparse.ArgumentParser(description='Rebuild the 396 Musgrave FWNO windows from the packaged original resistivity grids.')
    parser.add_argument('--source-dir', type=Path, default=DATA / 'source_models')
    parser.add_argument('--output', type=Path, default=OUTPUT / 'south_australia_windows.npz')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _, _, _, _, _, grid = build_windows(args.source_dir / 'iso-new.rho', args.output)
    with np.load(args.output, allow_pickle=False) as rebuilt, np.load(DATA / 'south_australia_windows.npz', allow_pickle=False) as saved:
        assert set(rebuilt.files) == set(saved.files)
        for name in saved.files:
            np.testing.assert_array_equal(rebuilt[name], saved[name], err_msg=name)
    grid['source'] = 'source_models/iso-new.rho'
    grid.pop('path', None)
    grid['all_archived_arrays_reproduced_exactly'] = True
    args.output.with_suffix('.json').write_text(json.dumps(grid, indent=2), encoding='utf-8')
    print('All 396 input windows and geometry arrays reproduced exactly:', args.output)

if __name__ == '__main__':
    main()
