import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from paths import CHECKPOINT, DATA, FWNO_SOURCE, OUTPUT


def coarsen(data, ids, mean, scale):
    result = np.empty((len(ids), 12 * 12 * 20), dtype=np.float32)
    for start in range(0, len(ids), 192):
        selected = ids[start:start+192]
        values = (data[selected] - mean) / scale
        result[start:start+len(selected)] = values.reshape(-1, 12, 4, 12, 4, 20).mean(axis=(2, 4)).reshape(len(selected), -1)
    assert np.isfinite(result).all()
    return result


def main():
    parser = argparse.ArgumentParser(description='Optionally refit the archived joint t-SNE using an external 27,000-input cache.')
    parser.add_argument('--cache', required=True, type=Path,
                        help='External cache_nz20_clean_27000 directory, or its x_nz20_float32.npy')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'tsne_refit')
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    from sklearn import __version__ as sklearn_version
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    from metrics import sha256
    sys.path.insert(0, str(FWNO_SOURCE))
    from HTUFNO_AEM_v6_main import safe_torch_load
    raw_path = args.cache / 'x_nz20_float32.npy' if args.cache.is_dir() else args.cache
    cache_metadata = json.loads(raw_path.with_name('metadata.json').read_text(encoding='utf-8'))
    expected_cache = json.loads((DATA / 'training_input_cache_metadata.json').read_text(encoding='utf-8'))
    file_names = lambda metadata: [str(p).replace('\\', '/').split('/')[-1]
                                   for p in metadata['signature']['paths']]
    assert file_names(cache_metadata) == file_names(expected_cache), 'Training file order differs'
    for key in ('n_total', 'shape_x', 'shape_y', 'times', 'heights', 'file_counts'):
        assert cache_metadata[key] == expected_cache[key], f'Cache metadata differs: {key}'
    for key in ('version', 'sig_key', 'nx', 'ny', 'nz', 'keep_heights'):
        assert cache_metadata['signature'][key] == expected_cache['signature'][key], key
    expected = json.loads((DATA / 'prediction_manifest.json').read_text(encoding='utf-8'))
    assert sha256(args.checkpoint) == expected['checkpoint_sha256']
    split = json.loads((DATA / 'training_split_indices.json').read_text(encoding='utf-8'))
    train_ids = np.sort(np.asarray(split['train_idx']))
    assert len(train_ids) == 21600
    assert len(np.unique(np.r_[train_ids, split['test_idx'], split['val_idx']])) == 27000
    raw = np.load(raw_path, mmap_mode='r')
    assert raw.shape == (27000, 48, 48, 20) and raw.dtype == np.float32
    ckpt = safe_torch_load(str(args.checkpoint), map_location='cpu')
    norm = ckpt['x_normalizer']
    assert int(ckpt['epoch']) == 82 and norm.flag.lower() == 'point'
    mean = np.asarray(norm.mean, dtype=np.float32).reshape(48, 48, 20)
    scale = np.asarray(norm.std, dtype=np.float32).reshape(48, 48, 20) + norm.eps
    del ckpt
    train = coarsen(raw, train_ids, mean, scale)
    with np.load(DATA / 'south_australia_windows.npz', allow_pickle=False) as windows:
        field = coarsen(windows['sig'], np.arange(396), mean, scale)
    with threadpool_limits(limits=args.threads):
        pca = PCA(n_components=50, svd_solver='randomized', random_state=1234).fit(train)
        values = np.vstack((pca.transform(train), pca.transform(field)))
        tsne = TSNE(n_components=2, perplexity=40, max_iter=1200, init='pca',
                    learning_rate='auto', random_state=1234, n_jobs=args.threads, verbose=2)
        embedding = tsne.fit_transform(values)
    assert embedding.shape == (21996, 2) and np.isfinite(embedding).all()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'tsne_coordinates.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'id', 'tsne_1', 'tsne_2'])
        writer.writerows(('synthetic_train', i, *point) for i, point in zip(train_ids, embedding[:21600]))
        writer.writerows(('south_australia', i, *point) for i, point in zip(range(1, 397), embedding[21600:]))
    record = {'sklearn': sklearn_version, 'seed': 1234, 'perplexity': 40, 'pca_components': 50,
              'iterations': 1200, 'threads': args.threads, 'training_models': 21600, 'field_windows': 396,
              'pca_training_variance': float(pca.explained_variance_ratio_.sum()),
              'kl_divergence': float(tsne.kl_divergence_),
              'scope': 'Same normalization, split, pooling, PCA and t-SNE settings. Embedding coordinates can vary with library versions and numerical backends.'}
    (args.output / 'tsne_refit_manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Refitted coordinates saved to {args.output}')


if __name__ == '__main__':
    main()
