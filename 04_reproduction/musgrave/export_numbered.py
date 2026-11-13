import argparse
from pathlib import Path
import zipfile

import numpy as np

from paths import DATA, OUTPUT


def main():
    parser = argparse.ArgumentParser(description='Export exact archived FD input models and regenerate numbered FWNO responses.')
    parser.add_argument('--output', type=Path, default=OUTPUT / 'numbered_export')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA / 'FD_input_models_396.zip') as archive:
        expected = {f'AEM_TD_Field_model/AEM_model_{i}.m' for i in range(1, 397)}
        assert set(archive.namelist()) == expected
        for name in sorted(expected):
            dest = args.output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
    response_dir = args.output / 'FWNO_predictions_50m'
    response_dir.mkdir(exist_ok=True)
    with np.load(DATA / 'fwno_predictions_50m.npz', allow_pickle=False) as source:
        for model_id, response in zip(source['model_ids'], source['amplitude_nT_per_s']):
            target = response_dir / f'dBzdt_predict_all_{model_id}.txt'
            np.savetxt(target, response, fmt='%.9g')
            np.testing.assert_array_equal(np.loadtxt(target, dtype=np.float32), response)
    (args.output / 'time_gates_50m.csv').write_bytes((DATA / 'time_gates_50m.csv').read_bytes())
    print(f'Exported 396 exact FD input models and 396 FWNO response files to {args.output}')


if __name__ == '__main__':
    main()
