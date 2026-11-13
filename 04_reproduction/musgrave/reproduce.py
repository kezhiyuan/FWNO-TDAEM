import argparse
import os
from pathlib import Path
import subprocess
import sys

from metrics import run
from paths import DATA, OUTPUT


def main():
    parser = argparse.ArgumentParser(description='CPU reproduction: verify the archive, recompute errors and redraw Figures 13–17.')
    parser.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args()
    run()
    if not args.metrics_only:
        here = Path(__file__).resolve().parent
        flightlines = Path(os.environ.get('FWNO_MUSGRAVE_FLIGHTLINES', DATA / 'source_models/iso.imp'))
        for script in ('draw_fig13_geology.py', 'draw_fig14_model.py',
                       'draw_fig15_tsne.py', 'draw_fig16_17_responses.py'):
            if script == 'draw_fig13_geology.py' and not flightlines.is_file():
                print('Geology and flight-line figure requires the original iso.imp. Set FWNO_MUSGRAVE_FLIGHTLINES to its location.')
                continue
            subprocess.run([sys.executable, str(here / script)], check=True)
    print(f'Reproduction complete: {OUTPUT}')


if __name__ == '__main__':
    main()
