from pathlib import Path
import sys
import json
import hashlib

from paths import FIGURES
OUT = FIGURES
import map_cartographic as source


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, locator = source.create_figure('(b) VTEM flight lines')
    fig.set_size_inches(7.55, 4.35, forward=True)
    geology, flight, locator_ax, legend = fig.axes
    geology.set_position([.083, .287, .548, .618])
    flight.set_position([.709, .645, .239, .260])
    locator_ax.set_position([.719, .255, .219, .255])
    legend.set_position([.083, .009, .865, .190])
    geology.set_title('(a) Musgrave geology', loc='left', pad=5, weight='bold')
    flight.set_title('(b) VTEM flight lines', loc='left', pad=5, weight='bold')
    flight.set_xlabel('X (km)')
    flight.set_ylabel('Y (km)')

    geology.text(.022, .900, 'Map center: 129.9009°E, 26.2122°S',
                 transform=geology.transAxes, ha='left', va='top',
                 fontsize=10.8, color=source.SECONDARY, zorder=4)
    files = {}
    for suffix in ('png', 'pdf', 'svg'):
        destination = OUT / f'fig13_musgrave_vtem_flightlines.{suffix}'
        fig.savefig(destination, dpi=450)
        files[suffix] = str(destination.resolve())
    source.plt.close(fig)
    record = {
        'files': files,
        'geology_sha256': hashlib.sha256(source.SOURCE.read_bytes()).hexdigest(),
        'flightlines_sha256': hashlib.sha256(source.ARCHIVE.read_bytes()).hexdigest(),
        'figure_size_in': [7.55, 4.35],
        'previous_figure_size_in': [7.55, 4.68],
        'map_center_lon_lat_deg_gda94': [129.90090372586482, -26.212231238114214],
        'changes': 'Compact layout with the geographic center of panel (a) labeled. The flight axes retain their separate coordinate frame.',
        'preserved': 'Geological polygons, map bounds and scale, flight coordinates, palettes, legend entries and locator data.',
    }
    (OUT / 'fig13_manifest.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(files['png'])


if __name__ == '__main__':
    main()
