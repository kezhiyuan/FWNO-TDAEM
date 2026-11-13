import json

from pathlib import Path

import sys

import matplotlib.patheffects as effects

from matplotlib.patches import Polygon, Rectangle

from matplotlib.transforms import Affine2D

import numpy as np

from shapely.geometry import box, shape

from shapely.ops import unary_union

from paths import DATA, FIGURES

HERE = DATA / 'geology'

PREVIOUS = HERE

from map_base import ARCHIVE, BBOX, SOURCE, UNITS, patch, plt, polygons

INK = '#253943'

SECONDARY = '#50626a'

LINE = '#637880'

FLIGHT = '#20647b'

HIGHLIGHT = '#9c4d3e'

UNITS_STYLE = dict(zip(UNITS, [
    ('Aeolian sand', '#f6eddb'), ('Alluvium', '#afd0c0'),
    ('Colluvial sand', '#d9bc91'), ('Calcrete', '#dcd890'),
    ('Ferricrete', '#c88758'), ('Playa sediments', '#b9d8e4'),
    ('Pitjantjatjara granitoids', '#d0928e'), ('Giles gabbroic rocks', '#9b7eb3'),
    ('Doleritic dykes', '#5d5b77'), ('Birksgate gneiss', '#779e91'),
    ('Birksgate granulite', '#537a86'), ('Birksgate quartzite', '#a4b4cc'),
]))

def plot_geology(ax):
    src = json.loads(SOURCE.read_text(encoding='utf-8'))
    assert src['crs']['properties']['name'].endswith('28352')
    bounds = box(*BBOX)
    clipped = []
    for feature in src['features']:
        name = feature['properties']['name']
        geom = shape(feature['geometry']).intersection(bounds)
        if geom.is_empty:
            continue
        clipped.append(geom)
        for polygon in polygons(geom):
            symbol = patch(polygon, UNITS_STYLE[name][1])
            symbol.set_edgecolor('#6d6259')
            symbol.set_linewidth(.42)
            ax.add_patch(symbol)
    assert len(clipped) == 37
    np.testing.assert_allclose(unary_union(clipped).area, bounds.area, atol=.01, rtol=0)

    ax.set(xlim=(580, 600), ylim=(7094, 7107), xlabel='Easting (km)',
           ylabel='Northing (km)', xticks=[580, 585, 590, 595, 600],
           yticks=[7094, 7097, 7100, 7103, 7106])
    ax.set_aspect('equal', adjustable='box')
    ax.ticklabel_format(axis='both', useOffset=False, style='plain')
    ax.grid(color='white', linewidth=.55, alpha=.66)
    ax.set_axisbelow(False)
    ax.tick_params(top=True, right=True, labeltop=False, labelright=False)
    ax.set_title('(a) Musgrave geology', loc='left', pad=8, weight='bold')

    ax.add_patch(Polygon([(598.85, 7105.78), (598.56, 7104.93), (598.85, 7105.11)],
                         facecolor=INK, edgecolor=INK, linewidth=.6, zorder=4))
    ax.add_patch(Polygon([(598.85, 7105.78), (599.14, 7104.93), (598.85, 7105.11)],
                         facecolor='white', edgecolor=INK, linewidth=.6, zorder=4))
    ax.text(598.85, 7105.96, 'N', ha='center', va='bottom', weight='bold', zorder=4)

    sx, sy = 588.8, 7094.69
    for offset, face in [(0, INK), (2.5, 'white')]:
        ax.add_patch(Rectangle((sx + offset, sy), 2.5, .13, facecolor=face,
                               edgecolor=INK, linewidth=.55, zorder=4))
    halo = [effects.withStroke(linewidth=2.3, foreground='white')]
    for x, label in [(sx, '0'), (sx + 5, '5 km')]:
        ax.text(x, sy + .24, label, ha='center', va='bottom', fontsize=9.5,
                path_effects=halo, zorder=4)
    ax.text(.022, .958, 'GDA94 / MGA zone 52', transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color=SECONDARY, zorder=4)

def plot_flight_lines(ax, title):
    archive = np.loadtxt(ARCHIVE)
    assert archive.shape == (210, 29)
    np.testing.assert_array_equal(archive[:, 0], np.arange(1, 211))
    for index, data in enumerate(np.split(archive, 7), start=1):
        x = (data[:, 1] - 584000.) / 1000.
        y = data[:, 2] / 1000.
        ax.plot(x, y, color=FLIGHT, linewidth=.66, marker='o', markersize=1.75,
                markeredgewidth=0, zorder=3)
        end = np.argmax(x)
        ax.text(x[end] + .095, y[end], f'L{index}', fontsize=8.7, color=FLIGHT,
                ha='left', va='center')
    ax.set(xlim=(-.08, 4.27), ylim=(-1.32, .12), xticks=[0, 1, 2, 3, 4],
           yticks=[-1.2, -.8, -.4, 0], xlabel='Local X (km)', ylabel='Local Y (km)')
    ax.tick_params(labelsize=9)
    ax.grid(color='#d8e4e6', linewidth=.51, alpha=.85)
    ax.set_axisbelow(True)
    ax.set_title(title, loc='left', pad=8, weight='bold')

def plot_locator(ax, locator):
    features = {f['properties']['name']: f for f in locator['features']}
    to_degrees = Affine2D().scale(1000) + ax.transData
    for name, face in [('Australia', '#e7ecea'), ('South Australia', '#b2c9bc')]:
        for poly in polygons(shape(features[name]['geometry'])):
            symbol = patch(poly, face)
            symbol.set_transform(to_degrees)
            symbol.set_edgecolor(LINE)
            symbol.set_linewidth(.49)
            ax.add_patch(symbol)

    lon, lat = locator['map_center_lonlat']
    ax.scatter(lon, lat, s=30, color=HIGHLIGHT, edgecolor='white', linewidth=.8, zorder=5)
    ax.annotate('Musgrave', (lon, lat), xytext=(137, -22.8), fontsize=10,
                color=HIGHLIGHT, weight='bold', ha='left', va='center',
                arrowprops={'arrowstyle': '-', 'color': HIGHLIGHT, 'lw': .7})
    ax.text(135.5, -31.6, 'SA', ha='center', va='center', fontsize=9, color=INK)
    ax.set(xlim=(111.5, 155.4), ylim=(-44.7, -10.2))
    ax.set_aspect(1 / np.cos(np.deg2rad(27)), adjustable='box')
    ax.set_axis_off()
    ax.set_title('Location of panel (a)', loc='left', fontsize=10, pad=3)

def add_legend(fig):
    units = list(UNITS_STYLE.values())
    legend = fig.add_axes([.083, .025, .865, .17])
    legend.set_axis_off()
    groups = [(0, 0, .215, 'Surficial units'), (6, .48, .274, 'Bedrock units')]
    for start, x, column_step, title in groups:
        legend.text(x, .91, title, size=10.8, weight='bold', transform=legend.transAxes)
        for index, (name, color) in enumerate(units[start:start + 6]):
            column, row = divmod(index, 3)
            left = x + column_step * column
            y = .68 - .26 * row
            legend.add_patch(Rectangle((left, y - .074), .034, .14,
                                       facecolor=color, edgecolor='#6d6259', linewidth=.48,
                                       transform=legend.transAxes))
            legend.text(left + .045, y, name, ha='left', va='center', size=10.25,
                        transform=legend.transAxes)

def create_figure(flight_title='(b) Archived flight lines'):
    locator = json.loads((HERE / 'australia_locator.geojson').read_text(encoding='utf-8'))
    plt.rcParams.update({'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': LINE,
                         'axes.linewidth': .63, 'xtick.color': INK, 'ytick.color': INK,
                         'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10.4,
                         'xtick.labelsize': 9.6, 'ytick.labelsize': 9.6,
                         'xtick.major.size': 3, 'ytick.major.size': 3,
                         'xtick.major.width': .65, 'ytick.major.width': .65,
                         'xtick.major.pad': 3, 'ytick.major.pad': 3,
                         'svg.fonttype': 'path'})
    fig = plt.figure(figsize=(7.55, 4.68))
    plot_geology(fig.add_axes([.083, .321, .548, .575]))
    plot_flight_lines(fig.add_axes([.709, .663, .239, .233]), flight_title)
    plot_locator(fig.add_axes([.719, .275, .219, .245]), locator)
    add_legend(fig)
    return fig, locator
