from pathlib import Path
import os

import json

import hashlib

from collections import defaultdict

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.path import Path as MplPath

from matplotlib.patches import PathPatch, Patch

from matplotlib.ticker import FuncFormatter

from shapely.geometry import shape, box

from shapely.geometry.polygon import orient

from shapely.ops import unary_union

from paths import DATA, FIGURES

HERE = FIGURES

SOURCE = DATA / 'geology/gssa_musgrave_geologic_units_epsg28352.geojson'

ARCHIVE = Path(os.environ.get('FWNO_MUSGRAVE_FLIGHTLINES', DATA / 'source_models/iso.imp'))

BBOX = (580000., 7094000., 600000., 7107000.)

INK = '#283C46'

BLUE = '#386D88'

UNITS = {
    'Holocene aeolian sediments': ('Aeolian sand', '#F4ECD1'),
    'Holocene alluvial/fluvial sediments': ('Alluvium', '#D8E6DE'),
    'Pleistocene regolith/colluvial unit 5': ('Colluvial sand', '#DCC4A8'),
    'Pleistocene calcrete': ('Calcrete', '#DDE2AE'),
    'Tertiary ferricrete': ('Ferricrete', '#D3A06C'),
    'Holocene lacustrine/playa unit 3': ('Playa sediments', '#CBDDE9'),
    'Pitjantjatjara Supersuite': ('Granitoids (Pitjantjatjara)', '#DDAAA4'),
    'Giles Complex unit 2': ('Gabbroic rocks (Giles)', '#9F8EBD'),
    'Mesoproterozoic unit 23': ('Doleritic dykes', '#786A93'),
    'Birksgate Complex': ('Gneiss (Birksgate)', '#9EB4AB'),
    'Birksgate Complex unit 3': ('Granulite (Birksgate)', '#739397'),
    'Birksgate Complex unit 6': ('Quartzite (Birksgate)', '#BBC8D6'),
}

plt.rcParams.update({
    'font.family':'Times New Roman', 'font.size':9,
    'axes.titlesize':10, 'axes.labelsize':9,
    'xtick.labelsize':8.5, 'ytick.labelsize':8.5,
    'axes.linewidth':.65, 'axes.labelcolor':INK, 'text.color':INK,
    'xtick.color':INK, 'ytick.color':INK, 'pdf.fonttype':42,
    'svg.fonttype':'none', 'savefig.facecolor':'white',
})

def polygons(geom):
    if geom.geom_type == 'Polygon':
        yield geom
    elif hasattr(geom, 'geoms'):
        for part in geom.geoms:
            yield from polygons(part)

def patch(poly, color):
    poly = orient(poly, sign=1.)
    vertices, codes = [], []
    for ring in (poly.exterior, *poly.interiors):
        points = np.asarray(ring.coords) / 1000.
        vertices.extend(points)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-2) + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(vertices, codes), facecolor=color,
                     edgecolor='#90958C', linewidth=.38, joinstyle='round')
