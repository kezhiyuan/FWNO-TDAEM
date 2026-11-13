from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from paths import DATA, FIGURES
OUT = FIGURES
OUT.mkdir(parents=True, exist_ok=True)
WINDOWS = DATA / 'south_australia_windows.npz'
ORIGINAL_MANIFEST = DATA / 'original_site_figures.json'
EXAMPLE_ID = 199
INK = '#24333D'

plt.rcParams.update({
    'font.family':'Times New Roman', 'font.size':9,
    'axes.titlesize':10, 'axes.labelsize':9, 'axes.linewidth':.65,
    'xtick.labelsize':8.5, 'ytick.labelsize':8.5,
    'axes.labelcolor':INK, 'text.color':INK,
    'xtick.color':INK, 'ytick.color':INK,
    'pdf.fonttype':42, 'svg.fonttype':'none', 'savefig.facecolor':'white',
})


def reconstruct_grid():
    with np.load(WINDOWS) as data:
        sig = data['sig']
        centers = data['centers_local_m']
        x, y, z = (data[name] for name in ('grid_x_edges_local_m', 'grid_y_edges_local_m', 'depth_edges_m'))
        ids = data['model_ids']
    assert sig.shape == (396, 48, 48, 20)
    assert np.array_equal(ids, np.arange(1, 397))
    assert np.all(centers-720 >= 0) and np.all(centers+720 <= [3400, 1600])
    log_rho = np.full((len(x)-1, len(y)-1, len(z)-1), np.nan, dtype=np.float32)
    for n, center in enumerate(centers):
        i, j = np.rint((center-720-[x[0], y[0]])/30).astype(int)
        dest = log_rho[i:i+48, j:j+48]
        populated = np.isfinite(dest)
        assert np.array_equal(dest[populated], -sig[n][populated])
        dest[:] = -sig[n]
    assert np.isfinite(log_rho).all()
    return log_rho, centers, x, y, z


def main():
    previous = next(item for item in json.loads(ORIGINAL_MANIFEST.read_text(encoding='utf-8'))['figures']
                    if item['number'] == 17)
    input_sha = hashlib.sha256(WINDOWS.read_bytes()).hexdigest()
    assert input_sha == previous['source_windows_sha256']
    log_rho, centers, x, y, z = reconstruct_grid()
    rho = 10.**log_rho
    xmid, ymid = (x[:-1]+x[1:])/2, (y[:-1]+y[1:])/2
    center = centers[EXAMPLE_ID-1]
    ix = int(np.searchsorted(x, center[0], side='right')-1)
    iy = int(np.searchsorted(y, center[1], side='right')-1)
    iz = int(np.searchsorted(z, 117., side='right')-1)
    norm = LogNorm(.05, 5000.)
    assert previous['normalization']['vmin'] == norm.vmin
    assert previous['normalization']['vmax'] == norm.vmax
    assert rho.min() >= norm.vmin and rho.max() <= norm.vmax
    assert float(xmid[ix]) == previous['cross_section_x_m']
    assert float(ymid[iy]) == previous['cross_section_y_m']
    cmap = plt.get_cmap('RdBu_r')


    halo = [pe.Stroke(linewidth=1.65, foreground=(.12, .17, .22, .55)), pe.Normal()]
    fig = plt.figure(figsize=(5.72, 4.80))
    ax = fig.add_axes([.66/5.72, 2.22/4.80, 4.84/5.72, (4.84*1600/3400)/4.80])
    mesh = ax.pcolormesh(x, y, rho[:, :, iz].T, shading='flat', cmap=cmap, norm=norm, rasterized=True)
    ax.scatter(*centers.T, s=1.7, facecolors='white', edgecolors='#4A5660',
               alpha=.85, linewidths=.14, zorder=4)
    outline = Rectangle(center-720, 1440, 1440, fill=False, edgecolor='white', linewidth=1.25,
                        linestyle=(0, (5, 3)), zorder=5)
    outline.set_path_effects([pe.Stroke(linewidth=2.1, foreground=(.12,.17,.22,.55)), pe.Normal()])
    ax.add_patch(outline)
    ax.axhline(ymid[iy], color='#212A34', lw=.75, ls=(0, (3, 3)), zorder=5)
    ax.axvline(xmid[ix], color='#212A34', lw=.75, ls=(0, (3, 3)), zorder=5)
    ax.text(center[0]-680, center[1]+650, f'Sample {EXAMPLE_ID}', color='white', fontsize=8.5,
            va='top', bbox={'facecolor':'#27313C','alpha':.7,'edgecolor':'none','pad':1.6}, zorder=6)
    ax.set(xlim=(0,3400), ylim=(0,1600), xlabel='X (m)', ylabel='Y (m)',
           xticks=[0,1000,2000,3000], yticks=[0,400,800,1200,1600])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'(a) Resampled model: depth {z[iz]:g}–{z[iz+1]:g} m', loc='left', pad=3)
    axb = fig.add_axes([.66/5.72, .90/4.80, 2.99/5.72, (2.99*1064/3400)/4.80])
    axc = fig.add_axes([4.095/5.72, .90/4.80, 1.405/5.72, (1.405*1064/1600)/4.80])
    for axis, coord, values, title, xlabel in (
        (axb, x, rho[:, iy, :].T, f'(b) Y = {ymid[iy]:g} m', 'X (m)'),
        (axc, y, rho[ix, :, :].T, f'(c) X = {xmid[ix]:g} m', 'Y (m)'),
    ):
        axis.pcolormesh(coord, z, values, shading='flat', cmap=cmap, norm=norm, rasterized=True)
        axis.set(ylim=(532,0), xlim=(coord[0],coord[-1]), xlabel=xlabel, yticks=[0,200,400])
        line = axis.axhline(485, color='white', lw=.75, ls=(0, (4,3)))
        line.set_path_effects(halo)
        axis.set_aspect(2., adjustable='box')
        axis.set_title(title, loc='left', pad=3)
        axis.tick_params(axis='both', length=2.5, pad=2)
    axb.set_ylabel('Depth (m)')
    axb.set_xticks([0,1000,2000,3000])
    axc.set_xticks([0,800,1600])
    axc.tick_params(labelleft=False)
    cax = fig.add_axes([.20, .425/4.80, .69, .07/4.80])
    bar = fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[.1,1,10,100,1000])
    bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    bar.ax.minorticks_off()
    bar.set_label('Resistivity (Ω·m)', labelpad=3)
    files = {}
    for extension in ('png','pdf','svg'):
        path = OUT/f'fig14_resistivity_blue_red.{extension}'
        fig.savefig(path, dpi=400)
        files[extension] = str(path.resolve())
    plt.close(fig)
    manifest = dict(previous)
    manifest.update(number=14, files=files, png=files['png'], pdf=files['pdf'], svg=files['svg'],
                    palette={'name':'RdBu_r','direction':'low blue to high red',
                             'normalization':'logarithmic','limits_ohm_m':[.05,5000.],
                             'data_smoothing':False,'resistivity_clipping':False},
                    previous_manifest=str(ORIGINAL_MANIFEST.resolve()),
                    retained={'window_id':EXAMPLE_ID, 'model_grid':list(log_rho.shape),
                              'depth_slice_m':[float(z[iz]),float(z[iz+1])],
                              'print_size_in':[5.72,4.80]},
                    annotation_change='Window 199 is labeled Sample 199, and Local X/Y becomes X/Y. Existing outlines, model cells, sections, and the 485 m line are preserved.',
                    layout_change='Canvas height reduced from 5.28 to 4.80 inches. Panels enlarged slightly and title, inter-panel, and colorbar spacing compacted.')
    manifest['caption_zh'] = previous['caption'].replace('窗口 199', '样本 199').replace('窗口199', '样本199') + '颜色由低阻蓝色连续过渡至高阻红色。'
    manifest['caption_en'] = previous['caption_en'].replace('window 199', 'sample 199').replace('Window 199', 'Sample 199') + ' Colors progress continuously from blue at low resistivity to red at high resistivity.'
    (OUT/'fig14_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'files':files,'source_sha256':input_sha,'range_ohm_m':[float(rho.min()),float(rho.max())]},ensure_ascii=False))


if __name__ == '__main__':
    main()
