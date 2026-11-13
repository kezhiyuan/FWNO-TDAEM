from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


from paths import DATA, FIGURES
OUT = FIGURES
OUT.mkdir(parents=True, exist_ok=True)
PREDICTIONS = DATA / 'fwno_predictions_50m.npz'
REFERENCES = DATA / 'FD_predictions_50m'
GATES = (0, 10, 15)
EXAMPLES = (67, 199, 331)
BLUE, INK = '#286A96', '#25333E'
ORANGE, TEAL = '#C47A38', '#288B83'

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.edgecolor': '#6C7881', 'axes.linewidth': .6,
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'stix',
    'savefig.facecolor': 'white', 'axes.unicode_minus': False,
})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arrays():
    with np.load(PREDICTIONS) as src:
        ids = src['model_ids'].copy()
        centers = src['centers_local_m'].copy()
        times = src['times_ms'].copy()
        pred = src['amplitude_nT_per_s'].astype(np.float64)
        assert float(src['height_m']) == 50.
    np.testing.assert_array_equal(ids, np.arange(1, 397))
    assert pred.shape == (396, 16) and centers.shape == (396, 2)
    assert times.shape == (16,) and np.all(np.diff(times) > 0)
    paths = [REFERENCES / f'dBzdt_predict_all_{i}.txt' for i in ids]
    reference = np.stack([np.loadtxt(path) for path in paths])
    assert reference.shape == pred.shape
    assert np.isfinite(reference).all() and np.all(reference > 0)
    assert np.isfinite(pred).all() and np.all(pred > 0)
    error = 100. * np.abs(pred-reference) / reference
    return ids, centers, times, pred, reference, error, paths


def save(fig, name):
    files = {}
    for ext in ('png', 'pdf', 'svg'):
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, dpi=400)
        files[ext] = str(path.resolve())
    plt.close(fig)
    return files


def maps(ids, centers, times, pred, reference, error, reference_label):
    xx, yy = (np.unique(centers[:, k]) for k in (0, 1))
    np.testing.assert_array_equal(centers, np.array([(x,y) for x in xx for y in yy]))
    xe, ye = (np.r_[axis-15, axis[-1]+15] for axis in (xx,yy))
    fig = plt.figure(figsize=(5.72, 6.6))
    lefts, widths = (.09, .365, .745), (.235, .235, .17)
    bottoms, height = (.66, .365, .07), .26
    error_vmax = float(np.ceil(error[:, GATES].max()))
    headers = [reference_label.replace(' ', '\n') if reference_label == 'Supplied reference' else reference_label,
               'FWNO', 'Relative error']
    for x, width, title in zip(lefts, widths, headers):
        fig.text(x+width/2, .978, title, ha='center', va='top', fontsize=10)
    color_limits = []
    for row, gate in enumerate(GATES):
        vmin = min(reference[:,gate].min(), pred[:,gate].min())
        vmax = max(reference[:,gate].max(), pred[:,gate].max())
        norm = LogNorm(vmin, vmax)
        color_limits.append([float(vmin), float(vmax)])
        for col, (values, cmap, nrm) in enumerate(((reference, 'viridis', norm),
                                                 (pred, 'viridis', norm),
                                                 (error, 'Blues', Normalize(0, error_vmax)))):
            ax = fig.add_axes([lefts[col], bottoms[row], widths[col], height])
            image = ax.pcolormesh(xe, ye, values[:,gate].reshape(len(xx),len(yy)).T,
                                 shading='flat', cmap=cmap, norm=nrm, rasterized=True)
            ax.set_xlim(xe[[0,-1]])
            ax.set_ylim(ye[[0,-1]])
            ax.set_xticks([1000,1800,2600])
            ax.set_yticks([725,800,875])
            ax.tick_params(length=2.2, pad=2, labelbottom=row == 2, labelleft=col == 0)
            if col == 0:
                ax.set_ylabel('Y (m)', labelpad=2)
            if row == 2:
                ax.set_xlabel('X (m)', labelpad=3)
            ax.set_title(f'({chr(97+row*3+col)})' + (f' {times[gate]:.3f} ms' if col == 0 else ''),
                         loc='left', pad=2)
            if row == 0 and col == 0:
                for model_id in EXAMPLES:
                    point = centers[model_id-1]
                    ax.scatter(*point, s=17, marker='^', facecolor='white', edgecolor=INK,
                               linewidth=.4, zorder=5)
                    ax.annotate(str(model_id), point, xytext=(0,8), textcoords='offset points',
                                ha='center', va='bottom', color='white', fontsize=8.5,
                                bbox={'facecolor': '#24333D', 'alpha': .65,
                                      'edgecolor': 'none', 'pad': .5}, zorder=6)
            if col == 1:
                amp_image = image
            elif col == 2:
                error_image = image

        cax = fig.add_axes([.620, bottoms[row], .015, height])
        ticks = np.geomspace(vmin, vmax, 3)
        exponent = int(np.floor(np.log10(vmax)))
        factor = 10.**exponent
        bar = fig.colorbar(amp_image, cax=cax, orientation='vertical', ticks=ticks,
                          format=FuncFormatter(lambda value, _, f=factor: f'{value/f:.2g}'))
        bar.minorticks_off()
        bar.ax.tick_params(length=2, pad=2)
        units = 'nT/s' if exponent == 0 else f'$10^{{{exponent}}}$ nT/s'
        bar.ax.text(.5,1.025,units,transform=bar.ax.transAxes,
                    ha='center',va='bottom',fontsize=8.5)
        cax = fig.add_axes([.932, bottoms[row], .015, height])
        bar = fig.colorbar(error_image, cax=cax, orientation='vertical', ticks=[0,error_vmax/2,error_vmax])
        bar.minorticks_off()
        bar.ax.tick_params(length=2, pad=2)
        bar.ax.text(.5,1.025,'%',transform=bar.ax.transAxes,
                    ha='center',va='bottom',fontsize=8.5)
    files = save(fig, 'fig16_fd_fwno_maps_50m')
    caption_zh = (
        '南澳模型396个窗口在50 m高度下的响应空间对比。三行依次为7.534、8.948和17.493 ms，'
        '三列依次为FD响应、FWNO预测和逐点相对误差。同一时间道的两幅响应图共用右侧的对数幅值色标，'
        f'误差图使用统一的蓝色顺序色标，范围为0–{error_vmax:g}%，无色标截断。三角形标出样本67、199和331。'
        '图中单元表示窗口中心的30 m间隔，纵横显示比例不同；坐标为模型局部坐标。'
        'FD响应读取自本轮更新的FD_predictions_50m目录。')
    caption_en = (
        'Spatial comparison for the 396 South Australia model windows at a height of 50 m. '
        'Rows correspond to 7.534, 8.948, and 17.493 ms; columns show FD responses, '
        'FWNO predictions, and pointwise relative errors. The response pair at each time gate shares a '
        'logarithmic amplitude scale to their right; error maps use a common sequential blue scale '
        f'from 0 to {error_vmax:g}%, without clipping. Triangles locate '
        'samples 67, 199, and 331. Cells represent the 30 m spacing between window centers; horizontal '
        'and vertical display scales differ. Coordinates are local to the model. Reference responses '
        'were read from the updated FD_predictions_50m directory.')
    return {'number':16, 'files':files, 'caption':caption_zh, 'caption_en':caption_en,
            'gate_indices_zero_based':list(GATES), 'gate_times_ms':times[list(GATES)].tolist(),
            'response_color_limits_nT_per_s':color_limits,
            'error_color_limits_percent':[0,error_vmax], 'error_colormap':'Blues',
            'clipped_values':int(np.sum(error[:,GATES]>error_vmax)),
            'layout':{'canvas_inches':[5.72,6.6], 'panel_height_inches':1.716,
                      'inter_row_gap_inches':.231, 'title_pad_points':2,
                      'previous_panel_height_inches':1.386,
                      'previous_inter_row_gap_inches':.594}}


def curves(ids, centers, times, pred, reference, error, reference_label):
    fig, axes = plt.subplots(3,2,figsize=(5.72,6.35),sharex=True)
    fig.subplots_adjust(left=.12, right=.975, bottom=.075, top=.91, hspace=.18, wspace=.44)
    handles = [Line2D([],[],color=INK,lw=1.2,marker='o',markersize=3.1,
                      markerfacecolor='white',markeredgewidth=.8,label=reference_label),
               Line2D([],[],color=BLUE,lw=1.1,ls='--',marker='x',markersize=3.7,label='FWNO'),
               Line2D([],[],color=ORANGE,lw=1.0,ls='--',label='5%'),
               Line2D([],[],color=TEAL,lw=1.0,ls=':',label='10%')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54,.99), ncol=4,
               frameon=False, columnspacing=1.7, handlelength=2.0, labelspacing=.4)
    per_example = []
    for row, model_id in enumerate(EXAMPLES):
        idx = int(np.flatnonzero(ids == model_id).item())
        left, right = axes[row]
        left.plot(times,reference[idx], color=INK,lw=1.2,marker='o',markersize=3.1,
                  markerfacecolor='white',markeredgewidth=.8,zorder=2)
        left.plot(times,pred[idx], color=BLUE,lw=1.1,ls='--',marker='x',markersize=3.7,
                  markeredgewidth=.85,zorder=3)
        left.set(yscale='log', ylabel='Amplitude (nT/s)', ylim=(3e-4,3.2))
        left.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
        left.yaxis.set_minor_formatter(NullFormatter())
        left.set_title(f'({chr(97+2*row)}) Sample {model_id}',loc='left',pad=2.5)
        right.plot(times,error[idx],color=BLUE,lw=1.15,marker='o',markersize=3.1,
                   markeredgewidth=0,zorder=3)
        right.axhline(5,color=ORANGE,ls='--',lw=1)
        right.axhline(10,color=TEAL,ls=':',lw=1)
        right.set(ylim=(0,11),yticks=[0,2,5,8,10],ylabel='Relative error (%)')
        right.set_title(f'({chr(98+2*row)}) MRE = {error[idx].mean():.2f}%',loc='left',pad=2.5)
        for ax in (left,right):
            ax.set_xlim(7.3,17.75)
            ax.set_xticks([8,10,12,14,16])
            ax.grid(axis='y',color='#DCE3E7',lw=.5)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(length=2.5,pad=2)
        per_example.append({'model_id':model_id,'center_local_m':centers[idx].tolist(),
                            'mre_percent':float(error[idx].mean()),
                            'maximum_relative_error_percent':float(error[idx].max())})
    for ax in axes[-1]: ax.set_xlabel('Time (ms)')
    files = save(fig,'fig17_fd_fwno_curves_50m')
    caption_zh = (
        '南澳样本67、199和331在50 m高度下的时间响应及相对误差，各行对应一个模型窗口。'
        '左列为FD响应与FWNO预测，幅值采用对数坐标；右列为16个时间道的逐点相对误差，'
        '橙色虚线和绿色点线分别表示5%与10%阈值，标题给出平均相对误差（MRE）。'
        '三个窗口的局部X坐标依次为1055、1715和2375 m，Y均为725 m；位置见图16。'
        '时间轴线性表示窗口开启时刻，响应为相应接收窗口的平均幅值。参考数据来源说明同图16。')
    caption_en = (
        'Time responses and relative errors for South Australia samples 67, 199, and 331 at a height of 50 m, '
        'with one model window per row. The left column compares FD responses with FWNO '
        'predictions on a logarithmic amplitude axis. The right column shows pointwise relative errors '
        'at the 16 time gates; orange dashed and green dotted lines mark 5% and 10%, respectively, and '
        'titles report the mean relative error (MRE). Local X coordinates are 1055, 1715, and 2375 m, '
        'respectively, with Y = 725 m; locations are shown in Figure 16. The linear time axis gives gate '
        'opening times, and responses are amplitudes averaged over the corresponding receiver windows. '
        'Reference-data provenance is as described in Figure 16.')
    return {'number':17,'files':files,'caption':caption_zh,'caption_en':caption_en,
            'example_models':per_example,'error_guide_lines_percent':[5,10],
            'layout':{'canvas_inches':[5.72,6.35],
                      'panel_height_inches':(.91-.075)/(3+2*.18)*6.35,
                      'inter_row_gap_inches':(.91-.075)/(3+2*.18)*6.35*.18,
                      'title_pad_points':2.5,
                      'previous_panel_height_inches':(.89-.08)/(3+2*.36)*6.35,
                      'previous_inter_row_gap_inches':(.89-.08)/(3+2*.36)*6.35*.36}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-label',default='FD')
    args=parser.parse_args()
    ids,centers,times,pred,reference,error,paths=load_arrays()
    figures=[maps(ids,centers,times,pred,reference,error,args.reference_label),
             curves(ids,centers,times,pred,reference,error,args.reference_label)]
    report={'figures':figures,'reference_label':args.reference_label,
            'reference_provenance':'Updated user-supplied FD_predictions_50m files; filenames and SHA-256 hashes bind the exact source used here. No FD solver was run by this plot script.',
            'response_unit':'nT/s','height_m':50.,'model_count':396,'response_count':6336,
            'mre_percent':float(error.mean()),'maximum_relative_error_percent':float(error.max()),
            'fraction_below_5_percent':float(np.mean(error<5)),
            'fraction_below_10_percent':float(np.mean(error<10)),
            'prediction_file':str(PREDICTIONS.resolve()),'prediction_sha256':sha(PREDICTIONS),
            'reference_files':[{'model_id':int(i),'path':str(p.resolve()),'sha256':sha(p)}
                               for i,p in zip(ids,paths)],
            'scope':'Existing response visualization only; no new simulation, training, inference or interpolation.'}
    target=OUT/'comparison_figures_manifest.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'manifest':str(target.resolve()),'pngs':[x['files']['png'] for x in figures]},ensure_ascii=False))


if __name__=='__main__': main()
