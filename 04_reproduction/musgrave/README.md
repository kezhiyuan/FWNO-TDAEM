# Musgrave：50 m 正演结果复现

在 `FWNO-TDAEM` 根目录中运行：

```powershell
python 04_reproduction/musgrave/reproduce.py
```

这是默认入口，仅使用CPU。程序核对封装数据的SHA256，复算396个窗口、16个时间道的误差，并重绘论文图件。图13需要原研究的 `iso.imp` 文件。取得文件后，将环境变量 `FWNO_MUSGRAVE_FLIGHTLINES` 设为该文件路径。新结果位于 `04_reproduction/musgrave/out/`。

仅核对数据和误差：

```powershell
python 04_reproduction/musgrave/reproduce.py --metrics-only
```

依赖：Python 3.10及以上、NumPy、Matplotlib、Shapely。图件优先使用Times New Roman。缺少该字体时Matplotlib可使用替代字体，排版可能略有变化。验证环境版本见同目录 `environment_versions.json`。

## 复现内容

| 入口 | 用途 | 默认输出 |
|---|---|---|
| `reproduce.py` | 哈希核对、指标复算、论文图重绘 | `out/` |
| `rebuild_windows.py` | 从原始电阻率模型重新构建396个输入窗口 | `out/south_australia_windows.npz` |
| `export_numbered.py` | 解出396份原始FD输入，并生成逐模型FWNO响应文本 | `out/numbered_export/` |
| `predict_50m.py` | 使用封装权重执行可选CUDA推理和计时 | `out/prediction_rerun/` |
| `rebuild_tsne.py` | 使用外部训练输入缓存重新拟合PCA与t-SNE | `out/tsne_refit/` |

`draw_fig13_geology.py`、`draw_fig14_model.py`、`draw_fig15_tsne.py`及`draw_fig16_17_responses.py`也可分别运行。对应地质与测线图、电阻率模型图、t-SNE分布图、FD/FWNO响应图和曲线误差图。默认五图使用归档预测与归档t-SNE坐标，不重新训练网络或拟合嵌入。

## 输入窗口与FD编号

```powershell
python 04_reproduction/musgrave/rebuild_windows.py
python 04_reproduction/musgrave/export_numbered.py
```

原始 `iso.rho` 与裁剪后的 `iso-new.rho` 均保存在 `03_results/musgrave/source_models/`。重建先核实移除两水平方向各端4个网格的对应关系，再按单元交叠体积对对数电阻率重采样。窗口输入存储 **log10电导率**，尺寸为 `396 × 48 × 48 × 20`，模型编号为1–396。深度范围为0–532 m。程序逐元素核对全部输入及几何数组是否与归档一致。

`FD_input_models_396.zip`保存实际导出的396份FD输入模型，`FWNO_numbered_predictions_396.zip`保存用户要求导出的396份FWNO编号文本响应，两者解压后均逐字节保持原文件。`export_numbered.py`从归档NPZ重新生成396份FWNO文本响应，并验证读回值。`time_gates_50m.csv`给出时间道顺序、开启时间（ms）、高度（50 m）和响应单位（nT/s）。

## 可选GPU预测

```powershell
python 04_reproduction/musgrave/predict_50m.py
```

需要PyTorch及可用CUDA。默认读取 `04_reproduction/fwno/` 中的共享源代码。请另行获取epoch82权重并放到 `03_results/synthetic/HTUFNO_AEM_v6_clean27000_N16_H7_1_best.pkl`。脚本通过SHA256核对权重和窗口数据。输入/输出归一化、架构和时间高度元数据均来自该权重。

默认batch为8，预热3批、重复计时3次。网络联合计算7个高度，再取50 m输出。计时包括归一化、数据传输、推理、逆归一化、幅值还原及返回主机，不包括磁盘读档和权重加载。归档三次总耗时为2.962236、2.516929及2.605787 s，中位数为2.605787 s。新环境的耗时与数值差异另存于新结果目录，不替换历史记录。

可设置 `--checkpoint`、`--windows`、`--output`、`--batch-size`、`--repetitions` 和 `--device`。此入口只接受与归档哈希一致的权重与输入，用于严格复现本算例。

## 可选t-SNE重拟合

```powershell
python 04_reproduction/musgrave/rebuild_tsne.py --cache "外部缓存目录"
```

外部缓存目录需含 `x_nz20_float32.npy` 和 `metadata.json`，与原 `cache_nz20_clean_27000`一致。约4.98 GB的训练输入未重复打包。脚本核对样本数、文件顺序和网格元数据，并使用封装的训练划分与epoch82归一化。仅对21,600个训练模型拟合50维PCA，再将训练模型与396个测区窗口联合嵌入。随机种子1234、困惑度40、迭代1200次。需额外安装PyTorch、PyYAML、scikit-learn及threadpoolctl。

重拟合结果单独保存。不同库版本与数值后端可能使t-SNE坐标变化，论文图15使用封装的原始坐标。当前复现代码不依赖历史论文阶段目录或Athabasca分析程序。

## 结果和来源

复算应得到平均相对误差 **2.1735428214297796%**、最大误差 **4.9746626717424345%**。6,336个响应点全部低于5%，396个窗口的全部16道均低于5%。误差定义为 `100 × abs(FWNO − FD) / FD`。

396个窗口来自同一Musgrave反演模型，彼此重叠。FD文本由用户提供，包内保留原始幅值、逐文件校验值与模型编号。这里可复现配对统计，未提供相应FD求解器配置和运行日志，因此此包不重新求解FD，也不测量FD运行时间。

地质矢量及来源许可元数据保存在 `03_results/musgrave/geology/`。GSSA地质资料采用CC BY 4.0，定位图来源为Natural Earth。关于原研究测线文件的获取，请参阅根目录 `DATA_SOURCES.md`。区域地质、档案测线与模型分别使用各自的坐标框架。

归档数组和FD参考幅值保留原始内容。历史配置及记录使用便携路径。`package_manifest.json`列出封装文件及哈希。`validation_report.json`记录原归档的CPU检查。

两个合成反演算例的数据见 `03_results/inversion_examples/`。
