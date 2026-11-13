from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
DATA = PACKAGE / '03_results/musgrave'
OUTPUT = Path(__file__).resolve().parent / 'out'
FIGURES = OUTPUT / 'figures'
FWNO_SOURCE = PACKAGE / '04_reproduction/fwno'
CHECKPOINT = PACKAGE / '03_results/synthetic/HTUFNO_AEM_v6_clean27000_N16_H7_1_best.pkl'
