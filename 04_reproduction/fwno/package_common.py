from pathlib import Path
from datetime import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
EVIDENCE = PACKAGE / "03_results" / "synthetic"
CHECKPOINT = EVIDENCE / "HTUFNO_AEM_v6_clean27000_N16_H7_1_best.pkl"
CHECKPOINT_SHA256 = "bfee61477e78f5260629b82857501f341a46945da3688e3fcbbde1809d84de97"
CONFIG_ITEM = "HTUFNO_AEM_v5_corrected_config"
MAT_NAMES = [f"test{i}-{i + 3000}TD.mat" for i in range(3000, 30000, 3000)]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def raw_files(directory):
    directory = Path(directory).resolve()
    files = [directory / name for name in MAT_NAMES]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw MAT files: " + "; ".join(missing))
    return files

def output_path(value, kind):
    out = Path(value).resolve() if value else (
        PACKAGE / "05_runs" / (kind + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    for protected in [PACKAGE / "03_results", PACKAGE / "04_reproduction"]:
        if out == protected or protected in out.parents:
            raise ValueError("Runtime outputs must be outside 03_results and 04_reproduction.")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Choose a new or empty output directory: {out}")
    return out
