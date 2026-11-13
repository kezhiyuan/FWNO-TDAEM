import argparse
import sys
from pathlib import Path
import yaml
from package_common import CONFIG_ITEM, EVIDENCE, json_write, output_path, raw_files

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description='Train the original clean-27,000 FWNO using an explicit raw-data directory.')
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate paths and configuration without training or writing.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    files = raw_files(args.raw_dir)
    out = output_path(args.output_dir, "training")
    config = yaml.safe_load((EVIDENCE / "original_training_config.yml").read_text(
        encoding="utf-8"))[CONFIG_ITEM]
    config["DATA_PATHS"] = [str(path) for path in files]
    config["output_dir"] = str(out)
    config["warm_start_checkpoint"] = None
    config["epochs"] = args.epochs
    config["cuda_id"] = args.cuda_id
    assert config["nz_earth"] == 20 and config["s"] == [48, 48, 20, 16]
    assert config["split_seed"] == 1234
    print(f"Models: 27000; files: 9; layers: 20; split seed: 1234; output: {out}")
    if args.dry_run:
        print("Dry run passed. No training or files written.")
        return
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "training_config.yml"
    config_path.write_text(yaml.safe_dump({CONFIG_ITEM: config}, sort_keys=False,
                                        allow_unicode=True), encoding="utf-8")
    import HTUFNO_AEM_v6_main as fwno
    result = fwno.main(item=CONFIG_ITEM, config_file=str(config_path))
    json_write(out / "returned_test_metrics.json", result)

if __name__ == "__main__":
    main()
