import sys
import argparse
import os
from pathlib import Path

HERE = Path(__file__).parent.resolve()
os.chdir(HERE)
sys.path.insert(0, str(HERE / "src"))

from loader import load_config, load_artifacts
from tables import (
    build_per_class_table,
    macro_summary_table,
    best_and_worst_classes,
    epoch_history_table,
)
from visualizer import (
    plot_training_curves,
    plot_per_class_metrics_bar,
    plot_confusion_matrix,
    plot_per_class_metric_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze DenseNet-121 training results")
    parser.add_argument("--config", default="configs/analysis_config.yaml")
    parser.add_argument(
        "--metric",
        default="f1",
        help="Metric used to rank classes in summary (default: f1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    acfg = load_config(args.config)
    tcfg = load_config(acfg.training_config)
    class_names = tcfg.data.classes
    out = acfg.output_dir

    print("=" * 60)
    print("Loading training artifacts (no inference)...")
    art = load_artifacts(acfg)
    print(f"  Epochs in history: {len(art.history)}")
    print(f"  Classes: {class_names}")
    print("=" * 60)

    build_per_class_table(art.best, class_names, out)
    macro_summary_table(art.best, out)
    best_and_worst_classes(art.best, class_names, out, metric=args.metric)
    epoch_history_table(acfg.metrics_csv, out)

    plot_training_curves(art.history, out)
    plot_per_class_metrics_bar(art.best, class_names, out)
    plot_confusion_matrix(art.confusion, class_names, out)
    plot_per_class_metric_comparison(art.best, class_names, out)

    print("=" * 60)
    print(f"Analysis complete. All outputs saved to: {Path(out).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
