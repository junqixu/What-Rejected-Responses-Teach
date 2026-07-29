from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from rebuttal.common import add_common_cli_args, configure_logging, ensure_output_dir, load_records, write_json
from rebuttal.identifiability.feature_schema import VALUE_MODES, difference_vector, feature_names
from rebuttal.identifiability.matrix_diagnostics import diagnose_matrix


LOGGER = logging.getLogger(__name__)


def build_matrix(records: list[dict], value_mode: str, orientation: str) -> np.ndarray:
    if not records:
        raise ValueError("no preference records were loaded")
    return np.asarray(
        [difference_vector(row, mode=value_mode, orientation=orientation) for row in records],
        dtype=np.float64,
    )


def save_spectrum_plot(singular_values: list[float], path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return "matplotlib and Pillow are unavailable; design_spectrum.png was not generated"
        width, height, margin = 800, 500, 70
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
        draw.line((margin, margin, margin, height - margin), fill="black", width=2)
        transformed = [float(np.log10(max(value, 1e-15))) for value in singular_values]
        low, high = min(transformed, default=-15.0), max(transformed, default=0.0)
        span = max(high - low, 1.0)
        points = []
        for index, value in enumerate(transformed):
            x = margin + index * (width - 2 * margin) / max(len(transformed) - 1, 1)
            y = height - margin - (value - low) * (height - 2 * margin) / span
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill="#2563eb", width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#1d4ed8")
        draw.text((margin, 22), "Preference design spectrum (log10 scale)", fill="black")
        draw.text((width // 2 - 55, height - 35), "singular-value index", fill="black")
        draw.text((8, margin - 10), f"10^{high:.1f}", fill="black")
        draw.text((8, height - margin - 10), f"10^{low:.1f}", fill="black")
        canvas.save(path)
        return None
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    indices = np.arange(1, len(singular_values) + 1)
    axis.plot(indices, singular_values, marker="o")
    axis.set(xlabel="Singular-value index", ylabel="Singular value", title="Preference design spectrum")
    axis.set_yscale("symlog", linthresh=1e-12)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return None


def run(args: argparse.Namespace) -> dict:
    records = load_records(args.data, args.max_samples)
    matrix = build_matrix(records, args.value_mode, args.orientation)
    names = feature_names(args.value_mode)
    report, nullspace = diagnose_matrix(matrix)
    report.update(
        {
            "orientation": args.orientation,
            "value_mode": args.value_mode,
            "feature_names": names,
            "source_files": [str(Path(path)) for path in args.data],
        }
    )

    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    output = ensure_output_dir(args.out, args.overwrite)
    np.save(output / "design_matrix.npy", matrix)
    np.save(output / "nullspace_basis.npy", nullspace)
    write_json(output / "feature_names.json", names)
    with (output / "singular_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "singular_value"))
        writer.writerows(enumerate(report["singular_values"], 1))
    warning = save_spectrum_plot(report["singular_values"], output / "design_spectrum.png")
    if warning:
        report.setdefault("warnings", []).append(warning)
        LOGGER.warning(warning)
    write_json(output / "matrix_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and diagnose a preference-difference design matrix.")
    parser.add_argument("--data", nargs="+", required=True, help="Preference JSON or JSONL files.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--value_mode", choices=VALUE_MODES, default="binary")
    parser.add_argument("--orientation", choices=("delta", "exposure"), default="delta")
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
