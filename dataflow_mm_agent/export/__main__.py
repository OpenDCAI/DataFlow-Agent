"""CLI: export trajectories to ms-swift messages JSONL."""

import argparse

from .swift import export_swift_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export trajectory JSON/JSONL or pipeline JSONL to ms-swift messages JSONL")
    parser.add_argument("sources", nargs="+", help="Trajectory JSON, TrajectoryStore JSONL, or pipeline JSONL files")
    parser.add_argument("-o", "--output", required=True, help="Output .jsonl")
    parser.add_argument("--image-mode", choices=("file", "base64"), default="file",
                        help="write image files referenced by absolute path (default) or inline data: URIs")
    parser.add_argument("--image-dir", help="Image directory for --image-mode file (default: <output stem>_images/)")
    parser.add_argument("--trajectory-key", default="trajectory", help="Trajectory column in pipeline rows")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    args = parser.parse_args(argv)
    try:
        report = export_swift_jsonl(
            args.sources,
            args.output,
            trajectory_key=args.trajectory_key,
            image_mode=args.image_mode,
            image_dir=args.image_dir,
            force=args.force,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Export failed: {exc}\n")
    print(f"{report.output}: {report.exported} samples, {report.images} images, {len(report.skipped)} skipped")
    for item in report.skipped:
        print(f"  skipped {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
