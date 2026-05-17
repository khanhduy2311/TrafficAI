import argparse

import config as cfg
from pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic speed violation detection")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--output", default="output", help="Directory for output files")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override speed limit in km/h",
    )
    parser.add_argument(
        "--ppm-center",
        type=float,
        default=None,
        help=f"Override PPM_CENTER (default: {cfg.PPM_CENTER})",
    )
    parser.add_argument(
        "--ppm-edge",
        type=float,
        default=None,
        help="Reserved for future tuning",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.limit is not None:
        cfg.DEFAULT_SPEED_LIMIT = args.limit
        print(f"Override speed limit: {args.limit} km/h")

    if args.ppm_center is not None:
        cfg.PPM_CENTER = args.ppm_center
        print(f"Override PPM_CENTER: {args.ppm_center}")

    if args.ppm_edge is not None:
        print(f"Received --ppm-edge={args.ppm_edge}, but this setting is not used yet.")

    run(source=args.source, output_dir=args.output)
