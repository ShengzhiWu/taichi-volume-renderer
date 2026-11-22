"""Generate the procedural limestone fountain VXB."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Path for the generated VXB")
    args = parser.parse_args()

    from .vxb_export import generate

    generate(args.output)


if __name__ == "__main__":
    main()
