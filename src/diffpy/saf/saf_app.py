import argparse
from importlib.metadata import metadata

from diffpy.saf.version import __version__  # noqa


def main():
    package_description = metadata("diffpy.saf")["Summary"]
    parser = argparse.ArgumentParser(
        prog="diffpy.saf",
        description=(
            f"{package_description}\n\n"
            "For more information, visit: "
            "https://github.com/diffpy/diffpy.saf/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the program's version number and exit",
    )

    args = parser.parse_args()

    if args.version:
        print(f"diffpy.saf {__version__}")
    else:
        # Default behavior when no arguments are given
        parser.print_help()


if __name__ == "__main__":
    main()
