"""Execute the committed M3 notebook from a fresh kernel, retaining output outside Git."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run every cell; errors propagate to CI instead of being embedded and ignored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/m3-executed.ipynb")
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / "notebooks/01_data_quality.ipynb", as_version=4)
    NotebookClient(
        notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)


if __name__ == "__main__":
    main()
