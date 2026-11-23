"""Zip result folders and download them to your computer from Colab.

  python scripts/download_outputs.py --dirs <figures dir> <failure_analysis dir> --name outputs
or in a cell:  from scripts.download_outputs import make_zip; ...
The zip is written next to --out_dir. Colab can only start a browser download from a notebook cell
(not from a `!python` subprocess), so after running this, run in a cell:
    from google.colab import files; files.download("<name>.zip")
"""
import argparse, os, shutil, sys, tempfile


def make_zip(dirs, name="outputs", out_dir="."):
    """Copy the existing folders into one staging folder and zip it. Returns the zip path."""
    stage = tempfile.mkdtemp()
    kept = []
    for d in dirs:
        if os.path.isdir(d):
            shutil.copytree(d, os.path.join(stage, os.path.basename(os.path.normpath(d))))
            kept.append(d)
        else:
            print("skipping (not found):", d)
    if not kept:
        raise SystemExit("none of the folders exist")
    path = shutil.make_archive(os.path.join(out_dir, name), "zip", stage)
    print("wrote", path, f"({os.path.getsize(path) / 1e6:.1f} MB) with", kept)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--name", default="outputs"); p.add_argument("--out_dir", default=".")
    a = p.parse_args(); z = make_zip(a.dirs, a.name, a.out_dir)
    try:
        from google.colab import files
        files.download(z)
    except (ImportError, AttributeError):
        # AttributeError: running as a subprocess (`!python ...`), where Colab has no notebook kernel
        print("Zip is ready at", z)
        print("To download it in Colab, run this in a notebook cell:")
        print(f'    from google.colab import files; files.download("{os.path.abspath(z)}")')
