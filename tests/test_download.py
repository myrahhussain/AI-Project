import os, sys, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from download_outputs import make_zip


def test_make_zip_collects_existing_folders_and_skips_missing(tmp_path):
    a = tmp_path / "figures"; a.mkdir(); (a / "x.png").write_bytes(b"png")
    z = make_zip([str(a), str(tmp_path / "missing")], "bundle", str(tmp_path))
    names = zipfile.ZipFile(z).namelist()
    assert "figures/x.png" in names
