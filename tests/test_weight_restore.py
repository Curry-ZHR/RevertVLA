"""Release installer rejects unsafe entries and preserves different local weights."""
import hashlib
import io
import tarfile
from pathlib import Path
import importlib.util
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/download_weights.py"
spec = importlib.util.spec_from_file_location("weight_download", path)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def archive(tmp_path, name, data=b"model"):
    target = tmp_path / "asset.tar.gz"
    with tarfile.open(target, "w:gz") as bundle:
        entry = tarfile.TarInfo(name)
        entry.size = len(data)
        bundle.addfile(entry, io.BytesIO(data))
    return target


@pytest.mark.parametrize("name", ["../escape", "/absolute", "unexpected"])
def test_archive_rejects_undeclared_or_escaping_path(tmp_path, name):
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ValueError):
        download.restore(archive(tmp_path, name), root, {"weights/model": hashlib.sha256(b"model").hexdigest()})
    assert not (tmp_path / "escape").exists()


def test_restore_verifies_file_and_preserves_different_local_copy(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    name = "weights/model"
    expected = {name: hashlib.sha256(b"model").hexdigest()}
    source = archive(tmp_path, name)
    download.restore(source, root, expected)
    assert (root / name).read_bytes() == b"model"
    (root / name).write_bytes(b"local-change")
    with pytest.raises(FileExistsError):
        download.restore(source, root, expected)
    assert (root / name).read_bytes() == b"local-change"


def test_bad_file_checksum_is_not_installed(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ValueError):
        download.restore(archive(tmp_path, "weights/model"), root, {"weights/model": "0" * 64})
    assert not (root / "weights/model").exists()
