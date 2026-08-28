import hashlib

from samson_mlip_visualizer.provenance import collect_provenance, model_digest


def test_model_digest_matches_hashlib(tmp_path):
    model = tmp_path / "model.pth"
    payload = b"deadbeef" * 100
    model.write_bytes(payload)
    sha, size = model_digest(model)
    assert sha == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)


def test_collect_provenance_fields(tmp_path):
    model = tmp_path / "model.model"
    model.write_bytes(b"abc")
    prov = collect_provenance(
        backend="mace", model_path=model, device="cuda", dtype="float32"
    )
    assert prov.backend == "mace"
    assert prov.device == "cuda"
    assert prov.dtype == "float32"
    assert prov.model_size_bytes == 3
    assert prov.model_sha256 == hashlib.sha256(b"abc").hexdigest()
    assert prov.created_utc.endswith("+00:00")
    # ase and numpy are hard dependencies, so their versions are always recorded.
    assert "ase" in prov.versions
    assert "numpy" in prov.versions


def test_provenance_roundtrips_through_dict_and_text(tmp_path):
    model = tmp_path / "m.pb"
    model.write_bytes(b"x")
    prov = collect_provenance(
        backend="deepmd", model_path=model, device="cpu", dtype="float64"
    )
    flat = prov.as_dict()
    assert flat["mlip_backend"] == "deepmd"
    assert flat["mlip_model_sha256"] == prov.model_sha256
    assert "deepmd" in prov.as_text()
