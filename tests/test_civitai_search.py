"""CivitAI LoRA search + download, ported from crispz-studio.

The part that is NOT a copy is the ranking. The single-base forks compare the CivitAI
base label for EQUALITY ('Z-Image'); that would match nothing here, because CivitAI labels
these LoRAs 'Flux.2 Klein 4B-base' -- read from a real sidecar on this machine. So klein
matches on a normalised PREFIX, and the VARIANT matters: a 4B LoRA does not load on a 9B
base (cz_pipeline refuses it with its reason), so the versions of the LOADED variant come
first and 'Klein only' filters on the family.

No network: _api_get and the download stream are stubbed. What is checked:
  - the ranking puts the loaded variant first, the other Klein ones next, the rest last;
  - 'Klein only' keeps both Klein variants and drops the foreign bases;
  - an unreadable variant degrades to the family ranking instead of breaking the search;
  - search_loras flattens one entry per model VERSION and survives a network failure;
  - a download whose SHA256 does not match is refused AND the file is removed;
  - an existing file is never overwritten.

Run:  .venv/Scripts/python tests/test_civitai_search.py

"""
import hashlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_civitai  # noqa: E402
import cz_ui  # noqa: E402


def _model(mid, name, versions):
    return {"id": mid, "name": name, "creator": {"username": "someone"},
            "nsfw": False, "modelVersions": versions}


def _ver(vid, base, fname="a.safetensors", sha="", size=2048):
    return {"id": vid, "name": f"v{vid}", "baseModel": base,
            "files": [{"name": fname, "primary": True, "sizeKB": size,
                       "downloadUrl": f"https://civitai.com/api/download/models/{vid}",
                       "hashes": {"SHA256": sha}}],
            "images": [{"url": "https://img/x.jpg"}]}


class _Api:
    """Stubs cz_civitai._api_get with a fixed payload (or None = network failure)."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __enter__(self):
        self.real = cz_civitai._api_get

        def fake(endpoint, params=None, api_key=None):
            self.calls.append((endpoint, params))
            return self.payload

        cz_civitai._api_get = fake
        return self

    def __exit__(self, *exc):
        cz_civitai._api_get = self.real
        return False


class _Variant:
    """Forces what _klein_base_prefixes() reads as the loaded variant."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        import cz_pipeline
        self.p = cz_pipeline
        self.dim, self.vn = cz_pipeline._base_hidden_dim, cz_pipeline._variant_name
        if self.name is None:
            cz_pipeline._base_hidden_dim = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("gated repo"))
        else:
            cz_pipeline._base_hidden_dim = lambda *a, **k: 1
            cz_pipeline._variant_name = lambda _d: self.name
        return self

    def __exit__(self, *exc):
        self.p._base_hidden_dim, self.p._variant_name = self.dim, self.vn
        return False


_PAYLOAD = {"items": [
    _model(1, "Foreign", [_ver(11, "SDXL 1.0")]),
    _model(2, "Klein4B", [_ver(22, "Flux.2 Klein 4B-base")]),
    _model(3, "Klein9B", [_ver(33, "Flux.2 Klein 9B-base")]),
]}


def _bases(dd_update):
    """The base model of each candidate, read back from the dropdown labels."""
    return [lbl.split("[", 1)[1].split("]", 1)[0] for lbl in dd_update["choices"]]


def test_search_flattens_one_entry_per_version():
    with _Api({"items": [_model(1, "Two", [_ver(11, "SDXL 1.0"), _ver(12, "SDXL 1.0")])]}):
        out = cz_civitai.search_loras("x")
    assert len(out) == 2, out
    assert [c["versionId"] for c in out] == [11, 12]
    assert out[0]["modelName"] == "Two" and out[0]["url"].endswith("/models/1")


def test_a_network_failure_is_an_empty_list_not_an_exception():
    with _Api(None):
        assert cz_civitai.search_loras("x") == []
    assert cz_civitai.search_loras("   ") == []          # empty query: no call at all


def test_the_loaded_variant_comes_first_then_the_other_klein():
    """The ranking the single-base forks do not need. 9B loaded -> the 9B LoRA first, the
    4B one next (same family), the SDXL one last."""
    with _Api(_PAYLOAD), _Variant("FLUX.2-klein-9B"):
        _q, dd, state, status = cz_ui._ui_civitai_lora_search("x", False)
    assert _bases(dd) == ["Flux.2 Klein 9B-base", "Flux.2 Klein 4B-base", "SDXL 1.0"]
    assert len(state) == 3
    assert "1 match the loaded variant" in status, status


def test_klein_only_keeps_both_variants_and_drops_the_foreign_base():
    """The family, not the exact variant: a 4B LoRA is still worth showing (switch the base
    and it loads), an SDXL one never is."""
    with _Api(_PAYLOAD), _Variant("FLUX.2-klein-9B"):
        _q, dd, _s, _st = cz_ui._ui_civitai_lora_search("x", True)
    assert _bases(dd) == ["Flux.2 Klein 9B-base", "Flux.2 Klein 4B-base"]


def test_an_unreadable_variant_still_ranks_by_family():
    """_base_hidden_dim hits the Hub and a gated 9B answers 401. That must not break a
    search: it degrades to the family ranking."""
    with _Api(_PAYLOAD), _Variant(None):
        _q, dd, _s, status = cz_ui._ui_civitai_lora_search("x", False)
    assert _bases(dd)[-1] == "SDXL 1.0", _bases(dd)
    assert "match the loaded variant" not in status, status


def test_no_result_names_the_filter_as_the_likely_cause():
    with _Api({"items": [_model(1, "Foreign", [_ver(11, "SDXL 1.0")])]}), \
            _Variant("FLUX.2-klein-9B"):
        _q, dd, state, status = cz_ui._ui_civitai_lora_search("x", True)
    assert dd["choices"] == [] and state == {}
    assert "Klein only" in status, status


class _Stream:
    """Stubs urlopen with a fixed body, so the download never touches the network."""

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        import urllib.request
        self.real = urllib.request.urlopen
        body = self.body

        class _R:
            headers = {"Content-Length": str(len(body))}

            def __init__(self):
                self._b = io.BytesIO(body)

            def read(self, n):
                return self._b.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        urllib.request.urlopen = lambda *a, **k: _R()
        return self

    def __exit__(self, *exc):
        import urllib.request
        urllib.request.urlopen = self.real
        return False


def _no_enrich():
    """fetch_civitai_for_model would hit the network after a download."""
    real = cz_civitai.fetch_civitai_for_model
    cz_civitai.fetch_civitai_for_model = lambda *a, **k: None
    return real


def test_a_sha256_mismatch_refuses_and_removes_the_file():
    """The whole point of checking during the stream: a corrupted LoRA must not be left on
    disk looking valid."""
    body = b"not-the-announced-bytes"
    real = _no_enrich()
    try:
        with _Stream(body), tempfile.TemporaryDirectory() as d:
            cand = {"downloadUrl": "https://x/y", "fileName": "bad.safetensors",
                    "sha256": "0" * 64, "sizeKB": 1}
            res = cz_civitai.download_model_file(cand, d)
            assert res["success"] is False, res
            assert "SHA256 mismatch" in res["message"], res
            assert os.listdir(d) == [], os.listdir(d)      # no .part left either
    finally:
        cz_civitai.fetch_civitai_for_model = real


def test_a_matching_sha256_lands_the_file():
    body = b"the-real-bytes"
    real = _no_enrich()
    try:
        with _Stream(body), tempfile.TemporaryDirectory() as d:
            cand = {"downloadUrl": "https://x/y", "fileName": "ok.safetensors",
                    "sha256": hashlib.sha256(body).hexdigest(), "sizeKB": 1}
            res = cz_civitai.download_model_file(cand, d)
            assert res["success"] is True, res
            assert "verified" in res["message"], res
            # The file AND its hash sidecar: _cache_sha256 writes '<stem>.civitai.json' so
            # the next scan does not re-read the whole file to recompute the hash.
            assert sorted(os.listdir(d)) == ["ok.civitai.json", "ok.safetensors"], \
                os.listdir(d)
            assert res["path"] == os.path.join(d, "ok.safetensors"), res
    finally:
        cz_civitai.fetch_civitai_for_model = real


def test_an_existing_file_is_never_overwritten():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "there.safetensors")
        with open(p, "wb") as f:
            f.write(b"mine")
        res = cz_civitai.download_model_file(
            {"downloadUrl": "https://x/y", "fileName": "there.safetensors"}, d)
        assert res["success"] is True and "already exists" in res["message"], res
        with open(p, "rb") as f:
            assert f.read() == b"mine", "the existing file was overwritten"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} CivitAI search/download tests passed.")
