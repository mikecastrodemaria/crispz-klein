"""The size asked for must reach the omni route, for EVERY op.

Regression (inherited from crispz-qwen-edit): `size_explicit` was only computed in
the op == "edit" branch. But a `gen` WITH refs takes the SAME omni route -- it left
without the flag, and generate_omni kept the dimensions of the REFERENCE. An
800x1312 panel rendered from a 1280x832 reference came back in landscape, then got
cropped at layout time (a cover with its title cut off).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_protocol as P


def _spec(**kw):
    s = {"protocol": 1, "op": "gen", "prompt": "a girl"}
    s.update(kw)
    return s


def test_size_explicit_on_gen_with_refs(tmpref=None):
    import tempfile
    from PIL import Image
    d = tempfile.mkdtemp(prefix="cz_size_")
    ref = os.path.join(d, "ref.png")
    Image.new("RGB", (1280, 832)).save(ref)

    out, _ = P.validate_spec(_spec(refs=[ref], width=800, height=1312))
    assert out["size_explicit"] is True, "gen + refs must pass the size along"
    assert (out["width"], out["height"]) == (800, 1312)

    # without width/height -> default 1024, and the flag must stay false so
    # the edit keeps its historical behaviour (the input's dimensions)
    out, _ = P.validate_spec(_spec(refs=[ref]))
    assert out["size_explicit"] is False, "no size asked for, nothing forced"
    print("OK test_size_explicit_on_gen_with_refs")


def test_size_explicit_without_refs():
    out, _ = P.validate_spec(_spec(width=832, height=1216))
    assert out["size_explicit"] is True
    out, _ = P.validate_spec(_spec())
    assert out["size_explicit"] is False
    print("OK test_size_explicit_without_refs")


def test_one_dimension_only_is_not_explicit():
    """width without height (or the reverse) = an incomplete size -> nothing forced."""
    out, _ = P.validate_spec(_spec(width=800))
    assert out["size_explicit"] is False, out["size_explicit"]
    print("OK test_one_dimension_only_is_not_explicit")


if __name__ == "__main__":
    test_size_explicit_on_gen_with_refs()
    test_size_explicit_without_refs()
    test_one_dimension_only_is_not_explicit()
    print("All protocol size tests passed.")
