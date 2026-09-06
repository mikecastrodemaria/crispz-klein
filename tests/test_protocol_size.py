"""La taille demandee doit atteindre la route omni, pour TOUS les ops.

Regression (heritee de crispz-qwen-edit): `size_explicit` n'etait calcule que dans
la branche op == "edit". Or un `gen` AVEC refs passe par la MEME route omni --
il repartait donc sans le flag, et generate_omni conservait les dimensions de la
REFERENCE. Une case 800x1312 rendue avec une reference 1280x832 ressortait en
paysage, puis se faisait recadrer a la composition (titre coupe sur une couverture).
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
    assert out["size_explicit"] is True, "gen + refs doit transmettre la taille"
    assert (out["width"], out["height"]) == (800, 1312)

    # sans width/height -> defaut 1024, et le flag doit rester faux pour que
    # l'edition garde le comportement historique (dimensions de l'entree)
    out, _ = P.validate_spec(_spec(refs=[ref]))
    assert out["size_explicit"] is False, "sans taille demandee, pas de forcage"
    print("OK test_size_explicit_on_gen_with_refs")


def test_size_explicit_without_refs():
    out, _ = P.validate_spec(_spec(width=832, height=1216))
    assert out["size_explicit"] is True
    out, _ = P.validate_spec(_spec())
    assert out["size_explicit"] is False
    print("OK test_size_explicit_without_refs")


def test_one_dimension_only_is_not_explicit():
    """width sans height (ou l'inverse) = taille incomplete -> on ne force pas."""
    out, _ = P.validate_spec(_spec(width=800))
    assert out["size_explicit"] is False, out["size_explicit"]
    print("OK test_one_dimension_only_is_not_explicit")


if __name__ == "__main__":
    test_size_explicit_on_gen_with_refs()
    test_size_explicit_without_refs()
    test_one_dimension_only_is_not_explicit()
    print("All protocol size tests passed.")
