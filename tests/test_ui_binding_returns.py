"""Un handler cable sur outputs=None ne doit rien renvoyer.

Gradio 5 avertit a CHAQUE declenchement quand une valeur renvoyee n'a pas de
composant pour l'accueillir ("A function returned too many output values"). Sur un
curseur, cela veut dire une paire d'avertissements par mouvement -- du bruit qui
finit par masquer un vrai avertissement.

Le test relit les branchements de cz_ui et verifie qu'aucun handler branche sur
outputs=None ne comporte de `return <valeur>`.

Run:  .venv/Scripts/python tests/test_ui_binding_returns.py
"""
import inspect
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui

BIND = re.compile(r"\.(?:change|click|input|release|select)\(\s*([A-Za-z_][\w.]*)\s*,"
                  r"\s*\[[^\]]*\]\s*,\s*None\s*\)")


def test_no_handler_returns_into_the_void():
    src = io.open(os.path.join(os.path.dirname(cz_ui.__file__), "cz_ui.py"),
                  encoding="utf-8").read()
    names = sorted(set(BIND.findall(src)))
    assert names, "aucun branchement outputs=None trouve: le motif a change ?"
    offenders = []
    for n in names:
        obj = cz_ui
        try:
            for part in n.split("."):
                obj = getattr(obj, part)
            body = inspect.getsource(obj).split(":", 1)[1]
        except Exception:
            continue
        if re.search(r"^\s+return\s+\S", body, re.M):
            offenders.append(n)
    assert not offenders, (
        f"branches sur outputs=None mais renvoient une valeur -> Gradio avertit a "
        f"chaque declenchement: {offenders}")
    print(f"OK test_no_handler_returns_into_the_void ({len(names)} branchements verifies)")


if __name__ == "__main__":
    test_no_handler_returns_into_the_void()
    print("ALL OK")
