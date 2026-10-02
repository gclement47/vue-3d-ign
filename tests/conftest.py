"""Réglages communs à toute la suite.

Aucun test ne lance de processus de calcul des toitures sans le demander.
Importé, vue3d/app.py autorise le bassin (vue3d/toits.py) et le fait naître
aussitôt : dix processus pendant toute la session, qui réexécuteraient le
script principal de pytest. VUE3D_TOITS_PROCESSUS=1, posé ici avant tout
import de vue3d, l'en empêche ; les tests du bassin passent `processus=`
et règlent eux-mêmes la variable.
"""

import os

import pytest

os.environ["VUE3D_TOITS_PROCESSUS"] = "1"


@pytest.fixture(autouse=True)
def _sans_bassin_implicite(monkeypatch):
    """Un appel sans `processus` calcule ses toitures ici, même si un test
    règle VUE3D_TOITS_PROCESSUS : seul `processus=` mène au bassin."""
    from vue3d import toits
    monkeypatch.setattr(toits, "_bassin_autorise", False)
