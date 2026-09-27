"""Relief RGE ALTI (vue3d/relief.py) : quantification et trous, sans réseau."""
import base64

import numpy as np
import pytest

from vue3d import relief
from vue3d.relief import RELIEF_NODATA, RELIEF_SENTINELLE


def _decoder(r):
    q = np.frombuffer(base64.b64decode(r["altitudes"]), dtype="<i2").reshape(r["height"], r["width"])
    return q


@pytest.fixture
def bil(monkeypatch):
    """Remplace le WMS par une grille donnée."""
    def poser(grille):
        monkeypatch.setattr(relief, "_bil", lambda *a: np.asarray(grille, dtype="<f4"))
    return poser


def test_le_relief_est_quantifie_au_decimetre_au_dessus_du_minimum(bil):
    bil(np.array([[100.04, 100.26], [101.0, 102.5]]))
    r = relief.fetch_relief(2, 48, 2.1, 48.1, taille=2)
    assert r["zero_m"] == 100.0
    assert _decoder(r).tolist() == [[0, 3], [10, 25]]


def test_la_scene_hors_couverture_n_a_pas_de_relief(bil):
    g = np.full((10, 10), 50.0)
    g[0, :3] = RELIEF_NODATA          # 3 % de trous : bord de couverture
    bil(g)
    assert relief.fetch_relief(2, 48, 2.1, 48.1, taille=10) is None


def test_l_anneau_garde_ses_trous_et_les_elargit(bil):
    """Mer ou frontière : l'anneau est servi, ses trous restent vides. Le
    service mélange ses −99 999 aux voisins : les altitudes aberrantes autour
    d'un trou (ici −61 m) partent avec lui, élargies de deux pixels."""
    g = np.full((12, 12), 30.0)
    g[:, :3] = RELIEF_NODATA
    g[:, 3] = -61.0
    bil(g)
    r = relief.fetch_relief_anneau(2, 48, 2.1, 48.1, taille=12)
    q = _decoder(r)
    trous = q == RELIEF_SENTINELLE
    assert trous[:, :6].all() and not trous[:, 6:].any()
    assert r["zero_m"] == 30.0


def test_l_anneau_entierement_hors_couverture_est_absent(bil):
    bil(np.full((8, 8), RELIEF_NODATA))
    assert relief.fetch_relief_anneau(2, 48, 2.1, 48.1, taille=8) is None
