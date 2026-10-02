"""Grille MNH (vue3d/mnh.py) : décodage, taille, repli MNS − MNT."""
import struct
import threading

import numpy as np
import pytest
import requests

from vue3d import mnh


def _ancien_decodage(contenu, largeur, hauteur):
    """Le décodage d'avant numpy, cellule à cellule : la référence."""
    brut = struct.unpack(f"<{largeur * hauteur}f", contenu[:largeur * hauteur * 4])
    return [0.0 if (v < 0 or v > 200 or v != v) else round(v, 1) for v in brut]


def test_le_decodage_rend_les_memes_hauteurs_qu_avant():
    """À l'octet près, signe de zéro compris : les ex aequo (0,25, 0,35 en
    float32), les bornes, le nodata, NaN et l'infini, puis du bruit."""
    speciaux = [0.0, -0.0, 0.25, 0.35, 0.05, 0.15, 2.675, 199.95, 200.0, 200.04, 200.1,
                -0.04, -99999.0, -9999.0, float("nan"), float("inf"), -float("inf"),
                1e-30, 12.349999, 12.35, 12.45, 87.65]
    alea = np.random.default_rng(3).uniform(-5, 210, 20000).astype(np.float32).tolist()
    valeurs = speciaux + alea
    contenu = struct.pack(f"<{len(valeurs)}f", *valeurs)
    attendu = _ancien_decodage(contenu, len(valeurs), 1)
    obtenu = mnh._grille_depuis_bil(contenu, len(valeurs), 1)
    assert type(obtenu) is list and all(type(v) is float for v in obtenu)
    assert [repr(v) for v in obtenu] == [repr(v) for v in attendu]


def test_un_bil_tronque_est_refuse():
    with pytest.raises(ValueError):
        mnh._grille_depuis_bil(b"\0" * 12, 2, 2)


def test_la_taille_de_la_grille_ne_depend_que_de_l_emprise(monkeypatch):
    """La scène demande l'orthophoto et le terrain à cette taille avant
    d'avoir la grille : fetch_mnh_grid doit prendre la même."""
    demandes = []

    def grille_wms(layer, w, s, e, n, largeur, hauteur, brut=False):
        demandes.append((largeur, hauteur))
        return [1.0] * (largeur * hauteur)

    monkeypatch.setattr(mnh, "_grille_wms", grille_wms)
    for bbox in ((5.19, 43.90, 5.21, 43.92), (7.745, 48.5773, 7.7554, 48.5863),
                 (2.1188, 48.8033, 2.1220, 48.8065)):
        grille = mnh.fetch_mnh_grid(*bbox, resolution_m=0.5, max_pixels=2048)
        attendu = mnh.dimensions_grille(*bbox, 0.5, 2048)
        assert (grille["width"], grille["height"]) == attendu == demandes[-1]


def test_le_repli_lit_mns_et_mnt_ensemble(monkeypatch):
    """Hors LiDAR HD, les deux grilles du repli sont lues en même temps."""
    barriere = threading.Barrier(2, timeout=5)

    def grille_wms(layer, w, s, e, n, largeur, hauteur, brut=False):
        if layer == mnh.MNH_LAYER:
            return [0.0] * (largeur * hauteur)
        barriere.wait()
        return [110.0 if layer == mnh.MNS_LAYER else 100.0] * (largeur * hauteur)

    monkeypatch.setattr(mnh, "_grille_wms", grille_wms)
    grille = mnh.fetch_mnh_grid(5.19, 43.90, 5.1902, 43.9002)
    assert grille["source"] == "mns_mnt" and grille["couvert"]
    assert set(grille["values"]) == {10.0}


def test_un_repli_en_echec_garde_la_reponse_lidar(monkeypatch):
    def grille_wms(layer, w, s, e, n, largeur, hauteur, brut=False):
        if layer == mnh.MNT_LAYER:
            raise requests.RequestException("HTTP 502")
        return [0.0 if layer == mnh.MNH_LAYER else 110.0] * (largeur * hauteur)

    monkeypatch.setattr(mnh, "_grille_wms", grille_wms)
    grille = mnh.fetch_mnh_grid(5.19, 43.90, 5.1902, 43.9002)
    assert not grille["couvert"] and grille["source"] is None
