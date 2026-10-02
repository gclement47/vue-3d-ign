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


def test_un_repli_en_echec_n_abandonne_pas_la_scene(monkeypatch):
    """Dans une scène, le MNS en échec abandonne le MNT, son voisin du
    repli, pas les autres lectures de la scène : la grille LiDAR, même vide,
    est rendue comme avant, et c'est la scène qui la juge."""
    import concurrent.futures

    from vue3d import geopf

    class Reponse:
        ok = True
        status_code = 200
        headers = {"Content-Type": "image/x-bil;bits=32"}

        def __init__(self, contenu):
            self.content = contenu

        def raise_for_status(self):
            pass

    def get(url, timeout=None):
        if f"LAYERS={mnh.MNS_LAYER}&" in url:
            raise requests.ConnectionError("panne")
        valeur = 110.0 if f"LAYERS={mnh.MNT_LAYER}&" in url else 0.0
        return Reponse(struct.pack("<4f", *[valeur] * 4))

    monkeypatch.setattr(geopf, "_places", geopf._Places(geopf.GEOPF_SIMULTANEES, geopf.GEOPF_FOND))
    monkeypatch.setattr(geopf.requests, "get", get)
    monkeypatch.setattr(geopf.time, "sleep", lambda s: None)
    monkeypatch.setattr(mnh, "dimensions_grille", lambda *a, **k: (2, 2))
    groupe = geopf.Groupe(de_scene=True)
    with concurrent.futures.ThreadPoolExecutor(1) as bassin:
        grille = groupe.soumettre(bassin, mnh.fetch_mnh_grid, 5.19, 43.90, 5.1902,
                                  43.9002).result(timeout=5)
    assert not grille["couvert"] and grille["source"] is None
    assert not groupe.abandonne()
