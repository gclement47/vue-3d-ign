"""Indice de verdure ExG de l'orthophoto (vue3d/ortho.py)."""
import io

import numpy as np
import pytest
from PIL import Image

from vue3d.ortho import exg_depuis_image, EXG_SEUIL


def _jpeg(pixels):
    img = Image.fromarray(np.asarray(pixels, dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")     # sans perte : les valeurs sont exactes
    return buf.getvalue()


def test_exg_separe_feuillage_et_toiture():
    # Ligne du haut : vert franc ; ligne du bas : tuile rougeâtre et ardoise.
    exg = exg_depuis_image(_jpeg([[[60, 100, 40], [80, 110, 60]],
                                  [[150, 90, 70], [90, 90, 100]]]))
    assert exg.shape == (2, 2) and exg.dtype == np.int8
    assert exg[0, 0] == 2 * 100 - 60 - 40 and exg[0, 1] == 2 * 110 - 80 - 60
    assert exg[1, 0] == 2 * 90 - 150 - 70 and exg[1, 1] == 2 * 90 - 90 - 100
    assert (exg[0] >= EXG_SEUIL).all() and (exg[1] < EXG_SEUIL).all()


def test_exg_est_borne_a_l_int8():
    exg = exg_depuis_image(_jpeg([[[0, 255, 0], [255, 0, 255]]]))
    assert exg[0, 0] == 127 and exg[0, 1] == -127


def test_la_mosaique_plafonnee_garde_ses_proportions(monkeypatch):
    """Une zone de 1 000 m dépasse 2 048 px à 0,4 m : les deux côtés sont
    réduits ensemble, sinon l'image sortirait étirée."""
    import math
    from vue3d import ortho
    monkeypatch.setattr(ortho, "_image_wms", lambda *bbox: b"jpeg")
    lat, delta = 45.7, 1000 / 2 / 111320
    _, largeur, hauteur = ortho.fetch_ortho_jpeg(4.8 - delta, lat - delta, 4.8 + delta, lat + delta)
    assert hauteur == 2048
    assert largeur / hauteur == pytest.approx(math.cos(math.radians(lat)), rel=0.01)


def test_l_orthophoto_des_detections_garde_sa_resolution_en_tuiles(monkeypatch):
    """0,2 m sur 1 000 m : 5 000 px de haut, lus en tuiles de 2 048 au plus,
    qui pavent l'emprise sans trou ni recouvrement."""
    from vue3d import ortho
    demandes = []

    def image(o, s, e, n, largeur, hauteur):
        demandes.append((o, s, e, n, largeur, hauteur))
        # Chaque tuile de son numéro : l'assemblage doit les mettre à leur place.
        return _jpeg(np.full((hauteur, largeur, 3), len(demandes)))

    monkeypatch.setattr(ortho, "_image_wms", image)
    lat, delta = 45.7, 1000 / 2 / 111320
    o, s, e, n = 4.8 - delta, lat - delta, 4.8 + delta, lat + delta
    rgb = ortho.fetch_ortho_rgb(o, s, e, n, 0.2)
    assert rgb.shape[0] == 4999 or rgb.shape[0] == 5000
    assert len(demandes) == 3 * 2 and max(max(d[4], d[5]) for d in demandes) <= 2048
    assert sum(d[4] * d[5] for d in demandes) == rgb.shape[0] * rgb.shape[1]
    # Première tuile au nord-ouest, dernière au sud-est, bords exacts.
    assert demandes[0][0] == pytest.approx(o) and demandes[0][3] == pytest.approx(n)
    assert demandes[-1][2] == pytest.approx(e) and demandes[-1][1] == pytest.approx(s)
    assert rgb[0, 0, 0] == 1 and rgb[-1, -1, 0] == 6 and rgb[0, 2048, 0] == 2
