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
    # Une à une : l'ordre des demandes est celui des tuiles.
    rgb = ortho.fetch_ortho_rgb(o, s, e, n, 0.2, fils=1)
    assert rgb.shape[0] == 4999 or rgb.shape[0] == 5000
    assert len(demandes) == 3 * 2 and max(max(d[4], d[5]) for d in demandes) <= 2048
    assert sum(d[4] * d[5] for d in demandes) == rgb.shape[0] * rgb.shape[1]
    # Première tuile au nord-ouest, dernière au sud-est, bords exacts.
    assert demandes[0][0] == pytest.approx(o) and demandes[0][3] == pytest.approx(n)
    assert demandes[-1][2] == pytest.approx(e) and demandes[-1][1] == pytest.approx(s)
    assert rgb[0, 0, 0] == 1 and rgb[-1, -1, 0] == 6 and rgb[0, 2048, 0] == 2


def _tuile_de_sa_place(demandes, delai=None):
    """Doublure du WMS : chaque tuile peinte d'une valeur tirée de son
    emprise, quel que soit l'ordre des appels ; `delai(o, n)` la retient, pour
    que les tuiles arrivent dans le désordre."""
    import threading
    import time
    verrou = threading.Lock()

    def image(o, s, e, n, largeur, hauteur):
        with verrou:
            demandes.append((o, s, e, n, largeur, hauteur))
        if delai:
            time.sleep(delai(o, n))
        valeur = (round(o * 1e5) * 7 + round(n * 1e5) * 13) % 251
        return _jpeg(np.full((hauteur, largeur, 3), valeur))
    return image


def test_les_tuiles_lues_ensemble_font_la_meme_image_qu_une_a_une(monkeypatch):
    """Les six tuiles d'une zone de 1 000 m, lues en même temps et rendues
    dans le désordre (la première la dernière) : mêmes demandes, même image."""
    from vue3d import ortho
    lat, delta = 45.7, 1000 / 2 / 111320
    o, s, e, n = 4.8 - delta, lat - delta, 4.8 + delta, lat + delta
    une_a_une, ensemble = [], []
    monkeypatch.setattr(ortho, "_image_wms", _tuile_de_sa_place(une_a_une))
    attendue = ortho.fetch_ortho_rgb(o, s, e, n, 0.2, fils=1)
    monkeypatch.setattr(ortho, "_image_wms", _tuile_de_sa_place(
        ensemble, lambda o_, n_: 0.05 if (o_, n_) == (une_a_une[0][0], une_a_une[0][3]) else 0))
    rgb = ortho.fetch_ortho_rgb(o, s, e, n, 0.2, fils=6)
    assert np.array_equal(rgb, attendue) and len(set(np.unique(rgb))) == 6
    assert sorted(ensemble) == sorted(une_a_une) and ensemble[0] == une_a_une[0]


def test_une_tuile_illisible_fait_echouer_toute_l_orthophoto(monkeypatch):
    """Une tuile perdue ne laisse pas un trou noir dans l'image : la lecture
    lève, et la couche n'est pas écrite."""
    import requests
    from vue3d import ortho
    demandes = []
    image = _tuile_de_sa_place(demandes)

    def en_panne(o, s, e, n, largeur, hauteur):
        if hauteur < 2048:                  # les tuiles de la rangée du sud
            raise requests.RequestException("HTTP 400")
        return image(o, s, e, n, largeur, hauteur)

    monkeypatch.setattr(ortho, "_image_wms", en_panne)
    lat, delta = 45.7, 1000 / 2 / 111320
    with pytest.raises(requests.RequestException):
        ortho.fetch_ortho_rgb(4.8 - delta, lat - delta, 4.8 + delta, lat + delta, 0.2, fils=6)


def test_une_tuile_en_echec_retient_celles_qui_attendent_leur_place(monkeypatch):
    """Une seule place : la tuile du nord-ouest la tient, les autres
    l'attendent, et celle du sud-est échoue. Aucune de celles qui attendaient
    ne part, et l'erreur rapportée est celle de la tuile en panne. Avant le
    groupe, deux tuiles partaient encore après l'échec de la première."""
    import threading
    import requests
    from vue3d import geopf, ortho
    monkeypatch.setattr(geopf, "_places", geopf._Places(1))
    lat, delta = 45.7, 1000 / 2 / 111320
    o, s, e, n = 4.8 - delta, lat - delta, 4.8 + delta, lat + delta
    parties, tient, libere, echec = [], threading.Event(), threading.Event(), threading.Event()

    def image(o_, s_, e_, n_, largeur, hauteur):
        if abs(e_ - e) < 1e-9 and abs(s_ - s) < 1e-9:      # sud-est : en panne
            tient.wait(5)
            threading.Timer(0.3, libere.set).start()     # le nord-ouest finira ensuite
            echec.set()
            raise requests.RequestException("HTTP 400 sur la tuile du sud-est")
        with geopf.place():
            parties.append(((o_, n_), echec.is_set()))
            if abs(o_ - o) < 1e-9 and abs(n_ - n) < 1e-9:  # nord-ouest : tient la place
                tient.set()
                libere.wait(5)
        return _jpeg(np.zeros((hauteur, largeur, 3)))

    monkeypatch.setattr(ortho, "_image_wms", image)
    with pytest.raises(requests.RequestException, match="sud-est"):
        ortho.fetch_ortho_rgb(o, s, e, n, 0.2, fils=6)
    assert [apres for _, apres in parties] == [False]
