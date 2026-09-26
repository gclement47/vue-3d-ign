"""Indice de verdure ExG de l'orthophoto (vue3d/ortho.py)."""
import io

import numpy as np
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
