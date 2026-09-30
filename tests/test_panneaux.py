"""Panneaux solaires (vue3d/panneaux.py) : projection, base, lecture, découpe."""
import gzip
import json
import math
import os

import pytest

from vue3d import panneaux
from vue3d.panneaux import (Lecteur, PanneauxMalConfigures, laea_vers_lonlat, lecteur,
                            panneaux_pour_emprise, preparer)

# Quatre points de France, EPSG:3035 -> WGS84 calculés par pyproj (PROJ 9).
REFERENCES = [(3356989.17, 3041589.22, -3.4442271, 49.689594),
              (4013593.16, 2309574.56, 6.1857517, 43.8251834),
              (3705227.96, 2548039.78, 2.0801568, 45.7499615),
              (3917522.8, 2903526.44, 4.4693725, 49.1083612)]


@pytest.mark.parametrize("x,y,lon,lat", REFERENCES)
def test_la_projection_inverse_retrouve_pyproj_au_centimetre(x, y, lon, lat):
    lo, la = laea_vers_lonlat(x, y)
    ecart_m = math.hypot((lo - lon) * 111320 * math.cos(math.radians(lat)), (la - lat) * 111320)
    assert ecart_m < 0.01


def test_l_origine_de_la_projection_est_son_point_de_reference():
    assert laea_vers_lonlat(4321000.0, 3210000.0) == pytest.approx((10.0, 52.0))


# Emprise carrée de 200 m à 45° N, 2° E, et un carré de 10 m en EPSG:3035
# posé dessus par la projection inverse... à l'envers : on part de WGS84 et on
# cherche les x, y qui y reviennent, par quelques itérations de Newton.
LAT, LON = 45.0, 2.0
KX, KY = 111320 * math.cos(math.radians(LAT)), 111320


def _vers_laea(lon, lat):
    x, y = 3705000.0, 2465000.0
    for _ in range(30):
        lo, la = laea_vers_lonlat(x, y)
        x += (lon - lo) * KX * 0.9
        y += (lat - la) * KY * 0.9
    return x, y


def _carre_3035(cx_m, cy_m, cote_m):
    """Polygone EPSG:3035 d'un carré centré à (cx_m, cy_m) mètres du point."""
    coins = [(cx_m - cote_m / 2, cy_m - cote_m / 2), (cx_m + cote_m / 2, cy_m - cote_m / 2),
             (cx_m + cote_m / 2, cy_m + cote_m / 2), (cx_m - cote_m / 2, cy_m + cote_m / 2)]
    pts = [_vers_laea(LON + x / KX, LAT + y / KY) for x, y in coins]
    return {"type": "Feature", "properties": {"surface": cote_m * cote_m, "kWp_approx": 3, "year": 2023,
                                              "yield_kwh": 1},
            "geometry": {"type": "Polygon", "coordinates": [pts + [pts[0]]]}}


@pytest.fixture
def base(tmp_path):
    """Deux installations : une au centre, une à 500 m, hors emprise ; et une
    géométrie sans polygone, ignorée. Un fichier compressé, un non."""
    a = tmp_path / "a.geojson.gz"
    with gzip.open(a, "wt") as f:
        json.dump({"type": "FeatureCollection", "features": [_carre_3035(0, 0, 10)]}, f)
    b = tmp_path / "b.geojson"
    b.write_text(json.dumps({"type": "FeatureCollection", "features": [
        _carre_3035(500, 0, 6), {"type": "Feature", "properties": {}, "geometry": None}]}))
    sortie = str(tmp_path / "panneaux.sqlite")
    assert preparer([str(a), str(b)], sortie) == 2
    return sortie


def test_la_base_se_lit_par_emprise(base):
    lire = Lecteur(base)
    d = 100 / KX, 100 / KY
    dedans = lire(LON - d[0], LAT - d[1], LON + d[0], LAT + d[1])
    assert len(dedans) == 1 and dedans[0]["surface"] == 100 and dedans[0]["kwp"] == 3
    (contour,) = [p["contour"] for p in dedans]
    assert len(contour) == 5 and all(abs(x - LON) * KX < 5.01 and abs(y - LAT) * KY < 5.01 for x, y in contour)
    # Une emprise qui touche le bord d'une installation la lit aussi.
    assert len(lire(LON + 4 / KX, LAT - d[1], LON + d[0], LAT + d[1])) == 1
    assert lire(LON + 6 / KX, LAT - d[1], LON + d[0], LAT + d[1]) == []


def test_la_decoupe_sur_l_emprise_garde_les_morceaux_en_sens_trigonometrique(base):
    d = 100 / KX, 100 / KY
    # Emprise dont le bord ouest coupe le carré central en deux.
    couche = panneaux_pour_emprise(LON, LAT - d[1], LON + d[0], LAT + d[1],
                                   Lecteur(base)(LON, LAT - d[1], LON + d[0], LAT + d[1]))
    assert couche["version"] == panneaux.PANNEAUX_VERSION
    (p,) = couche["panneaux"]
    assert p["surface"] == 100 and p["annee"] == 2023 and p["kwp"] == 3
    xs = [x for x, _ in p["contour"]]
    assert min(xs) == pytest.approx(LON, abs=1e-7) and max(xs) == pytest.approx(LON + 5 / KX, abs=2e-7)
    aire = sum(p["contour"][i][0] * p["contour"][(i + 1) % 4][1] - p["contour"][(i + 1) % 4][0] * p["contour"][i][1]
               for i in range(len(p["contour"]))) / 2
    assert aire > 0 and len(p["contour"]) == 4
    assert panneaux_pour_emprise(LON, LAT, LON + d[0], LAT + d[1], None) == {
        "version": panneaux.PANNEAUX_VERSION, "panneaux": []}


def test_sans_variable_la_couche_n_existe_pas(monkeypatch):
    monkeypatch.delenv("VUE3D_PANNEAUX", raising=False)
    assert lecteur() is None
    assert lecteur("  ") is None


def test_une_base_introuvable_arrete_le_demarrage(tmp_path):
    with pytest.raises(PanneauxMalConfigures, match="preparer_panneaux"):
        lecteur(str(tmp_path / "absente.sqlite"))


def test_la_base_est_relue_a_chaque_lecture_sans_la_garder_ouverte(base):
    lire = lecteur(base)
    assert lire.chemin == base and os.path.exists(base)
    d = 100 / KX, 100 / KY
    assert len(lire(LON - d[0], LAT - d[1], LON + d[0], LAT + d[1])) == 1
