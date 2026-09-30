"""Découpe des bâtiments sur l'emprise (vue3d/batiments.py), sans réseau."""
import math

import pytest
from shapely.geometry import shape

from vue3d.batiments import DECOUPE_RETRAIT_M, decouper_batiments

LAT0, LON0 = 45.0, 5.0
M_LON = 111320 * math.cos(math.radians(LAT0))
# Une scène de 200 m de côté, en mètres autour de (LON0, LAT0).
DEMI = 100.0
EMPRISE = (LON0 - DEMI / M_LON, LAT0 - DEMI / 111320, LON0 + DEMI / M_LON, LAT0 + DEMI / 111320)


def _deg(x, y, *z):
    return [LON0 + x / M_LON, LAT0 + y / 111320, *z]


def _rect(x0, y0, x1, y1):
    return [_deg(x0, y0), _deg(x1, y0), _deg(x1, y1), _deg(x0, y1), _deg(x0, y0)]


def _bat(cleabs, *anneaux, **props):
    return {"type": "Feature", "properties": {"cleabs": cleabs, **props},
            "geometry": {"type": "MultiPolygon", "coordinates": [list(anneaux)]}}


def _bornes_m(feature):
    minx, miny, maxx, maxy = shape(feature["geometry"]).bounds
    return ((minx - LON0) * M_LON, (miny - LAT0) * 111320,
            (maxx - LON0) * M_LON, (maxy - LAT0) * 111320)


def test_un_batiment_dans_la_scene_est_rendu_tel_quel():
    maison = _bat("MAISON", _rect(-10, -5, 10, 5), hauteur=6)
    (rendu,) = decouper_batiments({"features": [maison]}, *EMPRISE)["features"]
    assert rendu is maison and "coupe" not in rendu["properties"]


def test_un_batiment_qui_deborde_est_coupe_en_retrait_du_bord():
    """Un château de 300 m sur une scène de 200 m : le morceau gardé s'arrête
    avant le bord, là où la grille MNH l'encadre encore, et garde ses
    attributs."""
    chateau = _bat("CHATEAU", _rect(-150, -20, 150, 20), hauteur=27.7)
    (rendu,) = decouper_batiments({"features": [chateau]}, *EMPRISE)["features"]
    bord = DEMI - DECOUPE_RETRAIT_M
    assert _bornes_m(rendu) == pytest.approx((-bord, -20, bord, 20), abs=1e-6)
    assert rendu["properties"]["hauteur"] == 27.7 and rendu["properties"]["cleabs"] == "CHATEAU"
    # Deux tiers de l'emprise gardés ; la largeur est celle du château entier.
    assert rendu["properties"]["coupe"] == {"part": pytest.approx(2 * bord / 300, abs=0.01),
                                            "largeur_m": 40.0}
    # L'original n'est pas modifié : les houppiers le lisent entier.
    assert "coupe" not in chateau["properties"]


def test_la_cour_interieure_survit_a_la_decoupe():
    ilot = _bat("ILOT", _rect(-150, -30, 150, 30), _rect(-20, -10, 20, 10))
    (rendu,) = decouper_batiments({"features": [ilot]}, *EMPRISE)["features"]
    (morceau,) = shape(rendu["geometry"]).geoms
    assert len(morceau.interiors) == 1


def test_un_batiment_hors_du_cadre_est_retire():
    """Le WFS rend aussi ce qui ne fait que toucher l'emprise."""
    dehors = _bat("DEHORS", _rect(DEMI - 0.5, -5, DEMI + 20, 5))
    assert decouper_batiments({"features": [dehors]}, *EMPRISE)["features"] == []


def test_un_batiment_en_u_peut_rester_en_deux_morceaux():
    """Un U dont seules les deux ailes entrent dans la scène."""
    u = [_deg(80, -30), _deg(130, -30), _deg(130, 30), _deg(80, 30), _deg(80, 20),
         _deg(120, 20), _deg(120, -20), _deg(80, -20), _deg(80, -30)]
    (rendu,) = decouper_batiments({"features": [_bat("U", u)]}, *EMPRISE)["features"]
    assert len(shape(rendu["geometry"]).geoms) == 2


def test_les_altitudes_des_sommets_bd_topo_sont_ecartees():
    """La BD TOPO livre des sommets en 3D : le morceau est en 2D, sans Z
    interpolé ni NaN qui casserait le JSON de la scène."""
    import json
    en_3d = [_deg(x, y, 135.0) for x, y in ((-150, -20), (150, -20), (150, 20), (-150, 20), (-150, -20))]
    (rendu,) = decouper_batiments({"features": [_bat("B", en_3d)]}, *EMPRISE)["features"]
    assert all(len(p) == 2 for p in rendu["geometry"]["coordinates"][0][0])
    json.dumps(rendu, allow_nan=False)
