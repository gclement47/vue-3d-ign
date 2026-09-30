"""Réservoirs et constructions ponctuelles (vue3d/constructions.py), sans réseau."""
import math

import numpy as np
from shapely.geometry import Point, shape

from tests.test_houppiers import _emprise, _polygone_m
from vue3d import scene
from vue3d.constructions import (MASQUE_RESERVOIR_M, constructions_pour_emprise)

PAS = 0.5


def _grille(cylindres=(), nx=160, ny=160):
    """Grille MNH de 80 m de côté : des cylindres à toit plat (x, y, h, r), en
    mètres depuis le coin sud-ouest."""
    H = np.zeros((ny, nx), dtype=np.float32)
    X = (np.arange(nx) + 0.5) * PAS
    Y = (ny - np.arange(ny) - 0.5) * PAS          # ligne 0 au nord
    XX, YY = np.meshgrid(X, Y)
    for x, y, h, r in cylindres:
        H[np.hypot(XX - x, YY - y) <= r] = h
    grille = _emprise(H)
    grille.update({"couvert": True, "source": "lidar_hd", "resolution_m": PAS})
    return grille


def _lonlat(grille, x, y):
    west, south, east, north = grille["bbox"]
    return (west + (east - west) * x / (grille["width"] * PAS),
            south + (north - south) * y / (grille["height"] * PAS))


def _reservoir(grille, x, y, r, hauteur=None, nature="Réservoir industriel"):
    """Emprise circulaire à 24 côtés, comme celles de la BD TOPO."""
    pts = [_lonlat(grille, x + r * math.cos(a), y + r * math.sin(a))
           for a in np.linspace(0, 2 * math.pi, 25)]
    pts[-1] = pts[0]
    return {"type": "Feature", "properties": {"nature": nature, "hauteur": hauteur},
            "geometry": {"type": "MultiPolygon", "coordinates": [[[list(p) + [50.0] for p in pts]]]}}


def _point(grille, x, y, nature="Torchère", hauteur=None, detail=None):
    return {"type": "Feature",
            "properties": {"nature": nature, "hauteur": hauteur, "nature_detaillee": detail},
            "geometry": {"type": "Point", "coordinates": [*_lonlat(grille, x, y), 50.0]}}


def _construire(grille, reservoirs=(), points=(), batiments=()):
    return constructions_pour_emprise(
        *grille["bbox"], {"features": list(reservoirs)}, {"features": list(points)},
        {"features": list(batiments)}, grille)


def test_un_reservoir_garde_sa_hauteur_bd_topo():
    """Même quand le LiDAR le voit à une autre hauteur : la BD TOPO d'abord."""
    grille = _grille(cylindres=[(40, 40, 14.0, 10)])
    c, _ = _construire(grille, [_reservoir(grille, 40, 40, 10, hauteur=12.5)])
    (r,) = c["reservoirs"]
    assert (r["h"], r["source"], r["coupe"]) == (12.5, "bdtopo", False)
    assert r["nature"] == "Réservoir industriel"
    # Contour 2D fermé : les altitudes des sommets BD TOPO sont tombées.
    assert r["contour"][0] == r["contour"][-1] and all(len(p) == 2 for p in r["contour"])


def test_sans_hauteur_bd_topo_le_lidar_mesure_le_reservoir():
    grille = _grille(cylindres=[(40, 40, 9.0, 10)])
    c, _ = _construire(grille, [_reservoir(grille, 40, 40, 10)])
    (r,) = c["reservoirs"]
    assert (r["h"], r["source"]) == (9.0, "lidar_hd")


def test_un_reservoir_que_le_lidar_ne_voit_pas_reste_de_hauteur_inconnue():
    """À Feyzin, 14 citernes sur 30 sont à zéro dans le MNH : sans hauteur BD
    TOPO, on n'en invente pas, mais l'emprise reste."""
    grille = _grille()
    c, masque = _construire(grille, [_reservoir(grille, 40, 40, 10)])
    (r,) = c["reservoirs"]
    assert (r["h"], r["source"]) == (None, None)
    assert len(masque["features"]) == 1


def test_un_reservoir_au_bord_est_coupe_mais_masque_entier():
    grille = _grille()
    c, masque = _construire(grille, [_reservoir(grille, 78, 40, 10, hauteur=8)])
    (r,) = c["reservoirs"]
    west, south, east, north = grille["bbox"]
    assert r["coupe"] and max(p[0] for p in r["contour"]) < east
    # Le masque, lui, déborde : il est dilaté et n'est pas découpé.
    assert shape(masque["features"][0]["geometry"]).bounds[2] > east


def test_le_masque_d_un_reservoir_est_dilate():
    grille = _grille()
    res = _reservoir(grille, 40, 40, 10, hauteur=8)
    _, masque = _construire(grille, [res])
    m = shape(masque["features"][0]["geometry"])
    # À 1 m hors de la robe, dedans ; au-delà de la dilatation, dehors.
    assert m.contains(Point(*_lonlat(grille, 40 + 10 + MASQUE_RESERVOIR_M - 1, 40)))
    assert not m.contains(Point(*_lonlat(grille, 40 + 10 + MASQUE_RESERVOIR_M + 1, 40)))


def test_une_torchere_sans_hauteur_prend_celle_du_lidar_et_son_rayon():
    grille = _grille(cylindres=[(40, 40, 60.0, 3)])
    # À 2 m du sommet : la précision planimétrique de la BD TOPO.
    c, masque = _construire(grille, points=[_point(grille, 42, 40)])
    (p,) = c["ponctuelles"]
    assert (p["nature"], p["h"], p["source"]) == ("Torchère", 60.0, "lidar_hd")
    assert 2.5 <= p["r"] <= 3.5
    # La tache du MNH est retirée du sursol, en entier.
    m = shape(masque["features"][0]["geometry"])
    assert m.contains(Point(*_lonlat(grille, 37.5, 40)))


def test_une_antenne_que_le_lidar_ne_voit_pas_garde_sa_hauteur_bd_topo():
    grille = _grille()
    c, masque = _construire(grille, points=[
        _point(grille, 40, 40, "Antenne", 26.5, "Antenne-relais")])
    (p,) = c["ponctuelles"]
    assert (p["h"], p["source"], p["r"], p["detail"]) == (26.5, "bdtopo", None, "Antenne-relais")
    assert masque["features"] == []


def test_le_sommet_d_une_structure_voisine_ne_donne_ni_rayon_ni_masque():
    """BD TOPO 38,5 m, LiDAR 90 m à deux mètres : c'est la colonne d'à côté."""
    grille = _grille(cylindres=[(40, 40, 90.0, 3)])
    c, masque = _construire(grille, points=[_point(grille, 42, 40, hauteur=38.5)])
    (p,) = c["ponctuelles"]
    assert (p["h"], p["source"], p["r"]) == (38.5, "bdtopo", None)
    assert masque["features"] == []


def test_une_tache_fondue_dans_un_volume_voisin_ne_mesure_rien():
    """Une cheminée accolée à un portique de 30 m de long : hauteur lue, mais
    l'emprise au sol n'est pas la sienne."""
    grille = _grille(cylindres=[(40, 40, 30.0, 2)])
    H = np.asarray(grille["values"], dtype=np.float32).reshape(160, 160)
    H[78:82, 80:140] = 28.0                     # le portique, vers l'est
    grille["values"] = H.ravel().tolist()
    c, masque = _construire(grille, points=[_point(grille, 40, 40, "Cheminée")])
    (p,) = c["ponctuelles"]
    assert (p["h"], p["r"]) == (30.0, None) and masque["features"] == []


def test_sans_hauteur_declaree_ni_mesuree_rien_n_est_dessine():
    grille = _grille()
    c, _ = _construire(grille, points=[_point(grille, 40, 40, "Autre construction élevée")])
    assert c["ponctuelles"] == []


def test_un_point_dans_un_batiment_est_laisse_a_son_toit():
    grille = _grille(cylindres=[(40, 40, 50.0, 3)])
    batiment = _polygone_m(grille, 30, 30, 50, 50)
    c, masque = _construire(grille, points=[_point(grille, 40, 40, "Cheminée", 50.0)],
                            batiments=[batiment])
    assert c["ponctuelles"] == [] and masque["features"] == []


def test_les_natures_sans_forme_mesurable_sont_ecartees():
    grille = _grille(cylindres=[(40, 40, 20.0, 3)])
    points = [_point(grille, 40, 40, nature, 20.0)
              for nature in ("Clocher", "Croix", "Calvaire", "Transformateur", "Eolienne", "Minaret")]
    c, _ = _construire(grille, points=points)
    assert c["ponctuelles"] == []


def test_un_point_hors_de_l_emprise_est_ecarte():
    grille = _grille()
    c, _ = _construire(grille, points=[_point(grille, 95, 40, "Antenne", 30.0)])
    assert c["ponctuelles"] == []


def test_la_scene_ne_pose_plus_de_masse_sur_une_citerne():
    """Hors de toute emprise bâtie, la citerne est du sursol : des masses
    sans la couche, plus aucune avec — ni sur la robe, que le contour BD TOPO
    ne suit pas au demi-mètre."""
    grille = _grille(cylindres=[(40, 40, 12.0, 10)])
    # Emprise BD TOPO d'un mètre trop courte : le bord de la robe dépasse.
    res = _reservoir(grille, 40, 40, 9, hauteur=12.0)
    args = (*grille["bbox"], {"features": []}, {"features": []}, None, {"features": []},
            grille, None, None)
    sans = scene.assembler(*args)
    avec = scene.assembler(*args, constructions=({"features": [res]}, {"features": []}))
    assert len(sans["masses"]) > 0 and sans["constructions"] == {"reservoirs": [], "ponctuelles": []}
    assert avec["masses"] == [] and avec["houppiers"] == []
    assert len(avec["constructions"]["reservoirs"]) == 1
    # Le masque n'est pas embarqué.
    assert set(avec["constructions"]) == {"reservoirs", "ponctuelles"}
