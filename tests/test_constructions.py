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


def test_un_point_au_niveau_de_son_batiment_est_laisse_a_son_toit():
    """118,9 m dans un bâtiment de 119,1 m, à Gardanne."""
    grille = _grille(cylindres=[(40, 40, 119.0, 3)])
    batiment = _polygone_m(grille, 30, 30, 50, 50)
    batiment["properties"]["hauteur"] = 119.1
    c, _ = _construire(grille, points=[_point(grille, 40, 40, "Autre construction élevée", 118.9)],
                       batiments=[batiment])
    assert c["ponctuelles"] == []


def test_une_cheminee_qui_depasse_son_batiment_est_dessinee():
    """Une cheminée de 52,9 m sur un bâtiment de 17,9 m, à Lyon."""
    grille = _grille(cylindres=[(40, 40, 17.9, 15), (40, 40, 52.9, 3)])
    batiment = _polygone_m(grille, 30, 30, 50, 50)
    batiment["properties"]["hauteur"] = 17.9
    c, _ = _construire(grille, points=[_point(grille, 41, 40, "Cheminée", 52.9)],
                       batiments=[batiment])
    (p,) = c["ponctuelles"]
    assert (p["h"], p["source"]) == (52.9, "bdtopo") and 2.5 <= p["r"] <= 3.5


def test_le_fut_d_une_cheminee_sans_son_sommet_donne_sa_largeur():
    """Gardanne : 295 m déclarés, le LiDAR perd le fût au-dessus de 155 m ; le
    fût reste isolé, de 10 m de rayon, et sa tache donne la largeur."""
    grille = _grille(cylindres=[(40, 40, 155.0, 10)])
    c, masque = _construire(grille, points=[_point(grille, 44, 40, "Cheminée", 295.0)])
    (p,) = c["ponctuelles"]
    assert (p["h"], p["source"]) == (295.0, "bdtopo") and 9.0 <= p["r"] <= 10.5
    assert masque["features"]


def test_un_fut_a_peine_vu_ne_donne_pas_de_largeur():
    """Sous le tiers de la hauteur déclarée, la tache est celle d'un socle."""
    grille = _grille(cylindres=[(40, 40, 60.0, 10)])
    c, _ = _construire(grille, points=[_point(grille, 40, 40, "Cheminée", 220.0)])
    (p,) = c["ponctuelles"]
    assert (p["h"], p["r"]) == (220.0, None)


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
    assert len(sans["masses"]) > 0 and sans["constructions"] == {"reservoirs": [], "ponctuelles": [], "tours": []}
    assert avec["masses"] == [] and avec["houppiers"] == []
    assert len(avec["constructions"]["reservoirs"]) == 1
    # Le masque n'est pas embarqué.
    assert set(avec["constructions"]) == {"reservoirs", "ponctuelles", "tours"}


def _anneau_bati(grille, x, y, r, r_trou, hauteur, cleabs="BATIMENT_TOUR"):
    """Bâtiment rond évidé, comme la tour de Gardanne dans la BD TOPO."""
    def cercle(rr):
        pts = [list(_lonlat(grille, x + rr * math.cos(a), y + rr * math.sin(a)))
               for a in np.linspace(0, 2 * math.pi, 49)]
        pts[-1] = pts[0]
        return pts
    return {"type": "Feature", "properties": {"cleabs": cleabs, "hauteur": hauteur},
            "geometry": {"type": "Polygon", "coordinates": [cercle(r), cercle(r_trou)[::-1]]}}


def test_un_batiment_rond_haut_que_le_lidar_ne_voit_pas_est_une_tour():
    """Gardanne : 139,3 m déclarés, coque absente du MNH, et le point de 135 m
    au centre, dans le trou de l'emprise, absorbé par la tour."""
    grille = _grille()
    c, _ = _construire(grille, points=[_point(grille, 40, 40, "Autre construction élevée", 135.0)],
                       batiments=[_anneau_bati(grille, 40, 40, 30, 20, 139.3)])
    (t,) = c["tours"]
    assert (t["cleabs"], t["h"]) == ("BATIMENT_TOUR", 139.3) and 29 <= t["r"] <= 30
    lon, lat = _lonlat(grille, 40, 40)
    assert abs(t["lon"] - lon) < 1e-6 and abs(t["lat"] - lat) < 1e-6
    assert c["ponctuelles"] == []


def test_un_batiment_rond_que_le_lidar_voit_plein_reste_un_batiment():
    grille = _grille(cylindres=[(40, 40, 70.0, 30)])
    c, _ = _construire(grille, batiments=[_anneau_bati(grille, 40, 40, 30, 0.5, 70.0)])
    assert c["tours"] == []


def test_un_batiment_rond_plus_bas_que_les_arbres_n_est_pas_une_tour():
    grille = _grille()
    c, _ = _construire(grille, batiments=[_anneau_bati(grille, 40, 40, 30, 20, 45.0)])
    assert c["tours"] == []


def test_un_batiment_haut_et_carre_n_est_pas_une_tour():
    grille = _grille()
    batiment = _polygone_m(grille, 20, 20, 60, 60)
    batiment["properties"]["hauteur"] = 120.0
    c, _ = _construire(grille, batiments=[batiment])
    assert c["tours"] == []
