"""Segmentation des houppiers (vue3d/houppiers.py) sur des grilles synthétiques."""
import math

import numpy as np

from vue3d.houppiers import (
    segmenter_emprise, lisser, rayon_cellules, segmenter, HOUPPIERS_MIN_CELLULES)


def _grille(nx=80, ny=80, pas=0.5, arbres=(), haies=(), bruit=0.0):
    """Grille MNH synthétique : dômes paraboliques (x, y, h, r) et haies
    (x0, y0, x1, y1, h, demi-largeur), en mètres depuis le coin sud-ouest."""
    H = np.zeros((ny, nx), dtype=np.float32)
    X = (np.arange(nx) + 0.5) * pas
    Y = (ny - np.arange(ny) - 0.5) * pas      # ligne 0 au nord
    XX, YY = np.meshgrid(X, Y)
    for x, y, h, r in arbres:
        d = np.hypot(XX - x, YY - y)
        H = np.maximum(H, np.where(d < r, h * (1 - (d / r) ** 2) * 0.8 + h * 0.2 * (d < r), 0))
    for x0, y0, x1, y1, h, dl in haies:
        # Distance au segment.
        px, py = x1 - x0, y1 - y0
        t = np.clip(((XX - x0) * px + (YY - y0) * py) / (px * px + py * py), 0, 1)
        d = np.hypot(XX - (x0 + t * px), YY - (y0 + t * py))
        H = np.maximum(H, np.where(d < dl, h * (1 - 0.3 * (d / dl) ** 2), 0))
    if bruit:
        rng = np.random.default_rng(3)
        H = np.where(H > 0, H + rng.normal(0, bruit, H.shape).astype(np.float32), H)
    return H


def _emprise(H, pas=0.5, lat0=43.5, lon0=5.6):
    ny, nx = H.shape
    m_lon = 111320 * math.cos(math.radians(lat0))
    dlon = nx * pas / m_lon / 2
    dlat = ny * pas / 111320 / 2
    return {"bbox": [lon0 - dlon, lat0 - dlat, lon0 + dlon, lat0 + dlat],
            "width": nx, "height": ny, "values": H.ravel().tolist(), "seuil_m": 2.0}


def _polygone_m(grille, x0, y0, x1, y1):
    """Polygone GeoJSON en lon/lat depuis des mètres dans la grille."""
    west, south, east, north = grille["bbox"]
    nx, ny = grille["width"], grille["height"]
    largeur_m, hauteur_m = nx * 0.5, ny * 0.5
    def ll(x, y):
        return [west + (east - west) * x / largeur_m, south + (north - south) * y / hauteur_m]
    return {"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon",
                         "coordinates": [[ll(x0, y0), ll(x1, y0), ll(x1, y1), ll(x0, y1), ll(x0, y0)]]}}


def test_deux_arbres_isoles_donnent_deux_houppiers():
    H = _grille(arbres=[(10, 10, 12, 4), (28, 30, 8, 3)])
    grille = _emprise(H)
    veg = {"features": [dict(_polygone_m(grille, 0, 0, 40, 40),
                             properties={"nature": "Forêt fermée de feuillus"})]}
    res = segmenter_emprise(grille, None, {"features": []}, veg, None)
    hp = sorted(res["houppiers"], key=lambda a: -a["h"])
    assert len(hp) == 2
    assert abs(hp[0]["h"] - 12) < 0.3 and abs(hp[1]["h"] - 8) < 0.3
    # Rayon au sol proche de celui du dôme (la partie au-dessus de 2 m).
    assert 2.5 <= hp[0]["r"] <= 4.2 and 1.8 <= hp[1]["r"] <= 3.2
    assert hp[0]["nature"] == "Forêt fermée de feuillus"
    assert res["masses"] == []


def test_le_profil_radial_descend_du_sommet_au_bord():
    H = _grille(arbres=[(20, 20, 15, 6)])
    exg = np.full(H.shape, 20, dtype=np.int8)
    res = segmenter_emprise(_emprise(H), exg, {"features": []}, {"features": []}, None)
    assert len(res["houppiers"]) == 1
    p = res["houppiers"][0]["profil"]
    assert len(p) == 6 and p[0] >= 0.9
    assert all(p[i] >= p[i + 1] - 0.05 for i in range(5))
    assert p[-1] < 0.7


def test_l_orthophoto_verte_classe_en_vegetation():
    H = _grille(arbres=[(20, 20, 10, 4)])
    grille = _emprise(H)
    exg = np.full(H.shape, 20, dtype=np.int8)        # tout est vert
    res = segmenter_emprise(grille, exg, {"features": []}, {"features": []}, None)
    assert len(res["houppiers"]) == 1 and res["masses"] == []
    assert res["houppiers"][0]["nature"] is None and res["nb_ortho"] > 0
    exg[:] = -20                                       # tout est toiture
    res = segmenter_emprise(grille, exg, {"features": []}, {"features": []}, None)
    # Masse indéterminée, bridée à 3 m de rayon : un arbre de 4 m de rayon
    # en donne plusieurs, aucune ne dépassant la bride.
    assert res["houppiers"] == [] and len(res["masses"]) >= 1
    assert all(m["r"] <= 3.2 for m in res["masses"])


def test_une_emprise_batie_est_retiree():
    H = _grille(arbres=[(20, 20, 10, 4)])
    grille = _emprise(H)
    bat = {"features": [_polygone_m(grille, 14, 14, 26, 26)]}
    res = segmenter_emprise(grille, None, bat, {"features": []}, None)
    assert res["houppiers"] == [] and res["masses"] == []
    assert res["hauteur_max"] == 0.0


def _fleche_de_grue(H, hauteur=85.0):
    """Une flèche de grue : une bande de 1,5 m de large, à `hauteur`, qui
    traverse la grille d'ouest en est par-dessus ce qui s'y trouve."""
    H = H.copy()
    H[39:42, :] = hauteur
    return H


def test_hors_foret_ce_qui_depasse_quarante_metres_n_est_pas_dessine():
    """À Notre-Dame, la flèche d'une grue au-dessus des arbres des quais
    sortait en houppier de 88 m, plus haut que les tours. L'orthophoto est
    verte dessous ; elle ne dit rien de ce qui passe au-dessus."""
    H = _fleche_de_grue(_grille(arbres=[(20, 10, 12, 4)]))
    grille = _emprise(H)
    vert = np.full(H.shape, 20, dtype=np.int8)
    for exg in (vert, -vert):                       # houppier, puis masse
        res = segmenter_emprise(grille, exg, {"features": []}, {"features": []}, None)
        assert max(e["h"] for e in res["houppiers"] + res["masses"]) < 13
        assert res["hauteur_max"] < 13
    # L'arbre sous la flèche reste, à sa hauteur.
    res = segmenter_emprise(grille, vert, {"features": []}, {"features": []}, None)
    assert [round(a["h"]) for a in res["houppiers"]] == [12]
    # Un bois de ville n'y change rien : à Notre-Dame, la grue survole le square.
    bois = {"features": [dict(_polygone_m(grille, 0, 0, 40, 40), properties={"nature": "Bois"})]}
    res = segmenter_emprise(grille, vert, {"features": []}, bois, None)
    assert max(a["h"] for a in res["houppiers"]) < 13


def test_en_foret_un_arbre_de_plus_de_quarante_metres_reste_un_arbre():
    """Sapins des Vosges à 43 m, douglas de plus de 60 m : en forêt BD TOPO,
    rien n'est plafonné."""
    H = _grille(arbres=[(20, 20, 45, 7)])
    grille = _emprise(H)
    foret = {"features": [dict(_polygone_m(grille, 0, 0, 40, 40),
                               properties={"nature": "Forêt fermée de conifères"})]}
    res = segmenter_emprise(grille, None, {"features": []}, foret, None)
    assert len(res["houppiers"]) == 1 and abs(res["houppiers"][0]["h"] - 45) < 0.5


def test_une_haie_est_allongee_et_orientee():
    H = _grille(arbres=[], haies=[(5, 20, 35, 20, 3.5, 1.2)])
    grille = _emprise(H)
    exg = np.full(H.shape, 20, dtype=np.int8)
    res = segmenter_emprise(grille, exg, {"features": []}, {"features": []}, None)
    hp = res["houppiers"]
    assert 1 <= len(hp) <= 8                 # une file, pas une bille par cellule
    assert all(a["allongement"] >= 1.5 for a in hp)
    # Axe est-ouest (0° depuis l'est), à 15° près.
    assert all(min(a["axe_deg"], 180 - a["axe_deg"]) < 15 for a in hp)


def test_l_essence_bd_foret_est_reportee():
    H = _grille(arbres=[(20, 20, 14, 5)])
    grille = _emprise(H)
    veg = {"features": [dict(_polygone_m(grille, 0, 0, 40, 40), properties={"nature": "Bois"})]}
    foret = {"features": [dict(_polygone_m(grille, 0, 0, 40, 40), properties={"essence": "Pin d'Alep"})]}
    res = segmenter_emprise(grille, None, {"features": []}, veg, foret)
    assert res["houppiers"][0]["essence"] == "Pin d'Alep"


def test_un_arbre_bruite_reste_un_seul_houppier():
    # Branches : 0,4 m de bruit sur un dôme de 12 m. Sans lissage, chaque
    # bosse ferait un sommet.
    H = _grille(arbres=[(20, 20, 12, 4.5)], bruit=0.4)
    grille = _emprise(H)
    exg = np.full(H.shape, 20, dtype=np.int8)
    res = segmenter_emprise(grille, exg, {"features": []}, {"features": []}, None)
    assert len(res["houppiers"]) == 1


def test_les_segments_minuscules_sont_fondus():
    H = _grille(arbres=[(20, 20, 12, 4.5)], bruit=0.4)
    lisse = lisser(H)
    masque = H >= 2
    rayons = rayon_cellules(lisse, 0.5, 9.0)
    labels, apex = segmenter(lisse, masque, rayons, 18)
    n = np.bincount(labels.ravel(), minlength=len(apex))[1:]
    assert not ((0 < n) & (n < HOUPPIERS_MIN_CELLULES)).any()
    # Toutes les cellules du masque sont étiquetées (ou fondues dans un voisin).
    assert (labels[masque] > 0).mean() > 0.95


class _CacheVide:
    def filter_by(self, **kw):
        return self

    def first(self):
        return None


def test_houppiers_pour_emprise_assemble_sans_telechargement():
    """Les grilles sont fournies : aucun appel réseau, un résultat sérialisable."""
    import json
    from vue3d.houppiers import houppiers_pour_emprise
    H = _grille(arbres=[(10, 10, 12, 4), (28, 30, 8, 3)])
    grille = _emprise(H)
    grille.update({"couvert": True, "source": "lidar_hd", "resolution_m": 0.5})
    foret = {"features": [dict(_polygone_m(grille, 0, 0, 40, 40), properties={"essence": "Chêne vert"})]}
    res = houppiers_pour_emprise(*grille["bbox"], {"features": []}, {"features": []}, foret,
                                 grille, np.full(H.shape, 20, dtype=np.int8))
    assert res["couvert"] and res["ortho"] and res["source"] == "lidar_hd"
    assert len(res["houppiers"]) == 2
    assert {a["essence"] for a in res["houppiers"]} == {"Chêne vert"}
    assert json.loads(json.dumps(res)) == res


def test_hors_couverture_rien_n_est_segmente():
    from vue3d.houppiers import houppiers_pour_emprise
    grille = {"couvert": False, "source": None, "values": [], "width": 8, "height": 8,
              "bbox": [0, 0, 1, 1]}
    res = houppiers_pour_emprise(0, 0, 1, 1, {"features": []}, None, None, grille, None)
    assert res["couvert"] is False and res["houppiers"] == []
    assert res["veg_disponible"] is False and res["ortho"] is False
