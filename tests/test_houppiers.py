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


def test_la_fenetre_d_une_emprise_garde_toutes_ses_cellules():
    """Tester un polygone dans sa fenêtre rend le masque de la grille entière."""
    import shapely
    from shapely.geometry import Polygon, box
    from vue3d.houppiers import _fenetre
    lons = 5.6 + (np.arange(50) + 0.5) * 1e-5
    lats = 43.5 - (np.arange(40) + 0.5) * 1e-5
    LON, LAT = np.meshgrid(lons, lats)
    for geom in (Polygon([(5.6001, 43.4999), (5.6003, 43.49985), (5.60022, 43.4997)]),
                 box(5.5999, 43.4995, 5.60012, 43.49972),      # à cheval sur le bord
                 box(5.7, 43.6, 5.8, 43.7),                    # hors de la grille
                 Polygon()):
        fenetre = _fenetre(geom, lons, lats)
        masque = np.zeros(LON.shape, dtype=bool)
        masque[fenetre] = shapely.contains_xy(geom, LON[fenetre], LAT[fenetre])
        assert np.array_equal(masque, shapely.contains_xy(geom, LON, LAT))


# --- Équivalence avec la segmentation de la scène v14 ------------------------
#
# houppiers.py a été réécrit pour aller plus vite, à scène identique à
# l'octet. tests/houppiers_v14.py garde le code d'avant : les deux doivent
# rendre exactement la même chose sur des grilles tirées au hasard, bords de
# grille, plateaux, bâti, zones, forêt et flèche de grue compris.


def _scene_au_hasard(graine, nx=96, ny=74):
    """(grille, exg, bâtiments, végétation, forêts) tirés au hasard."""
    rng = np.random.default_rng(graine)
    largeur, hauteur = nx * 0.5, ny * 0.5
    arbres = [(rng.uniform(-3, largeur + 3), rng.uniform(-3, hauteur + 3),
               rng.uniform(2.5, 30), rng.uniform(1.2, 7)) for _ in range(rng.integers(4, 30))]
    haies = [(rng.uniform(0, largeur), rng.uniform(0, hauteur), rng.uniform(0, largeur),
              rng.uniform(0, hauteur), rng.uniform(2.5, 6), rng.uniform(0.5, 2))
             for _ in range(rng.integers(0, 4))]
    H = _grille(nx, ny, arbres=arbres, haies=haies, bruit=float(rng.choice([0, 0.3])))
    # Au décimètre, comme le MNH : des plateaux, que seul le bruit du lissage départage.
    H = np.round(H, 1).astype(np.float32)
    if rng.random() < 0.5:
        H = _fleche_de_grue(H, hauteur=float(rng.uniform(41, 90)))
    grille = _emprise(H)

    def rectangle(proprietes=None):
        x0, x1 = sorted(rng.uniform(-5, largeur + 5, 2))
        y0, y1 = sorted(rng.uniform(-5, hauteur + 5, 2))
        return dict(_polygone_m(grille, x0, y0, x1, y1), properties=proprietes or {})

    bat = {"features": [rectangle() for _ in range(rng.integers(0, 6))]}
    natures = ["Forêt fermée de feuillus", "Bois", "Haie", "Forêt ouverte", None]
    veg = {"features": [rectangle({"nature": natures[rng.integers(len(natures))]})
                        for _ in range(rng.integers(0, 5))]}
    essences = ["Chêne vert", "Pin d'Alep", "Hêtre"]
    forets = {"features": [rectangle({"essence": essences[rng.integers(len(essences))]})
                           for _ in range(rng.integers(0, 4))]}
    exg = None
    if rng.random() < 0.8:
        exg = rng.integers(-30, 30, H.shape).astype(np.int8)
    return grille, exg, bat, veg, forets


def test_segmenter_emprise_rend_la_scene_v14():
    import json
    from tests import houppiers_v14
    for graine in range(40):
        entree = _scene_au_hasard(graine)
        attendu = houppiers_v14.segmenter_emprise(*entree)
        obtenu = segmenter_emprise(*entree)
        # Comparés en JSON : un -0.0 pour un 0.0 serait un octet de différence.
        assert json.dumps(obtenu) == json.dumps(attendu), graine


def test_segmenter_rend_les_etiquettes_v14():
    """Sur des masques et des rayons quelconques, pas seulement ceux que
    segmenter_emprise produit : rayons jusqu'à 15 cellules, masque troué ;
    et sur des hauteurs sans le bruit du lissage, pleines d'égalités que
    l'ordre des voisins départage, ou négatives, où le bord de la grille ne
    compte pas comme une cellule à -1."""
    from tests import houppiers_v14
    for graine in range(30):
        rng = np.random.default_rng(100 + graine)
        grille = _scene_au_hasard(graine, nx=int(rng.integers(30, 90)),
                                  ny=int(rng.integers(30, 90)))[0]
        H = np.asarray(grille["values"], dtype=np.float32).reshape(grille["height"],
                                                                   grille["width"])
        masque = (H >= 2) & (rng.random(H.shape) > rng.uniform(0, 0.2))
        rayons = rng.integers(1, int(rng.integers(2, 16)), H.shape).astype(np.int32)
        etendue = float(rng.uniform(2, 30))
        for lisse in (lisser(H), np.round(lisser(H)), lisser(H) - 8):
            stats_v14, stats = {}, {}
            attendu = houppiers_v14.segmenter(lisse, masque.copy(), rayons, etendue, stats_v14)
            obtenu = segmenter(lisse, masque.copy(), rayons, etendue, stats)
            assert np.array_equal(obtenu[0], attendu[0]) and obtenu[1] == attendu[1], graine
            assert stats == stats_v14


def test_les_sommets_sont_ceux_des_dilatations_successives():
    from tests import houppiers_v14
    from vue3d.houppiers import sommets
    rng = np.random.default_rng(7)
    for _ in range(20):
        ny, nx = rng.integers(1, 60, 2)
        # Valeurs au décimètre : des égalités, que >= doit garder.
        lisse = np.round(rng.uniform(0, 20, (ny, nx)), 1).astype(np.float32)
        masque = rng.random((ny, nx)) > 0.3
        rayons = rng.integers(0, 14, (ny, nx)).astype(np.int32)
        assert np.array_equal(sommets(lisse, masque, rayons),
                              houppiers_v14.sommets(lisse, masque, rayons))


def test_la_dilatation_separee_est_la_dilatation_carree():
    from tests import houppiers_v14
    from vue3d.houppiers import _dilater
    rng = np.random.default_rng(5)
    for forme in ((1, 1), (1, 7), (6, 1), (2, 2), (33, 21)):
        g = rng.uniform(-5, 5, forme).astype(np.float32)
        g[rng.random(forme) < 0.05] = np.nan
        assert np.array_equal(_dilater(g), houppiers_v14._dilater(g), equal_nan=True)
        b = rng.random(forme) > 0.8
        assert np.array_equal(_dilater(b), houppiers_v14._dilater(b))


def test_arrondir_est_round_de_python():
    import math
    import struct
    from vue3d.houppiers import _arrondir
    rng = np.random.default_rng(11)
    valeurs = list(rng.uniform(-200, 200, 20000)) + list(rng.uniform(0, 1, 20000))
    # Au plus près des cas où l'arrondi bascule : les décimaux à demi-unité
    # (2.675, 0.125…), leurs voisins immédiats, des coordonnées.
    for chiffres in (1, 2, 7):
        for q in range(-3000, 3000):
            d = (q + 0.5) / 10 ** chiffres
            valeurs += [d, math.nextafter(d, math.inf), math.nextafter(d, -math.inf)]
    valeurs += [5.20030005, 43.91160005, 0.0, -0.0, -0.004, 1e300, -1e300,
                math.inf, -math.inf, math.nan, 5e-324]
    for chiffres in (1, 2, 7):
        obtenu = _arrondir(valeurs, chiffres)
        for v, o in zip(valeurs, obtenu):
            attendu = round(float(v), chiffres)
            # À l'octet près : même signe du zéro, NaN compris.
            assert struct.pack("<d", o) == struct.pack("<d", attendu), (v, chiffres)


def test_le_classement_par_blocs_rend_le_test_centre_par_centre():
    """Un bloc qu'aucun côté ne touche est tout dedans ou tout dehors :
    vérifié sur des polygones étoilés, troués, multiples, invalides (le
    papillon), en 3D, et sur des côtés qui passent par les centres."""
    import shapely
    from shapely.geometry import MultiPolygon, Polygon, box
    from vue3d.houppiers import BLOC as b, BLOCS_SEUIL, _dedans
    rng = np.random.default_rng(3)
    ny, nx = 250, 300
    lons = 5.6 + (np.arange(nx) + 0.5) * 1e-5
    lats = 43.5 - (np.arange(ny) + 0.5) * 1e-5
    LON, LAT = np.meshgrid(lons, lats)

    def etoile(cx, cy, r, n):
        a = np.sort(rng.uniform(0, 2 * np.pi, n))
        d = rng.uniform(0.3, 1, n) * r
        return list(zip(cx + d * np.cos(a), cy + d * np.sin(a)))

    c = (lons[150], lats[125])
    geoms = [Polygon(etoile(*c, 1.2e-3, 40)),
             Polygon(etoile(*c, 1.5e-3, 200), [etoile(*c, 3e-4, 12)]),
             MultiPolygon([Polygon(etoile(lons[60], lats[60], 5e-4, 30)),
                           Polygon(etoile(lons[220], lats[180], 7e-4, 25))]),
             # Papillon : invalide, GEOS compte les traversées quand même.
             Polygon([(lons[10], lats[10]), (lons[290], lats[240]), (lons[290], lats[10]),
                      (lons[10], lats[240])]),
             # Côtés sur les lignes et les colonnes de centres : centres au bord,
             # au milieu des blocs, puis sur leur première et leur dernière
             # ligne et colonne (blocs de la grille, puis de la sous-fenêtre).
             box(lons[20], lats[230], lons[270], lats[30]),
             box(lons[2 * b], lats[(ny // b - 2) * b - 1], lons[(nx // b - 3) * b - 1],
                 lats[3 * b]),
             box(lons[5 + b], lats[37 + ((ny - 50) // b) * b - 1],
                 lons[5 + ((nx - 50) // b) * b - 1], lats[37 + b]),
             Polygon([(x, y, 12.0) for x, y in etoile(*c, 1e-3, 25)]),
             # Fine lame en diagonale, à travers presque tous les blocs.
             Polygon([(lons[0], lats[0]), (lons[299], lats[249]), (lons[299], lats[247])])]
    fenetre = (slice(0, ny), slice(0, nx))
    for geom in geoms:
        choisies = rng.random((ny, nx)) > 0.3
        assert choisies.sum() >= BLOCS_SEUIL
        jj, ii = _dedans(geom, fenetre, choisies, lons, lats)
        obtenu = np.zeros((ny, nx), dtype=bool)
        obtenu[jj, ii] = True
        attendu = choisies & shapely.contains_xy(geom, LON, LAT)
        assert np.array_equal(obtenu, attendu), geom.wkt[:40]
        # Et dans une fenêtre qui ne commence pas au coin de la grille.
        sous = (slice(37, 241), slice(5, 263))
        jj, ii = _dedans(geom, sous, choisies[sous], lons, lats)
        obtenu = np.zeros((ny, nx), dtype=bool)
        obtenu[jj + 37, ii + 5] = True
        attendu = np.zeros((ny, nx), dtype=bool)
        attendu[sous] = choisies[sous] & shapely.contains_xy(geom, LON[sous], LAT[sous])
        assert np.array_equal(obtenu, attendu), geom.wkt[:40]


def test_le_classement_par_blocs_suit_geos_hors_de_la_boite():
    """Polygone invalide dont un trou sort de la coque, et de la boîte de la
    géométrie : la parité compte dedans les centres du trou, GEOS rejette
    ceux hors de la boîte. Un bloc à cheval sur le bord de la boîte, qu'aucun
    côté ne touche, mêle les deux réponses : sans les côtés de la boîte, son
    témoin, hors de la boîte, le rendait tout dehors."""
    import shapely
    from shapely import affinity
    from shapely.geometry import Polygon
    from vue3d.houppiers import BLOCS_SEUIL, _dedans, _fenetre
    ny, nx = 250, 300
    lons = 5.6 + (np.arange(nx) + 0.5) * 1e-5
    lats = 43.5 - (np.arange(ny) + 0.5) * 1e-5
    LON, LAT = np.meshgrid(lons, lats)

    def piege():
        # Une coque en L, dont le pied va jusqu'à x = 20 entre y = 0 et 1 ;
        # un trou hors de la coque, de x = 12 à 30 entre y = 3 et 9. Une
        # unité vaut huit cellules, un bloc : le bord est de la boîte coupe
        # les blocs de la colonne 176 à 183.
        g = Polygon([(0, 0), (20, 0), (20, 1), (10, 1), (10, 10), (0, 10)],
                    [[(12, 3), (30, 3), (30, 9), (12, 9)]])
        g = affinity.translate(affinity.scale(g, 8e-5, 8e-5, origin=(0, 0)), lons[20], lats[200])
        # GEOS répond au tout premier centre de la boîte d'une géométrie
        # préparée par un localisateur sans index (la coque, puis les trous),
        # qui ne compte pas les traversées : interrogée une fois en un point
        # sans ambiguïté, elle répond ensuite par la parité, partout.
        shapely.contains_xy(g, lons[60], lats[160])
        return g

    geom = piege()
    assert not geom.is_valid
    fenetres = [(slice(0, ny), slice(0, nx)), (slice(37, 241), slice(5, 263)),
                _fenetre(geom, lons, lats)]
    choisies = np.random.default_rng(5).random((ny, nx)) > 0.3
    for fenetre in fenetres:
        sous = choisies[fenetre]
        assert sous.sum() >= BLOCS_SEUIL
        jj, ii = _dedans(piege(), fenetre, sous, lons, lats)
        obtenu = np.zeros(sous.shape, dtype=bool)
        obtenu[jj, ii] = True
        attendu = sous & shapely.contains_xy(piege(), LON[fenetre], LAT[fenetre])
        assert np.array_equal(obtenu, attendu), fenetre
    # Le cas existe bien : des centres du trou, dans la boîte, sont dedans.
    assert shapely.contains_xy(geom, lons[176], lats[160])
    assert not shapely.contains_xy(geom, lons[183], lats[160])
