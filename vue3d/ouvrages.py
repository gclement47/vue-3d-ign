"""Ouvrages de la BD TOPO : murs, ponts, voies ferrées et terrains de sport.

Une couche à part de la scène, comme les monuments OSM (vue3d/monuments.py) :
la vue la demande une fois la scène affichée. Rien ici n'entre dans un calcul
de la scène — ni toit, ni houppier — et la scène n'a donc ni à l'attendre, ni
à échouer avec elle, ni à être reconstruite quand la couche change : elle a sa
propre version (OUVRAGES_VERSION) et son propre fichier de cache.

Quatre couches, lues sur l'emprise de la scène et découpées sur elle :

- `troncon_de_voie_ferree` : voies ferrées et tramways, en rubans ;
- `terrain_de_sport` : terrains, pistes et bassins, en aplats sur le sol ;
- `construction_lineaire`, natures « Mur » et « Pont » ;
- `construction_surfacique`, nature « Pont ».

**La hauteur d'un mur ou d'un pont est l'altitude de ses sommets.** Aucune de
ces couches n'a d'attribut de hauteur, mais leurs sommets sont en 3D, et leur
altitude est celle du HAUT de l'ouvrage : l'aqueduc du Pont du Gard y est à
66-67 m, soit 47,8 m au-dessus du Gardon pour 48 m de hauteur réelle ; les 30
murs des remparts de Carcassonne à 3 à 25 m au-dessus du relief, 8,8 m en
médiane. La hauteur est donc l'altitude du sommet moins le relief de la scène
à cet endroit. Ce sont deux sources, et l'écart entre elles passe dans la
hauteur : la BD TOPO annonce 1,5 m de précision altimétrique. Mesuré à
Carcassonne contre le MNH LiDAR (9e décile à 1,5 m de l'axe du mur), l'écart
est de 1,4 m en médiane, la hauteur ainsi comptée étant plutôt en dessous
(−1,1 m). D'où le seuil ci-dessous, et l'abandon des murs de soutènement,
dont toute la hauteur est une marche du relief que le RGE ALTI lisse (à
Gordes, 3 murs de 4 à 6,5 m, à cheval sur leur restanque).

Ce que la couche ne dessine pas : les tunnels (altitude absente, −1 000), les
quais, escaliers et dalles, posés sur un sol que le relief porte déjà, et tout
ouvrage « en projet ». Sans relief RGE ALTI dans la scène, ni mur ni pont : une
altitude ne se compare pas au repli mondial de la vue, à 30 m de maille.

**Masses expliquées.** Le MNH voit un rempart ou un tablier comme du sursol
non végétal, que la scène a segmenté en « masses ». Celles qui tombent sur un
ouvrage dessiné sont désignées à la vue, qui les retire : l'ouvrage les
remplace, comme une partie OSM remplace un bâtiment.
"""

import logging
import math
import statistics

import shapely
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape

from .couches import lire_couche
from .relief import echantillonneur

journal = logging.getLogger(__name__)

# Format de la couche. L'incrémenter ne refait que la couche, pas les scènes :
# le numéro est dans le nom du fichier de cache (scene.NOM_OUVRAGES).
OUVRAGES_VERSION = 1

COUCHES = {
    "lineaires": "BDTOPO_V3:construction_lineaire",
    "surfaciques": "BDTOPO_V3:construction_surfacique",
    "voies": "BDTOPO_V3:troncon_de_voie_ferree",
    "terrains": "BDTOPO_V3:terrain_de_sport",
}

# Sommet sans altitude : la BD TOPO y met −1 000 (tunnels de la Part-Dieu,
# téléskis de Chamonix, « méthode d'acquisition altimétrique : pas de Z »).
Z_ABSENT = -500.0
# Pas d'échantillonnage du relief le long d'un ouvrage : la maille du terrain
# de la vue.
PAS_M = 2.0
# En deçà, un mur n'est pas dessiné : c'est le seuil du sursol (mnh.MNH_SEUIL_M)
# et, à 1,5 m de précision altimétrique, la limite de ce que l'altitude d'un
# sommet sait dire. Hauteur médiane le long du mur.
MUR_HAUTEUR_MIN_M = 2.0
# Un pont est dessiné si son tablier passe quelque part à cette hauteur
# au-dessus du relief ; en dessous, le relief le porte déjà (à Carcassonne, un
# pont de 5 m de long, 1,3 m SOUS le relief). Celui de la gare de la
# Part-Dieu, 4 634 m², est à +0,7 m en médiane sur son remblai et à 6,4 m
# au-dessus des rues qu'il franchit : il est dessiné.
PONT_HAUTEUR_MIN_M = 2.0
# Largeur d'un pont en ligne : celle de la route qu'il porte, quand un tronçon
# hors sol le suit sur la moitié au moins de sa longueur (3 des 5 ponts
# dessinés dans dix scènes : 4, 4 et 5 m de chaussée). Sinon, celle d'une
# route à une chaussée, la médiane mesurée sur cinq scènes (vue :
# LARGEUR_ROUTE_PAR_NATURE).
PONT_LARGEUR_PAR_DEFAUT_M = 3.0
PONT_ROUTE_RAYON_M = 3.0
PONT_ROUTE_PART = 0.5
# Largeur dessinée d'une voie ferrée, par voie : une convention, l'ordre de
# grandeur d'une plateforme ballastée. La BD TOPO ne donne que le nombre de
# voies et la classe d'écartement. À la Part-Dieu, où chaque voie a son
# tronçon, l'entraxe mesuré entre tronçons voisins est de 7,3 m en médiane
# (quais compris) : à 4 m, les rubans ne se recouvrent pas.
LARGEUR_PAR_VOIE_M = {"Normale": 4.0, "Large": 4.0, "Etroite": 3.0}
# Masses du MNH expliquées par un ouvrage : à moins de MASSE_MUR_M d'un mur,
# ou de MASSE_PONT_MARGE_M du bord d'un tablier. Mesuré par couronnes d'un
# mètre, masses par mètre d'éloignement :
#
#   Carcassonne, murs (2 130 masses)      176, 57, 30, 23, 30, 29, 24, 29
#   Pont du Gard, ponts (257 masses)       58, 46, 26, 18,  3,  5,  6,  1
#
# Le palier est le fond du tissu alentour : il commence à 2 m d'un mur, à 3 ou
# 4 m de l'axe d'un pont de 3 à 5 m de large. Avec ces marges, 236 masses
# expliquées à Carcassonne, 135 au Pont du Gard.
# Mesure : python outils/mesure_constructions.py
MASSE_MUR_M = 2.0
MASSE_PONT_MARGE_M = 1.5
ETATS_ECARTES = ("En projet", "En construction")
# Arrondi des coordonnées : 7 décimales, un centimètre.
DECIMALES = 7


def fetch_ouvrages(west, south, east, north):
    """GeoJSON bruts des quatre couches sur l'emprise.

    Raises:
        requests.RequestException si l'une d'elles n'a pas pu être lue : la
        couche est complète ou n'existe pas, comme la scène.
    """
    return {cle: lire_couche(couche, west, south, east, north)
            for cle, couche in COUCHES.items()}


def _z(c):
    return c[2] if len(c) > 2 and c[2] is not None and c[2] > Z_ABSENT else None


def _lignes(geom):
    if geom.get("type") == "LineString":
        return [geom["coordinates"]]
    if geom.get("type") == "MultiLineString":
        return geom["coordinates"]
    return []


def _polygones(geom):
    if geom.get("type") == "Polygon":
        return [geom["coordinates"]]
    if geom.get("type") == "MultiPolygon":
        return geom["coordinates"]
    return []


def decouper_ligne(coords, west, south, east, north):
    """Morceaux d'une ligne 3D dans l'emprise.

    shapely découpe en 2D ; ici l'altitude d'un point de coupe est
    interpolée le long de son segment, puisque c'est elle qui fait la hauteur.

    Returns:
        liste de morceaux [(lon, lat, z), ...] ; z None si le sommet n'a pas
        d'altitude.
    """
    pts = [(c[0], c[1], _z(c)) for c in coords]

    def entre(a, b, t):
        z = None if a[2] is None or b[2] is None else a[2] + (b[2] - a[2]) * t
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, z)

    morceaux, courant = [], []
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t0, t1, dedans = 0.0, 1.0, True
        # Liang-Barsky : le segment contre chacun des quatre bords.
        for p, q in ((-dx, a[0] - west), (dx, east - a[0]), (-dy, a[1] - south), (dy, north - a[1])):
            if p == 0:
                dedans = dedans and q >= 0
            elif p < 0:
                t0 = max(t0, q / p)
            else:
                t1 = min(t1, q / p)
        if not dedans or t0 >= t1:
            if courant:
                morceaux.append(courant)
                courant = []
            continue
        if t0 > 0 or not courant:
            if courant:
                morceaux.append(courant)
            courant = [entre(a, b, t0)]
        courant.append(entre(a, b, t1))
        if t1 < 1:
            morceaux.append(courant)
            courant = []
    if courant:
        morceaux.append(courant)
    return morceaux


def _z_sur_contour(x, y, anneaux):
    """Altitude d'un point du contour découpé : celle du sommet d'origine s'il
    en est un, sinon interpolée sur le segment d'origine le plus proche."""
    meilleur = (math.inf, None)
    for anneau in anneaux:
        for a, b in zip(anneau, anneau[1:]):
            za, zb = _z(a), _z(b)
            if za is None or zb is None:
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            l2 = dx * dx + dy * dy
            t = 0.0 if l2 == 0 else min(max(((x - a[0]) * dx + (y - a[1]) * dy) / l2, 0.0), 1.0)
            d = math.hypot(x - a[0] - t * dx, y - a[1] - t * dy)
            if d < meilleur[0]:
                meilleur = (d, za + (zb - za) * t)
    return meilleur[1]


def _arrondir(pts, avec_z=True):
    if avec_z:
        return [[round(p[0], DECIMALES), round(p[1], DECIMALES), round(p[2], 1)] for p in pts]
    return [[round(p[0], DECIMALES), round(p[1], DECIMALES)] for p in pts]


def _arrondir_2d(geom):
    """GeoJSON 2D arrondi au centimètre."""
    def coords(c):
        if isinstance(c[0], (int, float)):
            return [round(c[0], DECIMALES), round(c[1], DECIMALES)]
        return [coords(x) for x in c]
    g = mapping(geom)
    return {"type": g["type"], "coordinates": coords(g["coordinates"])}


def ouvrages_pour_emprise(west, south, east, north, brut, relief=None, masses=None,
                          routes=None):
    """Ouvrages de l'emprise, prêts pour la vue.

    Args:
        brut: GeoJSON des quatre couches, de `fetch_ouvrages`.
        relief: relief embarqué de la scène (vue3d/relief.py) ; sans lui, ni
            mur ni pont.
        masses: masses de sursol de la scène, pour désigner celles qu'un
            ouvrage explique.
        routes: GeoJSON des routes de la scène, pour la largeur des ponts.

    Returns:
        dict(version, murs, ponts, voies, terrains, masses_expliquees) ; None
        si l'emprise n'a aucun ouvrage à dessiner.
        - murs : [{nature, detail, nom, h, ligne}] — `ligne` en [lon, lat, z],
          z altitude du haut du mur, h sa hauteur médiane au-dessus du relief ;
        - ponts : [{nature, detail, nom, h, largeur_m, largeur_mesuree, ligne}]
          ou [{nature, detail, nom, h, contour, trous}] selon la couche ; h est
          la plus grande hauteur du tablier au-dessus du relief ;
        - voies : [{nature, voies, largeur_m, au_sol, ligne}] — `ligne` en
          [lon, lat] au sol, en [lon, lat, z] sur un ouvrage ;
        - terrains : [{nature, detail, geometrie}], GeoJSON 2D ;
        - masses_expliquees : index dans `masses`.
    """
    brut = brut or {}
    kx = 111320 * math.cos(math.radians((south + north) / 2))
    ky = 111320
    cadre = box(west, south, east, north)
    altitude = echantillonneur(relief)

    def props_de(f):
        p = f.get("properties") or {}
        return None if p.get("etat_de_l_objet") in ETATS_ECARTES else p

    def hauteurs(pts):
        """Hauteurs au-dessus du relief le long d'une ligne 3D, tous les PAS_M."""
        out = []
        for a, b in zip(pts, pts[1:]):
            n = max(1, math.ceil(math.hypot((b[0] - a[0]) * kx, (b[1] - a[1]) * ky) / PAS_M))
            for k in range(n + 1):
                t = k / n
                sol = altitude(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                if sol is not None:
                    out.append(a[2] + (b[2] - a[2]) * t - sol)
        return out

    def en_m(pts):
        return [(p[0] * kx, p[1] * ky) for p in pts]

    murs, ponts, emprises = [], [], []        # emprises : zones où une masse est expliquée
    chaussees = []
    for f in (routes or {}).get("features", []):
        p = f.get("properties") or {}
        if str(p.get("position_par_rapport_au_sol") or "0") != "0" and (p.get("largeur_de_chaussee") or 0) > 0:
            for l in _lignes(f.get("geometry") or {}):
                if len(l) >= 2:
                    chaussees.append((LineString(en_m(l)), p["largeur_de_chaussee"]))

    if altitude is not None:
        for f in (brut.get("lineaires") or {}).get("features", []):
            p = props_de(f)
            if p is None or p.get("nature") not in ("Mur", "Pont"):
                continue
            entier = _lignes(f.get("geometry") or {})
            for coords in entier:
                for pts in decouper_ligne(coords, west, south, east, north):
                    if any(q[2] is None for q in pts):
                        continue
                    hs = hauteurs(pts)
                    if not hs:
                        continue
                    commun = {"nature": p["nature"], "detail": p.get("nature_detaillee"),
                              "nom": p.get("toponyme")}
                    axe = LineString(en_m(pts))
                    if p["nature"] == "Mur":
                        h = statistics.median(hs)
                        if h < MUR_HAUTEUR_MIN_M:
                            continue
                        murs.append({**commun, "h": round(h, 1), "ligne": _arrondir(pts)})
                        emprises.append(axe.buffer(MASSE_MUR_M))
                    else:
                        if max(hs) < PONT_HAUTEUR_MIN_M:
                            continue
                        # La route portée se cherche le long du pont entier.
                        pont = LineString(en_m(coords))
                        largeur = next((w for ligne, w in chaussees
                                        if ligne.intersection(pont.buffer(PONT_ROUTE_RAYON_M)).length
                                        >= PONT_ROUTE_PART * pont.length), None)
                        ponts.append({**commun, "h": round(max(hs), 1),
                                      "largeur_m": largeur or PONT_LARGEUR_PAR_DEFAUT_M,
                                      "largeur_mesuree": largeur is not None,
                                      "ligne": _arrondir(pts)})
                        emprises.append(axe.buffer((largeur or PONT_LARGEUR_PAR_DEFAUT_M) / 2
                                                   + MASSE_PONT_MARGE_M))
        for f in (brut.get("surfaciques") or {}).get("features", []):
            p = props_de(f)
            if p is None or p.get("nature") != "Pont":
                continue
            for anneaux in _polygones(f.get("geometry") or {}):
                try:
                    poly = Polygon([c[:2] for c in anneaux[0]],
                                   [[c[:2] for c in t] for t in anneaux[1:]]).buffer(0)
                except Exception:
                    continue
                decoupe = poly.intersection(cadre)
                for morceau in getattr(decoupe, "geoms", [decoupe]):
                    if morceau.geom_type != "Polygon" or not morceau.area:
                        continue
                    tours = [[(x, y, _z_sur_contour(x, y, anneaux)) for x, y in t.coords]
                             for t in (morceau.exterior, *morceau.interiors)]
                    if any(q[2] is None for t in tours for q in t):
                        continue
                    hs = hauteurs(tours[0])
                    if not hs or max(hs) < PONT_HAUTEUR_MIN_M:
                        continue
                    ponts.append({"nature": "Pont", "detail": p.get("nature_detaillee"),
                                  "nom": p.get("toponyme"), "h": round(max(hs), 1),
                                  "contour": _arrondir(tours[0]),
                                  "trous": [_arrondir(t) for t in tours[1:]]})
                    emprises.append(Polygon(en_m(tours[0])).buffer(MASSE_PONT_MARGE_M))

    voies = []
    for f in (brut.get("voies") or {}).get("features", []):
        p = props_de(f)
        if p is None:
            continue
        try:
            niveau = int(p.get("position_par_rapport_au_sol") or 0)
        except (TypeError, ValueError):
            niveau = 0
        if niveau < 0:
            continue                      # sous le sol : tunnel, tranchée couverte
        n_voies = p.get("nombre_de_voies") or 1
        for coords in _lignes(f.get("geometry") or {}):
            for pts in decouper_ligne(coords, west, south, east, north):
                # Sur un ouvrage, la voie est dessinée à son altitude ; posée
                # sur le relief, elle plongerait avec lui sous le viaduc.
                if niveau > 0 and (altitude is None or any(q[2] is None for q in pts)):
                    continue
                voies.append({"nature": p.get("nature"), "voies": n_voies,
                              "largeur_m": n_voies * LARGEUR_PAR_VOIE_M.get(p.get("largeur"), 4.0),
                              "au_sol": niveau == 0,
                              "ligne": _arrondir(pts, avec_z=niveau > 0)})

    terrains = []
    for f in (brut.get("terrains") or {}).get("features", []):
        p = props_de(f)
        if p is None or not f.get("geometry"):
            continue
        try:
            geom = shapely.force_2d(shape(f["geometry"])).buffer(0).intersection(cadre)
        except Exception:
            continue
        morceaux = [g for g in getattr(geom, "geoms", [geom])
                    if g.geom_type == "Polygon" and g.area > 0]
        if morceaux:
            terrains.append({"nature": p.get("nature"), "detail": p.get("nature_detaillee"),
                             "geometrie": _arrondir_2d(shapely.MultiPolygon(morceaux))})

    if not (murs or ponts or voies or terrains):
        return None
    zone = shapely.unary_union(emprises) if emprises else None
    expliquees = [i for i, m in enumerate(masses or [])
                  if zone is not None and zone.contains(Point(m["lon"] * kx, m["lat"] * ky))]
    journal.info("Ouvrages : %d mur(s), %d pont(s), %d voie(s), %d terrain(s), %d masse(s) expliquée(s)",
                 len(murs), len(ponts), len(voies), len(terrains), len(expliquees))
    return {"version": OUVRAGES_VERSION, "murs": murs, "ponts": ponts, "voies": voies,
            "terrains": terrains, "masses_expliquees": expliquees}
