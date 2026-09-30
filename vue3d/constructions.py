"""Réservoirs et constructions ponctuelles de la BD TOPO : citernes, châteaux
d'eau, torchères, cheminées, antennes, mâts.

Deux couches que la couche `batiment` ne contient pas :

- `reservoir` : une emprise et, neuf fois sur dix, une hauteur ;
- `construction_ponctuelle` : un point, une nature, parfois une hauteur.

Elles entrent dans la scène, et non dans une couche chargée après coup, parce
qu'elles touchent au calcul des houppiers. Hors de toute emprise bâtie, un
réservoir est du sursol comme un autre : sur quatre scènes de raffinerie
(Lavéra, Feyzin, Donges, Gonfreville ; 156 réservoirs), 2 426 houppiers et
masses sur 5 400 étaient posés sur une citerne. Leurs emprises rejoignent donc
le masque bâti de la segmentation.

**Hauteur d'un réservoir : la BD TOPO d'abord.** Le MNH LiDAR HD ne garde que
ce que sa classification tient pour du bâti ou de la végétation : à Feyzin, 14
citernes sur 30, bien visibles à l'orthophoto avec leur ombre, y sont à zéro
d'un bord à l'autre. Là où il les voit, il confirme la BD TOPO — écart médian
de 0,5 m entre sa médiane sur l'emprise et la hauteur déclarée, sur 130
réservoirs. Il ne sert donc que de repli, pour les 8 % de réservoirs sans
hauteur (52 sur 669 autour de douze lieux).

**Constructions ponctuelles : le LiDAR en voit une partie.** Mesuré sur 246
points autour de douze lieux, dans un rayon de 3 m (la précision planimétrique
annoncée est de 2,5 m) :

- torchères : le maximum du MNH retrouve la hauteur BD TOPO à 3 % près (8
  fois sur 11), ou tombe sur tout autre chose — la moitié, plus du double —
  quand le point voisine une structure plus haute ;
- torchères, cheminées, « autres constructions élevées » : 109 des 127
  points hors bâtiment n'ont pas de hauteur BD TOPO (71 torchères sur 86,
  bâtiments compris), et le MNH en donne une à 57 d'entre eux ;
- antennes : le treillis échappe au MNH — sur 20 antennes hors bâtiment de
  hauteur déclarée, il n'en voit pas 8 et lit moins des trois quarts de la
  hauteur de 9 autres —, mais la BD TOPO donne leur hauteur 24 fois sur 26.

Un point dans une emprise bâtie (37 % des torchères, 34 % des cheminées, 21
clochers sur 24) n'est pas dessiné : le toit mesuré du bâtiment le porte déjà.
Clochers et minarets sont donc laissés à leur bâtiment ; croix et calvaires
n'ont jamais de hauteur (0 sur 14) et le MNH n'y lit que les arbres voisins ;
un transformateur n'est pas une construction élevée ; la hauteur d'une
éolienne (renseignée 45 fois sur 300) ne dit pas ce qu'elle mesure, du mât ou
du bout de pale. Aucun de ceux-là n'est repris.

Une construction sans hauteur déclarée ni mesurée n'est pas dessinée : on
n'invente pas un mât. Un réservoir sans hauteur garde son emprise, que la vue
dessine pâle, comme un bâtiment de hauteur inconnue.
"""

import logging
import math

import numpy as np
import shapely
from shapely.geometry import Point, mapping, shape

from .batiments import decouper_batiments
from .mnh import MNH_SEUIL_M

journal = logging.getLogger(__name__)

COUCHE_RESERVOIRS = "BDTOPO_V3:reservoir"
COUCHE_PONCTUELLES = "BDTOPO_V3:construction_ponctuelle"

# Natures dessinées ; les autres sont écartées, raisons en tête de module.
NATURES_PONCTUELLES = ("Torchère", "Cheminée", "Antenne", "Autre construction élevée")

# Dilatation des réservoirs dans le masque bâti des houppiers. Le contour BD
# TOPO (précision 2,5 m) ne suit pas la robe au demi-mètre : sur 156
# réservoirs, 63 % des cellules de la première couronne d'un mètre dépassent
# encore 2 m (médiane par réservoir), 23 % de la deuxième, 7 % de la
# troisième. Houppiers et masses des quatre scènes, selon la dilatation :
#
#   sans masque   0 m     1 m     2 m     3 m     4 m
#      5 400     3 630   2 929   2 660   2 558   2 488
#
# Le deuxième mètre en retire encore 269, le troisième 102, le quatrième 70 :
# à 2 m la robe est partie, et ce qu'on retire au-delà est la tuyauterie.
# Mesure : python outils/mesure_constructions.py
MASQUE_RESERVOIR_M = 2.0
# Le LiDAR « voit » un réservoir quand la moitié au moins de son emprise
# dépasse le seuil du sursol : en deçà, la médiane de ses cellules hautes est
# celle d'un voisin ou d'un bout de robe. Sur 154 réservoirs mesurés, 123
# dépassent 80 %, 10 sont entre 50 et 80 %, 21 en dessous, dont 15 à zéro.
RESERVOIR_PART_VUE = 0.5

# Rayon de recherche du sommet d'une construction ponctuelle dans le MNH : la
# précision planimétrique de la BD TOPO, 2,5 m, et une maille.
SOMMET_RAYON_M = 3.0
# Le LiDAR confirme la hauteur déclarée quand il la retrouve à 15 % près. Sur
# 27 points qui ont les deux, le rapport vaut 0,97 à 1,01 (11 fois), 0,89
# (1) ; tous les autres sont à 0,81 ou moins — le treillis d'une antenne, lu
# en partie — ou à 2,3, le sommet d'une structure voisine.
ACCORD_HAUTEUR = 0.15
# L'emprise au sol d'une construction est la tache du MNH qui tient au sommet
# et dépasse la moitié de sa hauteur. Sur les 66 torchères, cheminées et
# constructions élevées que le LiDAR voit, elle s'étend à 4,4 m du point en
# médiane, et à 8 m au plus pour 52 d'entre elles. Des 14 autres, 12 passent
# 12 m et touchent pour la plupart le bord de la fenêtre : la tache a fusionné
# avec un bâtiment ou un portique voisin et ne mesure plus rien. Rayon
# équivalent des 52 retenues : 2,8 m en médiane (1,7 à 3,8 m du 1er au 9e
# décile).
ETENDUE_MAX_M = 8.0
# Fenêtre où la tache est cherchée : assez pour constater qu'elle déborde.
FENETRE_M = 12.0
# Arrondi des coordonnées : 7 décimales, un centimètre.
DECIMALES = 7


def _echelles(south, north):
    """Mètres par degré de longitude et de latitude au milieu de l'emprise."""
    return 111320 * math.cos(math.radians((south + north) / 2)), 111320


def _en_metres(geom, echelles):
    return shapely.transform(geom, lambda c: c * list(echelles))


def _en_degres(geom, echelles):
    return shapely.transform(geom, lambda c: c / list(echelles))


class _Grille:
    """La grille MNH, avec de quoi en lire une fenêtre autour d'une emprise."""

    def __init__(self, grille):
        self.west, self.south, self.east, self.north = grille["bbox"]
        self.nx, self.ny = grille["width"], grille["height"]
        self.H = np.asarray(grille["values"], dtype=np.float32).reshape(self.ny, self.nx)
        self.seuil = float(grille.get("seuil_m") or MNH_SEUIL_M)
        self.source = grille.get("source")
        self.lons = self.west + (self.east - self.west) * (np.arange(self.nx) + 0.5) / self.nx
        self.lats = self.north - (self.north - self.south) * (np.arange(self.ny) + 0.5) / self.ny

    def fenetre(self, west, south, east, north):
        """(hauteurs, LON, LAT) des cellules dont le centre est dans la boîte."""
        i = np.nonzero((self.lons >= west) & (self.lons <= east))[0]
        j = np.nonzero((self.lats >= south) & (self.lats <= north))[0]
        if not i.size or not j.size:
            return None
        js, is_ = slice(j[0], j[-1] + 1), slice(i[0], i[-1] + 1)
        LON, LAT = np.meshgrid(self.lons[is_], self.lats[js])
        return self.H[js, is_], LON, LAT


def hauteur_lidar_reservoir(geom, grille):
    """Hauteur d'un réservoir lue dans le MNH : la médiane des cellules de son
    emprise qui dépassent le seuil du sursol ; None si le LiDAR ne le voit pas
    (RESERVOIR_PART_VUE)."""
    f = grille.fenetre(*geom.bounds)
    if f is None:
        return None
    H, LON, LAT = f
    v = H[shapely.contains_xy(geom, LON, LAT)]
    hautes = v[v >= grille.seuil]
    if not v.size or hautes.size < RESERVOIR_PART_VUE * v.size:
        return None
    return round(float(np.median(hautes)), 1)


def sommet_lidar(lon, lat, grille, echelles):
    """Ce que le MNH montre autour d'un point : (hauteur, rayon, étendue).

    `hauteur` est le maximum à moins de SOMMET_RAYON_M ; None sous le seuil
    du sursol. `rayon` est le rayon équivalent de la tache qui tient à ce
    sommet et dépasse la moitié de sa hauteur, `etendue` la distance du point
    à sa cellule la plus lointaine, en mètres.
    """
    kx, ky = echelles
    f = grille.fenetre(lon - FENETRE_M / kx, lat - FENETRE_M / ky,
                       lon + FENETRE_M / kx, lat + FENETRE_M / ky)
    if f is None:
        return None, None, None
    H, LON, LAT = f
    d = np.hypot((LON - lon) * kx, (LAT - lat) * ky)
    proche = d <= SOMMET_RAYON_M
    if not proche.any():
        return None, None, None
    h = float(np.where(proche, H, -1.0).max())
    if h < grille.seuil:
        return None, None, None
    haut = H >= max(h / 2, grille.seuil)
    tache = np.zeros(H.shape, dtype=bool)
    tache[np.unravel_index(np.argmax(np.where(proche, H, -1.0)), H.shape)] = True
    while True:
        # Croissance en croix, bornée aux cellules hautes, jusqu'à stabilité.
        suite = tache.copy()
        suite[1:] |= tache[:-1]
        suite[:-1] |= tache[1:]
        suite[:, 1:] |= tache[:, :-1]
        suite[:, :-1] |= tache[:, 1:]
        suite &= haut
        if (suite == tache).all():
            break
        tache = suite
    pas = (grille.east - grille.west) * kx / grille.nx
    return (round(h, 1), round(math.sqrt(tache.sum() * pas * pas / math.pi), 1),
            float(d[tache].max()))


def _anneau(coords):
    return [[round(c[0], DECIMALES), round(c[1], DECIMALES)] for c in coords]


def _reservoirs(west, south, east, north, reservoirs, grille, echelles):
    """(réservoirs de la scène, emprises entières à masquer)."""
    out, masque = [], []
    for f in (reservoirs or {}).get("features", []):
        try:
            geom = shapely.force_2d(shape(f["geometry"]))
        except Exception:
            continue
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
        if geom.is_empty or not geom.area:
            continue
        # Entier dans le masque, comme les bâtiments (vue3d/scene.py) : dans
        # le retrait de la découpe, la robe passerait pour du sursol.
        masque.append(_en_degres(_en_metres(geom, echelles).buffer(MASQUE_RESERVOIR_M), echelles))
        props = f.get("properties") or {}
        h, source = props.get("hauteur"), "bdtopo"
        if not h or h <= 0:
            h = hauteur_lidar_reservoir(geom, grille)
            source = grille.source if h else None
        (coupe,) = decouper_batiments({"features": [{"geometry": mapping(geom), "properties": {}}]},
                                      west, south, east, north)["features"] or [None]
        if coupe is None:
            continue
        morceaux = shape(coupe["geometry"])
        for poly in getattr(morceaux, "geoms", [morceaux]):
            if poly.geom_type != "Polygon" or not poly.area:
                continue
            out.append({
                "nature": props.get("nature"), "nom": props.get("toponyme"),
                "h": round(float(h), 1) if h else None, "source": source,
                "coupe": "coupe" in coupe["properties"],
                "contour": _anneau(poly.exterior.coords),
                "trous": [_anneau(t.coords) for t in poly.interiors],
            })
    return out, masque


def _ponctuelles(west, south, east, north, ponctuelles, batiments, grille, echelles):
    """(constructions ponctuelles de la scène, disques à masquer)."""
    bati = []
    for f in (batiments or {}).get("features", []):
        try:
            bati.append(shapely.force_2d(shape(f["geometry"])))
        except Exception:
            continue
    out, masque = [], []
    for f in (ponctuelles or {}).get("features", []):
        props = f.get("properties") or {}
        geom = f.get("geometry") or {}
        if props.get("nature") not in NATURES_PONCTUELLES or geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        if not (west <= lon <= east and south <= lat <= north):
            continue
        point = Point(lon, lat)
        if any(b.contains(point) for b in bati):
            continue
        h_lidar, rayon, etendue = sommet_lidar(lon, lat, grille, echelles)
        h, source = props.get("hauteur"), "bdtopo"
        if not h or h <= 0:
            h, source = h_lidar, grille.source
        if not h:
            continue
        # La tache du MNH n'est celle de la construction que si son sommet
        # est le sien, et qu'elle ne s'est pas fondue dans une voisine.
        vue = (h_lidar is not None and abs(h_lidar - h) <= ACCORD_HAUTEUR * h
               and etendue <= ETENDUE_MAX_M)
        if vue:
            pas = (grille.east - grille.west) * echelles[0] / grille.nx
            disque = Point(lon * echelles[0], lat * echelles[1]).buffer(etendue + pas)
            masque.append(_en_degres(disque, echelles))
        out.append({
            "lon": round(lon, DECIMALES), "lat": round(lat, DECIMALES),
            "nature": props.get("nature"), "detail": props.get("nature_detaillee"),
            "nom": props.get("toponyme"),
            "h": round(float(h), 1), "source": source,
            # Rayon mesuré de l'emprise au sol ; None si le LiDAR ne la voit
            # pas, et la vue prend alors celui de la nature.
            "r": rayon if vue else None,
        })
    return out, masque


def constructions_pour_emprise(west, south, east, north, reservoirs, ponctuelles,
                               batiments, grille):
    """Réservoirs et constructions ponctuelles de l'emprise, prêts pour la scène.

    Args:
        reservoirs, ponctuelles: GeoJSON des couches reservoir et
            construction_ponctuelle.
        batiments: GeoJSON des bâtiments ENTIERS, pour écarter les points
            qu'un toit porte déjà.
        grille: grille MNH de la scène, pour les hauteurs de repli.

    Returns:
        (constructions, masque) :
        - constructions = dict(reservoirs=[{nature, nom, h, source, coupe,
          contour, trous}], ponctuelles=[{lon, lat, nature, detail, nom, h,
          source, r}]) ; `source` vaut "bdtopo", ou celle de la grille quand
          la hauteur y est lue, ou None pour un réservoir de hauteur inconnue ;
        - masque = GeoJSON des emprises à retirer du sursol avant de segmenter
          les houppiers, à joindre aux bâtiments. Jamais embarqué.
    """
    echelles = _echelles(south, north)
    g = _Grille(grille)
    cuves, masque_cuves = _reservoirs(west, south, east, north, reservoirs, g, echelles)
    points, masque_points = _ponctuelles(west, south, east, north, ponctuelles,
                                         batiments, g, echelles)
    masque = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(m)}
        for m in masque_cuves + masque_points]}
    journal.info("Constructions : %d réservoir(s), %d ponctuelle(s)", len(cuves), len(points))
    return {"reservoirs": cuves, "ponctuelles": points}, masque
