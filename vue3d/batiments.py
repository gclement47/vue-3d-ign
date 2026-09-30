"""Bâtiments de la scène : les emprises BD TOPO, découpées sur l'emprise.

Le WFS rend un bâtiment entier dès qu'il touche l'emprise, et la BD TOPO fait
un seul objet d'un monument : le château de Versailles y est une emprise de
22 790 m², 233 × 410 m, plus grande que la scène (235 × 356 m). Or tout ce qui
mesure un toit s'arrête au bord de la scène — la grille MNH à 0,5 m n'existe
pas au-delà. Un bâtiment qui débordait ne recevait donc ni pans ni surface
mesurée, et gardait son toit résumé : à Versailles, trois « corps » de 100 m
de côté, une dalle à peine inclinée sous la photo aérienne.

Les bâtiments sont donc découpés, comme l'eau (vue3d/eau.py) : la scène
s'arrête à son bord, et le morceau gardé est un bâtiment ordinaire, que la
grille encadre. Mesuré sur cinq lieux — bâtiments coupés parmi ceux de la
scène, puis ceux d'entre eux qui y gagnent une forme mesurée, pans ou surface :

    Versailles     4 sur 5      3   dont le château, 44 000 cellules de toit
    Strasbourg    56 sur 205   22
    Paris (Cité)  35 sur 111    3
    Gordes        16 sur 247    0
    Chambord       3 sur 10     0

Les autres bâtiments coupés gardent leur toit résumé, comme avant. Un morceau
trop petit pour être mesuré (moins de 5 m² une fois érodé : 2 à Strasbourg, 3
à Paris) retombe sur les hauteurs BD TOPO ; un bâtiment coupé en plusieurs
morceaux (2 à Strasbourg, 3 à Paris) garde son toit résumé, la forme mesurée
demandant une emprise d'un seul tenant.

Le morceau garde les attributs BD TOPO du bâtiment entier — hauteur,
altitudes, usage — et dit qu'il est coupé (`coupe`), pour que la fiche de la
vue ne le présente pas comme un bâtiment complet.
"""

import math

import shapely
from shapely.geometry import box, mapping, shape

from .pans import PANS_DEBORD_M
from .toits import TOITS_RESOLUTION_M, _largeur_min

# Retrait du cadre de découpe sur l'emprise. Un toit mesuré demande des
# cellules au-delà du contour : une colonne de centres pour que la vue
# interpole jusqu'au bord (toits.cellules_du_toit), et PANS_DEBORD_M pour que
# les frontières entre pans traversent franchement l'emprise (pans.py). Au
# ras de la grille, le morceau n'aurait ni l'un ni l'autre.
DECOUPE_RETRAIT_M = PANS_DEBORD_M + TOITS_RESOLUTION_M / 2


def _polygones(geom):
    """Les seuls polygones d'aire non nulle : une découpe peut laisser, au
    bord du cadre, une ligne ou un point de contact."""
    return [g for g in getattr(geom, "geoms", [geom])
            if g.geom_type == "Polygon" and g.area > 0]


def decouper_batiments(batiments, west, south, east, north):
    """Bâtiments découpés sur l'emprise, en retrait de DECOUPE_RETRAIT_M.

    Args:
        batiments: GeoJSON de la couche BD TOPO des bâtiments.

    Returns:
        le même GeoJSON, où un bâtiment qui tient dans le cadre est rendu tel
        quel, un bâtiment qui en sort est remplacé par ce qu'il en reste —
        géométrie 2D, et propriété `coupe` = dict(part, largeur_m) : la part
        de son emprise gardée, et le petit côté du bâtiment ENTIER, auquel se
        juge la pente de son toit (toits.profil_toit) — et un bâtiment hors
        du cadre est retiré.
    """
    lat0 = (south + north) / 2
    m_lon = 111320 * math.cos(math.radians(lat0))
    cadre = box(west + DECOUPE_RETRAIT_M / m_lon, south + DECOUPE_RETRAIT_M / 111320,
                east - DECOUPE_RETRAIT_M / m_lon, north - DECOUPE_RETRAIT_M / 111320)
    gardes = []
    for f in (batiments or {}).get("features", []):
        try:
            geom = shapely.force_2d(shape(f["geometry"]))
        except Exception:
            # Sans géométrie lisible, les toitures et la vue l'ignorent déjà.
            gardes.append(f)
            continue
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
        if cadre.contains(geom):
            gardes.append(f)
            continue
        morceaux = _polygones(geom.intersection(cadre))
        if not morceaux:
            continue
        entier_m = shapely.transform(geom, lambda c: c * [m_lon, 111320])
        coupe = {"part": round(sum(m.area for m in morceaux) / geom.area, 2),
                 "largeur_m": round(_largeur_min(entier_m), 1)}
        gardes.append({**f,
                       "geometry": mapping(shapely.MultiPolygon(morceaux)),
                       "properties": {**(f.get("properties") or {}), "coupe": coupe}})
    return {**(batiments or {}), "features": gardes}
