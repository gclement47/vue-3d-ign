"""Eau de surface : étendues et cours d'eau de la BD TOPO, découpés sur l'emprise.

Deux couches, et une règle pour ne pas dessiner une rivière deux fois :

- `surface_hydrographique` : polygones des lacs, retenues, bassins, mares, et
  des rivières assez larges pour avoir une emprise ;
- `troncon_hydrographique` : les axes. Celui d'une rivière qui a sa surface
  est `fictif` — il la traverse sans rien ajouter —, et on ne le garde pas.
  Restent les ruisseaux, fossés et canaux étroits, qui n'existent qu'en ligne.

Vérifié sur quatre lieux (Petite France à Strasbourg, Pont du Gard, vieil
Annecy, Chambord) : tous les axes des rivières de 15 m et plus y sont fictifs,
et les tronçons non fictifs sont de la classe « Entre 0 et 5 m ». Un tronçon
sous le sol (position -1 : canal couvert, passage sous une place) n'est pas
dessiné.

Le WFS rend l'objet entier dès qu'il touche l'emprise : le lac d'Annecy
arrive complet pour une scène de quelques centaines de mètres. Tout est donc
découpé sur l'emprise avant d'entrer dans la scène.

La hauteur, elle, reste à la vue : l'eau est posée sur le relief de la scène,
comme le sol. Les sommets de la BD TOPO portent bien une altitude, mais d'une
autre source que le relief ; les mêler ferait flotter ou disparaître l'eau.
"""

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

COUCHE_SURFACES_EAU = "BDTOPO_V3:surface_hydrographique"
COUCHE_COURS_EAU = "BDTOPO_V3:troncon_hydrographique"

# Largeur dessinée d'un cours d'eau sans surface, au milieu de sa classe. La
# BD TOPO ne donne qu'une classe ; « Sans objet » et « En attente de mise à
# jour » sont traités comme la plus étroite, seule classe observée hors des
# axes fictifs.
LARGEUR_PAR_CLASSE_M = {
    "Entre 0 et 5 m": 2.5,
    "Entre 5 et 15 m": 10.0,
    "Entre 15 et 50 m": 30.0,
    "Plus de 50 m": 60.0,
}
LARGEUR_PAR_DEFAUT_M = 2.5
# Arrondi des coordonnées : 7 décimales, un centimètre.
DECIMALES = 7


def _arrondir(geom):
    """GeoJSON 2D arrondi : la scène n'a que faire des Z ni des micromètres."""
    def coords(c):
        if isinstance(c[0], (int, float)):
            return [round(c[0], DECIMALES), round(c[1], DECIMALES)]
        return [coords(x) for x in c]
    g = mapping(geom)
    return {"type": g["type"], "coordinates": coords(g["coordinates"])}


def _parties(geom, types):
    """Les seules parties d'un type donné : un découpage peut laisser, au bord
    de l'emprise, un bout de ligne à côté d'un polygone, ou un point de contact
    à côté d'une ligne."""
    if geom.geom_type in types:
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type in types]
    return unary_union(parts) if parts else None


def _sous_le_sol(props):
    try:
        return int(props.get("position_par_rapport_au_sol") or 0) < 0
    except (TypeError, ValueError):
        return False


def eau_pour_emprise(west, south, east, north, surfaces, cours):
    """Étendues et cours d'eau de l'emprise, prêts pour la scène.

    Args:
        surfaces, cours: GeoJSON des couches surface_hydrographique et
            troncon_hydrographique.

    Returns:
        dict(surfaces=[{nature, persistance, geometrie}],
             cours=[{nature, persistance, largeur_m, geometrie}]) ; géométries
        en GeoJSON 2D, découpées sur l'emprise.
    """
    emprise = box(west, south, east, north)
    out = {"surfaces": [], "cours": []}
    for f in (surfaces or {}).get("features", []):
        props = f.get("properties") or {}
        if not f.get("geometry") or _sous_le_sol(props):
            continue
        geom = _parties(shape(f["geometry"]).buffer(0).intersection(emprise),
                        ("Polygon", "MultiPolygon"))
        if geom is None or geom.is_empty or geom.area == 0:
            continue
        out["surfaces"].append({
            "nature": props.get("nature"), "persistance": props.get("persistance"),
            "geometrie": _arrondir(geom),
        })
    for f in (cours or {}).get("features", []):
        props = f.get("properties") or {}
        if not f.get("geometry") or props.get("fictif") or _sous_le_sol(props):
            continue
        geom = _parties(shape(f["geometry"]).intersection(emprise),
                        ("LineString", "MultiLineString"))
        if geom is None or geom.is_empty or geom.length == 0:
            continue
        out["cours"].append({
            "nature": props.get("nature"), "persistance": props.get("persistance"),
            "largeur_m": LARGEUR_PAR_CLASSE_M.get(props.get("classe_de_largeur"),
                                                  LARGEUR_PAR_DEFAUT_M),
            "geometrie": _arrondir(geom),
        })
    return out
