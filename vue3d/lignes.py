"""Lignes électriques à haute tension de la BD TOPO, avec leurs supports.

La BD TOPO ne décrit que le réseau de transport (RTE, 63 à 400 kV) : les
lignes de distribution d'Enedis, sur poteaux devant les maisons, n'y sont pas.

Deux couches, qui ne se recouvrent qu'en partie :

- `ligne_electrique` : l'axe de chaque ligne. Ses sommets sont les supports —
  l'écart entre sommets est celui des pylônes, 249 m en médiane en 63 kV,
  335 m en 225 kV, 494 m en 400 kV (mesuré sur 1 538 portées autour de
  Versailles, du Pont du Gard et de Carcassonne) ;
- `pylone` : des points avec une hauteur, mais pour une part seulement des
  supports (13 pylônes pour 269 sommets autour de Carcassonne).

Chaque sommet reçoit donc la hauteur du pylône BD TOPO posé dessus, et à
défaut la hauteur médiane des pylônes de sa tension.

Le WFS rend une ligne dès que sa boîte englobante touche l'emprise : autour de
Gordes, une ligne de 400 kV dont aucun point ne passe à moins de 22 km. Seules
les portées qui traversent vraiment l'emprise sont gardées, avec leurs deux
supports, même celui qui tombe au-dehors : le câble doit aller jusqu'au bord.
"""

import math

from shapely.geometry import LineString, box

COUCHE_LIGNES = "BDTOPO_V3:ligne_electrique"
COUCHE_PYLONES = "BDTOPO_V3:pylone"

# Hauteur médiane des pylônes BD TOPO rattachés à une ligne de cette tension
# (à moins de 10 m d'un de ses sommets), sur les trois mêmes zones :
#
#   tension   pylônes   p25    médiane   p75
#   63 kV       323     15,9    23,0     26,8 m
#   225 kV       89     29,7    35,2     47,5 m
#   400 kV      112     45,0    50,5     52,1 m
HAUTEUR_PAR_TENSION_M = {"63 kV": 23.0, "90 kV": 23.0, "150 kV": 35.2,
                         "225 kV": 35.2, "400 kV": 50.5}
# Tension inconnue : la plus basse, celle des lignes les plus nombreuses.
HAUTEUR_PAR_DEFAUT_M = 23.0
# Un pylône BD TOPO est rattaché au sommet de ligne le plus proche en deçà.
PYLONE_RAYON_M = 10.0


def _metres(lat0):
    kx = 111320 * math.cos(math.radians(lat0))
    return lambda a, b: math.hypot((a[0] - b[0]) * kx, (a[1] - b[1]) * 111320)


def lignes_pour_emprise(west, south, east, north, lignes, pylones):
    """Lignes à haute tension qui traversent l'emprise, supports compris.

    Args:
        lignes, pylones: GeoJSON des couches ligne_electrique et pylone, lues
            sur une emprise au moins aussi grande.

    Returns:
        liste de {tension, gestionnaire, supports: [[lon, lat, hauteur_m,
        mesuree], ...]} — une entrée par suite continue de portées qui
        touchent l'emprise ; `mesuree` dit si la hauteur vient d'un pylône BD
        TOPO ou de la médiane de la tension.
    """
    emprise = box(west, south, east, north)
    distance = _metres((south + north) / 2)
    pts_pylones = [((p.get("geometry") or {}).get("coordinates"), (p.get("properties") or {}).get("hauteur"))
                   for p in (pylones or {}).get("features", [])]
    pts_pylones = [(c[:2], h) for c, h in pts_pylones if c and h]

    def hauteur(sommet, tension):
        proches = [(distance(sommet, c), h) for c, h in pts_pylones]
        proches = [x for x in proches if x[0] <= PYLONE_RAYON_M]
        if proches:
            return min(proches)[1], True
        return HAUTEUR_PAR_TENSION_M.get(tension, HAUTEUR_PAR_DEFAUT_M), False

    out = []
    for f in (lignes or {}).get("features", []):
        geom = f.get("geometry") or {}
        props = f.get("properties") or {}
        if geom.get("type") == "LineString":
            troncons = [geom["coordinates"]]
        elif geom.get("type") == "MultiLineString":
            troncons = geom["coordinates"]
        else:
            continue
        tension = props.get("voltage")
        for coords in troncons:
            sommets = [c[:2] for c in coords]
            # Portées qui touchent l'emprise, regroupées en suites continues.
            suite = []
            for a, b in zip(sommets, sommets[1:]):
                if LineString([a, b]).intersects(emprise):
                    if not suite or suite[-1] != a:
                        if suite:
                            out.append((tension, props, suite))
                        suite = [a]
                    suite.append(b)
            if suite:
                out.append((tension, props, suite))
    return [{"tension": tension, "gestionnaire": props.get("gestionnaire"),
             "supports": [[round(s[0], 7), round(s[1], 7), *hauteur(s, tension)] for s in suite]}
            for tension, props, suite in out]
