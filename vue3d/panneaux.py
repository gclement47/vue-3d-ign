"""Panneaux solaires en toiture, d'après le registre OpenPVMapper.

Une couche à part et optionnelle, comme les véhicules (vue3d/vehicules.py),
mais qui ne fait tourner aucun réseau : c'est un registre qu'on lit. Gabriel
Kasmi (Mines Paris-PSL) a passé toute la France métropolitaine à son
détecteur DeepPVMapper sur la BD ORTHO à 20 cm, et publie le résultat sur
Zenodo (10.5281/zenodo.19188878, CC-BY 4.0) : 460 755 installations en
toiture, chacune avec son polygone, sa surface, sa puissance estimée et
l'année de la photo où elle a été vue.

**Pourquoi le registre et pas le réseau.** Les poids publiés du réseau ont
été essayés (2026-09-30) : 79 % d'exactitude sur leur propre jeu de test, et
rien de trouvé sur Gordes ni Carcassonne — où le registre compte pourtant
une installation chacun. Les auteurs ont fait le travail à l'échelle du pays
avec le pipeline complet ; on lit leur résultat.

**Ce que la couche affirme.** Les installations que ce registre connaît, là
où il les met. Il est daté (photos de 2018 à 2024), il manque des
installations et en invente, comme tout détecteur, et une installation
de Carcassonne y est décalée d'un ou deux mètres par rapport à notre
orthophoto, prise une autre année. Seules les installations résidentielles
(1,7 à 36 kWc) posées sur un bâtiment y figurent : ni centrales au sol, ni
grandes toitures.

**Comment elle est servie.** Le registre pèse 211 Mo compressés, en douze
fichiers régionaux EPSG:3035. `preparer` les lit une fois, à la construction
de l'image (outils/preparer_panneaux.py), les reprojette en WGS84 et les
range dans une base SQLite avec un index spatial R-tree : une emprise de
scène s'y lit en quelques millisecondes. Sans `VUE3D_PANNEAUX`, rien n'est
lu ni chargé.
"""

import gzip
import json
import logging
import math
import os
import sqlite3
import struct

journal = logging.getLogger(__name__)

# Format de la couche. L'incrémenter ne refait que la couche, pas les scènes.
PANNEAUX_VERSION = 1
# Le registre : « individual-regions.zip » du dépôt Zenodo, douze GeoJSON
# compressés, en EPSG:3035.
REGISTRE_URL = "https://zenodo.org/records/19188878/files/individual-regions.zip?download=1"
REGISTRE_LICENCE = "OpenPVMapper (G. Kasmi), CC-BY 4.0"
# Arrondi des coordonnées : 7 décimales, un centimètre.
DECIMALES = 7
# Dans la base, un contour est une suite d'entiers de 1e-7 degré : 8 octets par
# sommet au lieu d'une trentaine en texte. 471 449 installations de 17 sommets
# en médiane : 119 Mo de base au lieu de 294.
_UNITE = 10 ** DECIMALES

# --- EPSG:3035 (LAEA Europe) vers WGS84, sur l'ellipsoïde GRS80 --------------
# D'après Snyder, Map Projections: A Working Manual (1987), p. 187-190. Sur la
# sphère, la même projection se trompe de kilomètres ; sur l'ellipsoïde, le
# code ci-dessous s'écarte de pyproj de 0,7 mm au plus sur 2 000 points tirés
# en France. Écrit ici pour ne dépendre d'aucune bibliothèque de projection :
# la lecture du registre est la seule chose du projet qui en ait besoin.
_A, _F = 6378137.0, 1 / 298.257222101
_E2 = 2 * _F - _F * _F
_E = math.sqrt(_E2)
_LAT0, _LON0, _X0, _Y0 = math.radians(52.0), math.radians(10.0), 4321000.0, 3210000.0


def _q(phi):
    s = math.sin(phi)
    return (1 - _E2) * (s / (1 - _E2 * s * s) - math.log((1 - _E * s) / (1 + _E * s)) / (2 * _E))


_QP = _q(math.pi / 2)
_BETA1 = math.asin(_q(_LAT0) / _QP)
_RQ = _A * math.sqrt(_QP / 2)
_M1 = math.cos(_LAT0) / math.sqrt(1 - _E2 * math.sin(_LAT0) ** 2)
_D = _A * _M1 / (_RQ * math.cos(_BETA1))


def laea_vers_lonlat(x, y):
    """(lon, lat) en degrés d'un point EPSG:3035 (x vers l'est, y vers le nord)."""
    x, y = x - _X0, y - _Y0
    rho = math.hypot(x / _D, _D * y)
    if rho == 0:
        return math.degrees(_LON0), math.degrees(_LAT0)
    ce = 2 * math.asin(rho / (2 * _RQ))
    beta = math.asin(math.cos(ce) * math.sin(_BETA1) + _D * y * math.sin(ce) * math.cos(_BETA1) / rho)
    lon = _LON0 + math.atan2(x * math.sin(ce),
                             _D * rho * math.cos(_BETA1) * math.cos(ce)
                             - _D * _D * y * math.sin(_BETA1) * math.sin(ce))
    # Latitude authalique -> géodésique, série de Snyder (3-18).
    phi = (beta + (_E2 / 3 + 31 * _E2 ** 2 / 180 + 517 * _E2 ** 3 / 5040) * math.sin(2 * beta)
           + (23 * _E2 ** 2 / 360 + 251 * _E2 ** 3 / 3780) * math.sin(4 * beta)
           + (761 * _E2 ** 3 / 45360) * math.sin(6 * beta))
    return math.degrees(lon), math.degrees(phi)


# --- Préparation : les GeoJSON régionaux -> une base SQLite ---------------------

def preparer(fichiers, sortie):
    """Range les installations des GeoJSON régionaux (EPSG:3035, .gz ou non)
    dans une base SQLite à index spatial, en WGS84.

    Returns:
        le nombre d'installations rangées.
    """
    if os.path.exists(sortie):
        os.remove(sortie)
    base = sqlite3.connect(sortie)
    base.executescript("""
        CREATE TABLE panneaux (id INTEGER PRIMARY KEY, surface REAL, kwp REAL, annee INTEGER,
                               contour BLOB);
        CREATE VIRTUAL TABLE emprises USING rtree(id, lon_min, lon_max, lat_min, lat_max);
    """)
    n = 0
    for fichier in fichiers:
        ouvrir = gzip.open if str(fichier).endswith(".gz") else open
        with ouvrir(fichier, "rt", encoding="utf-8") as f:
            collection = json.load(f)
        for f in collection.get("features", []):
            geom = f.get("geometry") or {}
            anneaux = geom.get("coordinates") if geom.get("type") == "Polygon" else None
            if not anneaux:
                continue
            contour = [laea_vers_lonlat(x, y) for x, y in anneaux[0]]
            entiers = [round(v * _UNITE) for pt in contour for v in pt]
            lons, lats = entiers[0::2], entiers[1::2]
            p = f.get("properties") or {}
            n += 1
            base.execute("INSERT INTO panneaux VALUES (?, ?, ?, ?, ?)",
                         (n, p.get("surface"), p.get("kWp_approx"), p.get("year"),
                          struct.pack(f"<{len(entiers)}i", *entiers)))
            base.execute("INSERT INTO emprises VALUES (?, ?, ?, ?, ?)",
                         (n, min(lons) / _UNITE, max(lons) / _UNITE, min(lats) / _UNITE, max(lats) / _UNITE))
        base.commit()
        journal.info("Panneaux : %s lu, %d installation(s) en tout", fichier, n)
    base.execute("VACUUM")
    base.close()
    return n


def _contour(blob):
    entiers = struct.unpack(f"<{len(blob) // 4}i", blob)
    return [[entiers[i] / _UNITE, entiers[i + 1] / _UNITE] for i in range(0, len(entiers), 2)]


# --- Lecture ----------------------------------------------------------------------

class PanneauxMalConfigures(RuntimeError):
    """`VUE3D_PANNEAUX` désigne une base introuvable : erreur de déploiement,
    dite au démarrage."""


class Lecteur:
    """La lecture du registre pour `scene.Cache`, sur l'emprise d'une scène."""

    def __init__(self, chemin):
        self.chemin = chemin

    def __call__(self, west, south, east, north):
        """Installations dont l'emprise touche celle de la scène.

        Returns:
            [{contour: [[lon, lat], ...], surface, kwp, annee}], le contour
            entier, non découpé.
        """
        # Une connexion par lecture : la base est en lecture seule, et les
        # lectures partent de plusieurs fils.
        base = sqlite3.connect(f"file:{self.chemin}?mode=ro", uri=True)
        try:
            lignes = base.execute(
                "SELECT p.contour, p.surface, p.kwp, p.annee FROM emprises e JOIN panneaux p ON p.id = e.id "
                "WHERE e.lon_max >= ? AND e.lon_min <= ? AND e.lat_max >= ? AND e.lat_min <= ?",
                (west, east, south, north)).fetchall()
        finally:
            base.close()
        return [{"contour": _contour(c), "surface": s, "kwp": k, "annee": a} for c, s, k, a in lignes]


def lecteur(chemin=None):
    """Le `Lecteur` de `VUE3D_PANNEAUX` (chemin de la base SQLite) ; None si la
    variable est vide, la couche n'existe pas.

    Raises:
        PanneauxMalConfigures si la base est introuvable.
    """
    chemin = (os.environ.get("VUE3D_PANNEAUX", "") if chemin is None else chemin).strip()
    if not chemin:
        return None
    if not os.path.exists(chemin):
        raise PanneauxMalConfigures(
            f"VUE3D_PANNEAUX={chemin} : base introuvable ; "
            f"python outils/preparer_panneaux.py {chemin}")
    journal.info("Panneaux solaires : registre %s", chemin)
    return Lecteur(chemin)


def panneaux_pour_emprise(west, south, east, north, brut):
    """Installations de l'emprise, découpées dessus, prêtes pour la vue.

    Returns:
        dict(version, panneaux) ; panneaux : [{contour: [[lon, lat], ...] en
        sens trigonométrique, surface (m², du registre, installation
        entière), kwp, annee}]. Une installation coupée par le cadre donne
        un morceau par partie restante.
    """
    from shapely.geometry import Polygon, box
    cadre = box(west, south, east, north)
    panneaux = []
    for p in brut or []:
        try:
            poly = Polygon(p["contour"]).buffer(0)
        except (ValueError, KeyError):
            continue
        decoupe = poly.intersection(cadre)
        for morceau in getattr(decoupe, "geoms", [decoupe]):
            if morceau.geom_type != "Polygon" or morceau.area == 0:
                continue
            contour = list(morceau.exterior.coords)
            if morceau.exterior.is_ccw is False:
                contour.reverse()
            panneaux.append({
                "contour": [[round(x, DECIMALES), round(y, DECIMALES)] for x, y in contour[:-1]],
                "surface": p.get("surface"), "kwp": p.get("kwp"), "annee": p.get("annee")})
    journal.info("Panneaux solaires : %d installation(s) sur l'emprise", len(panneaux))
    return {"version": PANNEAUX_VERSION, "panneaux": panneaux}
