"""Construction d'une scène 3D autour d'un point, et son cache disque.

Une scène, c'est tout ce que la vue 3D affiche autour d'une coordonnée, dans un
carré de ±0,0016° (environ 356 m de côté) :

- les bâtiments BD TOPO et leurs toitures mesurées au LiDAR HD ;
- les parties de monuments OSM (`building:part`), là où elles sont plus
  riches que la BD TOPO ;
- les houppiers et masses de sursol segmentés sur le MNH LiDAR HD à 0,5 m ;
- les routes, pour orienter Street View ;
- le relief RGE ALTI, quantifié au décimètre ;
- un anneau de relief grossier sur 2 km de côté, pour que la scène ne flotte
  pas dans le vide ;
- et, dans un fichier à part, une mosaïque d'orthophoto en une seule image.

La construction coûte une vingtaine à une trentaine de secondes, dont la moitié
à télécharger la grille MNH. Le résultat est donc mis en cache sur disque, par
point arrondi à 4 décimales (une dizaine de mètres) : deux demandes voisines
partagent la même scène.

**Une scène est complète ou n'existe pas.** Le cache ne périme pas — les
campagnes LiDAR, la BD TOPO et la BD Forêt se renouvellent au mieux une fois
l'an — donc une scène mise en cache pendant une panne resterait fausse pour
toujours. La construction distingue « la donnée n'existe pas ici », qui est un
fait (hors couverture RGE ALTI, le relief est simplement absent), de « on n'a
pas réussi à la lire », qui est un incident : rien n'est alors écrit, et la
demande suivante réessaie.
"""

import gzip
import json
import logging
import math
import os
import tempfile
import threading

from .couches import (COUCHE_BATIMENTS, COUCHE_FORET, COUCHE_ROUTES,
                      COUCHE_VEGETATION, lire_couche)
from .eau import COUCHE_COURS_EAU, COUCHE_SURFACES_EAU, eau_pour_emprise
from .lignes import COUCHE_LIGNES, COUCHE_PYLONES, lignes_pour_emprise
from .houppiers import houppiers_pour_emprise
from .mnh import fetch_mnh_grid, fetch_sol_grid
from .monuments import fetch_monuments, monuments_pour_emprise
from .ortho import fetch_exg_grid, fetch_ortho_jpeg
from .relief import fetch_relief, fetch_relief_anneau
from .toits import TOITS_RESOLUTION_M, toits_pour_emprise

journal = logging.getLogger(__name__)

# Format de la scène. L'incrémenter invalide tout le cache.
# 2 : anneau de relief autour de l'emprise.
# 3 : surface mesurée des toits fiables.
# 4 : eau de surface.
# 5 : lignes à haute tension.
# 6 : monuments OSM (building:part).
SCENE_VERSION = 6
# Demi-côté de l'emprise, en degrés : ~178 m de part et d'autre du point.
SCENE_DELTA = 0.0016
# Demi-côté de l'anneau, en mètres et non en degrés : carré sur le terrain.
# La vue s'éloigne de 900 m au plus et son brouillard s'achève à 900 m de la
# caméra : au-delà d'un kilomètre du point, l'anneau ne serait jamais vu.
ANNEAU_DEMI_M = 1000
# Arrondi du point pour la clé de cache : 4 décimales, une dizaine de mètres.
SCENE_ARRONDI = 4
NOM_SCENE = "scene.json.gz"
NOM_ORTHO = "ortho.jpg"

# Le service couvre la France. Au-delà, les couches répondent vide et la scène
# n'aurait rien à montrer : mieux vaut le dire tout de suite.
EMPRISE_SERVIE = {"lat": (41.0, 51.6), "lon": (-5.8, 10.0)}


class SceneIncomplete(RuntimeError):
    """Une source n'a pas répondu : rien n'est mis en cache, réessayer plus tard."""


class HorsEmprise(ValueError):
    """Le point est hors de la zone couverte par les données IGN."""


def point_normalise(lat, lon):
    """Point arrondi qui sert de clé, après contrôle de l'emprise servie."""
    lat, lon = float(lat), float(lon)
    if not (EMPRISE_SERVIE["lat"][0] <= lat <= EMPRISE_SERVIE["lat"][1]
            and EMPRISE_SERVIE["lon"][0] <= lon <= EMPRISE_SERVIE["lon"][1]):
        raise HorsEmprise(
            "Point hors de France métropolitaine : les données IGN n'y sont pas servies.")
    return round(lat, SCENE_ARRONDI), round(lon, SCENE_ARRONDI)


def emprise(lat, lon, delta=SCENE_DELTA):
    """Emprise (ouest, sud, est, nord) autour d'un point."""
    return lon - delta, lat - delta, lon + delta, lat + delta


def emprise_anneau(lat, lon, demi_m=ANNEAU_DEMI_M):
    """Emprise (ouest, sud, est, nord) de l'anneau, carrée en mètres."""
    dlat = demi_m / 111320
    dlon = demi_m / (111320 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def assembler(west, south, east, north, batiments, vegetation, forets, routes,
              grille, exg, relief, anneau=None, sol=None, eau=None, lignes=None,
              monuments=None):
    """Contenu de la scène à partir des sources déjà obtenues.

    Séparée de `construire` pour être testable sans réseau. Les grilles MNH,
    ExG et de terrain ne sont PAS embarquées : ce sont des entrées de calcul
    de 1,9 Mo, 475 Ko et 1,4 Mo, sans usage une fois les toitures et les
    houppiers obtenus.
    """
    toits = toits_pour_emprise(west, south, east, north, batiments, grille, exg, sol)
    veg = houppiers_pour_emprise(west, south, east, north, batiments, vegetation,
                                 forets, grille, exg)
    return {
        "version": SCENE_VERSION,
        "bbox": [west, south, east, north],
        "batiments": batiments,
        "toits": toits,
        "routes": routes,
        "houppiers": veg.get("houppiers", []),
        "masses": veg.get("masses", []),
        "vegetation": {cle: veg.get(cle) for cle in (
            "source", "couvert", "resolution_m", "seuil_m", "ortho",
            "veg_disponible", "foret_disponible", "hauteur_max", "nb_ortho")},
        # None hors couverture RGE ALTI : la vue bascule sur son repli mondial.
        "relief": relief,
        # Contient l'emprise, que la vue découpe : ses trous (mer, frontière)
        # restent vides. None si rien n'y est couvert.
        "anneau": anneau,
        # Étendues et cours d'eau, découpés sur l'emprise (vue3d/eau.py).
        "eau": eau_pour_emprise(west, south, east, north, *eau) if eau else None,
        # Sur l'emprise de l'anneau : une ligne se voit de loin (vue3d/lignes.py).
        "lignes": lignes,
        # Parties de monuments OSM et bâtiments BD TOPO qu'elles remplacent
        # (vue3d/monuments.py). None si l'emprise n'en a aucune — le cas de
        # presque partout.
        "monuments": monuments_pour_emprise(west, south, east, north,
                                            monuments, batiments)
                     if monuments is not None else None,
    }


def construire(lat, lon):
    """Construit la scène d'un point.

    Returns:
        (octets gzip de la scène, octets JPEG de l'orthophoto).
    Raises:
        SceneIncomplete si une source n'a pas pu être lue.
    """
    west, south, east, north = emprise(lat, lon)

    def lire(nom, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise SceneIncomplete(f"{nom} illisible : {exc}") from exc

    batiments = lire("bâtiments", lire_couche, COUCHE_BATIMENTS, west, south, east, north)
    vegetation = lire("zones de végétation", lire_couche, COUCHE_VEGETATION,
                      west, south, east, north)
    forets = lire("BD Forêt", lire_couche, COUCHE_FORET, west, south, east, north)
    routes = lire("routes", lire_couche, COUCHE_ROUTES, west, south, east, north)
    eau = (lire("étendues d'eau", lire_couche, COUCHE_SURFACES_EAU, west, south, east, north),
           lire("cours d'eau", lire_couche, COUCHE_COURS_EAU, west, south, east, north))

    # La grille à 0,5 m sert aux toitures ET aux houppiers : lue une fois.
    grille = lire("hauteurs du sursol", fetch_mnh_grid, west, south, east, north,
                  resolution_m=TOITS_RESOLUTION_M, max_pixels=2048)
    if not grille.get("couvert"):
        # Ni LiDAR HD ni repli photogrammétrique : la scène n'aurait ni
        # toiture mesurée ni houppier. Le repli dépend d'un second service qui
        # peut lui aussi tomber : on ne fige pas une scène vide.
        raise SceneIncomplete("hauteurs du sursol indisponibles (ni LiDAR HD ni MNS − MNT)")
    exg = lire("orthophoto", fetch_exg_grid, west, south, east, north,
               grille["width"], grille["height"])
    # Le terrain dont le MNH est tiré, pour redresser les toits sur la pente :
    # une entrée de calcul, comme le MNH, jamais embarquée dans la scène.
    sol = lire("terrain sous les toits", fetch_sol_grid, west, south, east, north,
               grille["width"], grille["height"], grille["source"])
    relief = lire("relief", fetch_relief, west, south, east, north)
    anneau = lire("relief de l'anneau", fetch_relief_anneau, *emprise_anneau(lat, lon))
    lignes = lignes_pour_emprise(
        *emprise_anneau(lat, lon),
        lire("lignes électriques", lire_couche, COUCHE_LIGNES, *emprise_anneau(lat, lon)),
        lire("pylônes", lire_couche, COUCHE_PYLONES, *emprise_anneau(lat, lon)))
    # La seule source hors Géoplateforme : Overpass (OpenStreetMap). Même
    # règle que les autres — un échec lève SceneIncomplete, une emprise sans
    # partie est un fait et donne une couche nulle.
    monuments = lire("monuments OSM", fetch_monuments, west, south, east, north)
    mosaique, _, _ = lire("mosaïque d'orthophoto", fetch_ortho_jpeg, west, south, east, north)

    scene = assembler(west, south, east, north, batiments, vegetation, forets,
                      routes, grille, exg, relief, anneau, sol, eau, lignes,
                      monuments)
    journal.info("Scène %.4f, %.4f : %d bâtiment(s), %d houppier(s), source %s",
                 lat, lon, len(batiments.get("features", [])),
                 len(scene["houppiers"]), grille.get("source"))
    return gzip.compress(json.dumps(scene, separators=(",", ":")).encode(), 6), mosaique


class Cache:
    """Scènes sur disque, une par point arrondi.

    Un verrou par point évite que deux demandes simultanées construisent deux
    fois la même scène — trente secondes et quelques mégaoctets d'appels IGN
    pour rien. L'écriture passe par un fichier temporaire renommé : une scène
    lue est toujours une scène entière.
    """

    def __init__(self, dossier):
        self.dossier = dossier
        os.makedirs(dossier, exist_ok=True)
        self._verrous = {}
        self._verrou_global = threading.Lock()

    def _dossier_point(self, lat, lon):
        return os.path.join(self.dossier, f"v{SCENE_VERSION}",
                            f"{lat:.{SCENE_ARRONDI}f}_{lon:.{SCENE_ARRONDI}f}")

    def _verrou(self, cle):
        with self._verrou_global:
            return self._verrous.setdefault(cle, threading.Lock())

    def chemin(self, lat, lon, nom):
        return os.path.join(self._dossier_point(lat, lon), nom)

    def present(self, lat, lon):
        return all(os.path.exists(self.chemin(lat, lon, n)) for n in (NOM_SCENE, NOM_ORTHO))

    def obtenir(self, lat, lon, construire=construire):
        """Chemin de la scène du point, construite au besoin."""
        lat, lon = point_normalise(lat, lon)
        if self.present(lat, lon):
            return self._dossier_point(lat, lon)
        with self._verrou((lat, lon)):
            if self.present(lat, lon):          # construite pendant l'attente
                return self._dossier_point(lat, lon)
            scene, mosaique = construire(lat, lon)
            dossier = self._dossier_point(lat, lon)
            os.makedirs(dossier, exist_ok=True)
            # L'orthophoto d'abord, la scène ensuite : `present` teste les deux,
            # une interruption entre les deux laisse une scène absente, pas
            # une scène sans image.
            for nom, octets in ((NOM_ORTHO, mosaique), (NOM_SCENE, scene)):
                fd, tmp = tempfile.mkstemp(dir=dossier)
                with os.fdopen(fd, "wb") as f:
                    f.write(octets)
                os.replace(tmp, os.path.join(dossier, nom))
            return dossier
