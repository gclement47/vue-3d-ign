"""Construction d'une scène 3D autour d'un point, et son cache disque.

Une scène, c'est tout ce que la vue 3D affiche autour d'une coordonnée, dans un
carré de ±0,0016° (environ 356 m de côté) :

- les bâtiments BD TOPO, découpés sur l'emprise, et leurs toitures mesurées
  au LiDAR HD ;
- les houppiers et masses de sursol segmentés sur le MNH LiDAR HD à 0,5 m ;
- les réservoirs et constructions ponctuelles de la BD TOPO (citernes,
  torchères, cheminées, antennes), qui sortent du sursol avant la
  segmentation ;
- les routes, pour orienter Street View ;
- le relief RGE ALTI, quantifié au décimètre ;
- un anneau de relief grossier sur 2 km de côté, pour que la scène ne flotte
  pas dans le vide ;
- et, dans un fichier à part, une mosaïque d'orthophoto en une seule image.

Les monuments OSM (`building:part`), là où ils sont plus riches que la BD
TOPO, forment une couche à part (`Cache.obtenir_monuments`), que la vue
demande une fois la scène affichée. OpenStreetMap répond de 0,6 s à plus de
100 s et tombe parfois : la scène ne l'attend pas, et n'échoue pas avec lui.
La lecture d'Overpass part pourtant dès la demande de la scène, en tâche de
fond, pour que la couche soit souvent prête quand la vue la demande.

Les ouvrages de la BD TOPO (murs, ponts, voies ferrées, terrains de sport,
vue3d/ouvrages.py) suivent le même chemin (`Cache.obtenir_ouvrages`) : rien de
ce qu'ils décrivent n'entre dans un calcul de la scène, qui n'a donc pas à
échouer avec eux ni à être reconstruite quand leur format change.

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

import concurrent.futures
import gzip
import json
import logging
import math
import os
import tempfile
import threading
import time

from .batiments import decouper_batiments
from .constructions import (COUCHE_PONCTUELLES, COUCHE_RESERVOIRS,
                            constructions_pour_emprise)
from .couches import (COUCHE_BATIMENTS, COUCHE_FORET, COUCHE_ROUTES,
                      COUCHE_VEGETATION, lire_couche)
from .eau import COUCHE_COURS_EAU, COUCHE_SURFACES_EAU, eau_pour_emprise
from .lignes import COUCHE_LIGNES, COUCHE_PYLONES, lignes_pour_emprise
from .houppiers import houppiers_pour_emprise
from .mnh import fetch_mnh_grid, fetch_sol_grid
from .monuments import fetch_monuments, monuments_pour_emprise
from .ortho import fetch_exg_grid, fetch_ortho_jpeg
from .ouvrages import OUVRAGES_VERSION, fetch_ouvrages, ouvrages_pour_emprise
from .relief import fetch_relief, fetch_relief_anneau
from .toits import TOITS_RESOLUTION_M, toits_pour_emprise

journal = logging.getLogger(__name__)

# Format de la scène. L'incrémenter invalide tout le cache.
# 2 : anneau de relief autour de l'emprise.
# 3 : surface mesurée des toits fiables.
# 4 : eau de surface.
# 5 : lignes à haute tension.
# 6 : monuments OSM (building:part).
# 7 : profil minimal (hauteur inconnue) pour les bâtiments illisibles.
# 8 : toits en pans (vue3d/pans.py), préférés à la surface mesurée.
# 9 : monuments OSM hors de la scène, en couche à part (NOM_MONUMENTS).
# 10 : bâtiments découpés sur l'emprise (vue3d/batiments.py).
# 11 : réservoirs et constructions ponctuelles (vue3d/constructions.py),
#      retirés du sursol des houppiers.
SCENE_VERSION = 11
# Demi-côté de l'emprise, en degrés : ~178 m de part et d'autre du point.
SCENE_DELTA = 0.0016
# Demi-côté de l'anneau, en mètres et non en degrés : carré sur le terrain.
# La vue s'éloigne de 900 m au plus et son brouillard s'achève à 900 m de la
# caméra : au-delà d'un kilomètre du point, l'anneau ne serait jamais vu.
ANNEAU_DEMI_M = 1000
# Arrondi du point pour la clé de cache : 4 décimales, une dizaine de mètres.
SCENE_ARRONDI = 4
# Étapes d'une construction, que la page affiche pendant l'attente : les
# seize lectures de construire(), puis les toitures et les houppiers
# d'assembler(). Un compteur plutôt qu'un pourcentage : les étapes sont très
# inégales (moins d'une seconde pour la plupart des lectures, plusieurs pour
# la grille MNH et les deux calculs), une part du temps serait fausse.
ETAPES_SCENE = 18
NOM_SCENE = "scene.json.gz"
NOM_ORTHO = "ortho.jpg"
NOM_MONUMENTS = "monuments.json.gz"
# La version de la couche est dans son nom : la changer ne refait qu'elle.
NOM_OUVRAGES = f"ouvrages-v{OUVRAGES_VERSION}.json.gz"

# Le service couvre la France. Au-delà, les couches répondent vide et la scène
# n'aurait rien à montrer : mieux vaut le dire tout de suite.
EMPRISE_SERVIE = {"lat": (41.0, 51.6), "lon": (-5.8, 10.0)}


class SceneIncomplete(RuntimeError):
    """Une source n'a pas répondu : rien n'est mis en cache, réessayer plus tard."""


class HorsEmprise(ValueError):
    """Le point est hors de la zone couverte par les données IGN."""


class MonumentsIndisponibles(RuntimeError):
    """Overpass n'a pas répondu : la couche OSM n'est pas mise en cache, la
    scène reste affichée sans elle, réessayer plus tard."""


class OuvragesIndisponibles(RuntimeError):
    """Une couche d'ouvrages de l'IGN n'a pas répondu : rien n'est mis en
    cache, la scène reste affichée sans eux, réessayer plus tard."""


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
              avancer=None, constructions=None):
    """Contenu de la scène à partir des sources déjà obtenues.

    Séparée de `construire` pour être testable sans réseau. Les grilles MNH,
    ExG et de terrain ne sont PAS embarquées : ce sont des entrées de calcul
    de 1,9 Mo, 475 Ko et 1,4 Mo, sans usage une fois les toitures et les
    houppiers obtenus. `avancer`, s'il est donné, est appelé au début de
    chacun des deux calculs, avec son libellé. `constructions` : les couches
    BD TOPO (réservoirs, constructions ponctuelles), None sans elles.
    """
    avancer = avancer or (lambda libelle: None)
    avancer("toitures")
    # Découpés pour les toitures et pour la vue. Les houppiers gardent les
    # bâtiments entiers : leur masque bâti s'arrête de toute façon à la
    # grille, et dans le retrait de la découpe un toit passerait pour du
    # sursol.
    decoupes = decouper_batiments(batiments, west, south, east, north)
    toits = toits_pour_emprise(west, south, east, north, decoupes, grille, exg, sol)
    avancer("houppiers")
    # Réservoirs et constructions ponctuelles sortent du sursol avec les
    # bâtiments : sans cela, une citerne se couvre de masses et de houppiers
    # (vue3d/constructions.py).
    construits, masque = constructions_pour_emprise(
        west, south, east, north, *(constructions or (None, None)), batiments, grille)
    bati = {"features": (batiments or {}).get("features", []) + masque["features"]}
    veg = houppiers_pour_emprise(west, south, east, north, bati, vegetation,
                                 forets, grille, exg)
    return {
        "version": SCENE_VERSION,
        "bbox": [west, south, east, north],
        "batiments": decoupes,
        "toits": toits,
        "routes": routes,
        # Réservoirs découpés sur l'emprise et constructions ponctuelles.
        "constructions": construits,
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
    }


def construire(lat, lon, avancer=None):
    """Construit la scène d'un point.

    Args:
        avancer: appelé au début de chacune des ETAPES_SCENE étapes avec son
            libellé, pour le suivi que la page affiche.

    Returns:
        (octets gzip de la scène, octets JPEG de l'orthophoto).
    Raises:
        SceneIncomplete si une source n'a pas pu être lue.
    """
    avancer = avancer or (lambda libelle: None)
    west, south, east, north = emprise(lat, lon)

    def lire(nom, fn, *args, **kwargs):
        avancer(nom)
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
    constructions = (
        lire("réservoirs", lire_couche, COUCHE_RESERVOIRS, west, south, east, north),
        lire("constructions ponctuelles", lire_couche, COUCHE_PONCTUELLES,
             west, south, east, north))

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
    mosaique, _, _ = lire("mosaïque d'orthophoto", fetch_ortho_jpeg, west, south, east, north)

    scene = assembler(west, south, east, north, batiments, vegetation, forets,
                      routes, grille, exg, relief, anneau, sol, eau, lignes, avancer,
                      constructions)
    journal.info("Scène %.4f, %.4f : %d bâtiment(s), %d houppier(s), source %s",
                 lat, lon, len(scene["batiments"].get("features", [])),
                 len(scene["houppiers"]), grille.get("source"))
    return gzip.compress(json.dumps(scene, separators=(",", ":")).encode(), 6), mosaique


class Cache:
    """Scènes sur disque, une par point arrondi.

    Un verrou par point évite que deux demandes simultanées construisent deux
    fois la même scène — trente secondes et quelques mégaoctets d'appels IGN
    pour rien. L'écriture passe par un fichier temporaire renommé : une scène
    lue est toujours une scène entière.

    Les couches à part — monuments OSM, ouvrages BD TOPO — se rangent à côté
    de la scène, sous la même règle : écrites entières, ou pas du tout.
    """

    def __init__(self, dossier, lire_monuments=fetch_monuments, lire_ouvrages=fetch_ouvrages):
        self.dossier = dossier
        os.makedirs(dossier, exist_ok=True)
        self._verrous = {}
        self._verrou_global = threading.Lock()
        # Constructions en cours, par point : en mémoire, comme les verrous —
        # le service tourne en un seul processus (Dockerfile).
        self._avancements = {}
        # Lectures lancées en tâche de fond, par couche et par point (Future) ;
        # les lecteurs sont injectables pour les tests, qui n'appellent pas le
        # réseau. Un bassin de fils par source : Overpass peut tenir les siens
        # plus de 100 s, l'IGN n'a pas à attendre derrière lui.
        self.lire_monuments = lire_monuments
        self.lire_ouvrages = lire_ouvrages
        self._lectures = {}
        self._taches = {
            NOM_MONUMENTS: concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="overpass"),
            NOM_OUVRAGES: concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="ouvrages"),
        }

    def _ecrire(self, dossier, nom, octets):
        """Écrit d'un bloc : un fichier temporaire, renommé une fois complet."""
        fd, tmp = tempfile.mkstemp(dir=dossier)
        with os.fdopen(fd, "wb") as f:
            f.write(octets)
        os.replace(tmp, os.path.join(dossier, nom))

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

    def avancement(self, lat, lon):
        """Où en est la scène du point : prête, en construction (à quelle
        étape, depuis combien de secondes), ou pas encore commencée."""
        lat, lon = point_normalise(lat, lon)
        if self.present(lat, lon):
            return {"etat": "prete"}
        a = self._avancements.get((lat, lon))
        if a is None:
            return {"etat": "attente"}
        return {"etat": "construction", "etape": a["etape"], "total": ETAPES_SCENE,
                "libelle": a["libelle"], "secondes": round(time.monotonic() - a["debut"])}

    def obtenir(self, lat, lon, construire=construire):
        """Chemin de la scène du point, construite au besoin."""
        lat, lon = point_normalise(lat, lon)
        if self.present(lat, lon):
            return self._dossier_point(lat, lon)
        with self._verrou((lat, lon)):
            if self.present(lat, lon):          # construite pendant l'attente
                return self._dossier_point(lat, lon)
            debut = time.monotonic()
            etape = [0]

            def avancer(libelle):
                etape[0] += 1
                # Remplacé d'un bloc : une lecture concurrente ne voit jamais
                # l'étape d'un libellé et le libellé d'une autre.
                self._avancements[(lat, lon)] = {"etape": etape[0], "libelle": libelle,
                                                 "debut": debut}

            try:
                scene, mosaique = construire(lat, lon, avancer=avancer)
                dossier = self._dossier_point(lat, lon)
                os.makedirs(dossier, exist_ok=True)
                # L'orthophoto d'abord, la scène ensuite : `present` teste les
                # deux, une interruption entre les deux laisse une scène
                # absente, pas une scène sans image.
                for nom, octets in ((NOM_ORTHO, mosaique), (NOM_SCENE, scene)):
                    self._ecrire(dossier, nom, octets)
                return dossier
            finally:
                # Succès ou échec, la construction n'est plus en cours.
                self._avancements.pop((lat, lon), None)

    def _prelire(self, nom, lat, lon, lire):
        """Lance en tâche de fond la lecture d'une couche à part, si elle
        n'est ni en cache ni déjà demandée pour ce point."""
        lat, lon = point_normalise(lat, lon)
        if os.path.exists(self.chemin(lat, lon, nom)):
            return
        with self._verrou_global:
            if (nom, lat, lon) not in self._lectures:
                self._lectures[(nom, lat, lon)] = self._taches[nom].submit(
                    lire, *emprise(lat, lon))

    def _obtenir_couche(self, nom, lat, lon, construire, lire, indisponible, assembler):
        """Chemin du dossier où la couche `nom` du point est écrite.

        La couche lit la scène, qui est donc construite d'abord au besoin. La
        réponse de la source est celle de la tâche de fond si elle a été
        lancée, sinon elle est lue ici.

        Args:
            lire: lecture de la source sur une emprise.
            indisponible: exception levée, à partir du message, si la source
                n'a pas répondu — rien n'est alors écrit.
            assembler: (emprise, réponse de la source, scène) -> couche.
        """
        dossier = self.obtenir(lat, lon, construire=construire)
        lat, lon = point_normalise(lat, lon)
        chemin = os.path.join(dossier, nom)
        if os.path.exists(chemin):
            return dossier
        with self._verrou((nom, lat, lon)):
            if os.path.exists(chemin):          # écrite pendant l'attente
                return dossier
            with self._verrou_global:
                tache = self._lectures.pop((nom, lat, lon), None)
            try:
                brut = tache.result() if tache else lire(*emprise(lat, lon))
            except Exception as exc:
                raise indisponible(str(exc)) from exc
            with gzip.open(os.path.join(dossier, NOM_SCENE), "rt", encoding="utf-8") as f:
                scene = json.load(f)
            couche = assembler(emprise(lat, lon), brut, scene)
            self._ecrire(dossier, nom,
                         gzip.compress(json.dumps(couche, separators=(",", ":")).encode(), 6))
            # Une lecture relancée entre-temps (page rechargée) ne sert plus.
            with self._verrou_global:
                self._lectures.pop((nom, lat, lon), None)
            return dossier

    def prelire_monuments(self, lat, lon):
        """Lance la lecture d'Overpass en tâche de fond, si la couche OSM du
        point n'est ni en cache ni déjà demandée.

        Appelée à la demande de la scène : Overpass ne dépend pas de l'IGN, sa
        réponse arrive pendant la construction, et la couche est souvent prête
        quand la vue la demande.
        """
        self._prelire(NOM_MONUMENTS, lat, lon, self.lire_monuments)

    def obtenir_monuments(self, lat, lon, construire=construire):
        """Chemin du dossier où la couche OSM du point est écrite.

        La couche lit les bâtiments de la scène — les hauteurs de repli des
        parties, les bâtiments qu'elles remplacent.

        Raises:
            MonumentsIndisponibles si Overpass n'a pas répondu : rien n'est
            écrit, la demande suivante réessaie.
        """
        # Parties OSM et bâtiments BD TOPO qu'elles remplacent
        # (vue3d/monuments.py). None si l'emprise n'en a aucune — le cas de
        # presque partout, qui est un fait et se met en cache comme tel.
        return self._obtenir_couche(
            NOM_MONUMENTS, lat, lon, construire, self.lire_monuments,
            lambda message: MonumentsIndisponibles(f"monuments OSM illisibles : {message}"),
            lambda bbox, brut, scene: monuments_pour_emprise(
                *bbox, brut, scene.get("batiments") or {"features": []}))

    def prelire_ouvrages(self, lat, lon):
        """Lance la lecture des couches d'ouvrages en tâche de fond : quatre
        petites lectures WFS, finies bien avant la scène."""
        self._prelire(NOM_OUVRAGES, lat, lon, self.lire_ouvrages)

    def obtenir_ouvrages(self, lat, lon, construire=construire):
        """Chemin du dossier où la couche des ouvrages du point est écrite.

        La couche lit le relief de la scène (la hauteur d'un mur est
        l'altitude de son sommet moins le relief), ses masses de sursol
        (celles qu'un ouvrage explique) et ses routes (la largeur d'un pont).

        Raises:
            OuvragesIndisponibles si une couche de l'IGN n'a pas répondu :
            rien n'est écrit, la demande suivante réessaie.
        """
        return self._obtenir_couche(
            NOM_OUVRAGES, lat, lon, construire, self.lire_ouvrages,
            lambda message: OuvragesIndisponibles(f"ouvrages illisibles : {message}"),
            lambda bbox, brut, scene: ouvrages_pour_emprise(
                *bbox, brut, scene.get("relief"), scene.get("masses"), scene.get("routes")))
