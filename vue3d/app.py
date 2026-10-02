"""Serveur de la vue 3D.

    GET /                          la page ; `?lat=…&lon=…` pour viser un point
                                   et `&zone=…` pour le côté de la zone, en mètres
    GET /api/scene?lat=…&lon=…     la scène, JSON gzippé (construite au besoin)
    GET /api/ortho?lat=…&lon=…     l'orthophoto de la scène, en JPEG
    GET /api/monuments?lat=…&lon=…   la couche des monuments OSM, JSON gzippé
    GET /api/ouvrages?lat=…&lon=…    la couche des ouvrages BD TOPO, JSON gzippé
    GET /api/piscines?lat=…&lon=…    les piscines de l'orthophoto, si le service a un détecteur
    GET /api/vehicules?lat=…&lon=…&detecteur=…   les véhicules vus d'un détecteur du service
    GET /api/panneaux?lat=…&lon=…    les panneaux solaires du registre, si le service en a un
    GET /api/avancement?lat=…&lon=…  l'étape de la construction en cours
    GET /api/sante                 contrôle de vie, pour Docker

Chaque route d'API accepte `zone=` (150 à 1 000 m, arrondie à 50 m) : sans
elle, l'emprise par défaut d'environ 356 m.

La première demande d'un point construit sa scène : une vingtaine à une
trentaine de secondes, que la page annonce. Les suivantes la lisent sur disque.
"""

import logging
import os

from flask import Flask, jsonify, request, send_from_directory

from .monuments import fetch_monuments
from .ouvrages import fetch_ouvrages
from .panneaux import REGISTRE_LICENCE
from .panneaux import lecteur as lecteur_panneaux
from .scene import (NOM_MONUMENTS, NOM_ORTHO, NOM_OUVRAGES, NOM_PANNEAUX, NOM_SCENE, Cache,
                    HorsEmprise, MonumentsIndisponibles, OuvragesIndisponibles,
                    PanneauxIndisponibles, SceneIncomplete, VehiculesDesactives,
                    VehiculesIndisponibles, zone_normalisee)
from .scene import construire as construire_scene
from .toits import autoriser_bassin
from .vehicules import MODE_PAR_DEFAUT
from .vehicules import lecteur as lecteur_vehicules

# Noms de couche pour dossier_scene : les fichiers, eux, dépendent du mode et
# du détecteur.
COUCHE_VEHICULES = "vehicules"
COUCHE_PISCINES = "piscines"

logging.basicConfig(level=os.environ.get("VUE3D_LOG", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s : %(message)s")

ICI = os.path.dirname(os.path.abspath(__file__))


def creer_app(dossier_cache=None, construire=construire_scene, lire_monuments=fetch_monuments,
              lire_ouvrages=fetch_ouvrages, lire_vehicules=None, lire_panneaux=None):
    """`construire`, `lire_monuments`, `lire_ouvrages`, `lire_vehicules` et
    `lire_panneaux` sont injectables pour les tests, qui n'appellent ni l'IGN
    ni Overpass et ne chargent aucun réseau ni registre. `lire_vehicules` :
    de `vehicules.lecteur()` ; None, le service n'a ni véhicules ni piscines.
    `lire_panneaux` : de `panneaux.lecteur()` ; None, pas de panneaux."""
    app = Flask(__name__, static_folder=os.path.join(ICI, "static"), static_url_path="/static")
    # Absolu : send_from_directory résout un chemin relatif depuis le dossier
    # de l'application, pas depuis le répertoire courant — avec
    # VUE3D_CACHE=./cache, l'orthophoto répondait 404.
    cache = Cache(os.path.abspath(dossier_cache or os.environ.get("VUE3D_CACHE", "/tmp/vue3d-cache")),
                  lire_monuments=lire_monuments, lire_ouvrages=lire_ouvrages,
                  lire_vehicules=lire_vehicules, lire_panneaux=lire_panneaux)

    def point():
        """(lat, lon, zone) de la requête ; None si l'un d'eux est illisible."""
        try:
            return (float(request.args["lat"]), float(request.args["lon"]),
                    zone_normalisee(request.args.get("zone")))
        except (KeyError, ValueError):
            return None

    MESSAGE_POINT = ("Paramètres lat et lon attendus, en degrés décimaux ; zone, "
                     "facultative, en mètres.")

    def erreur(code, message):
        return jsonify({"erreur": message}), code

    def dossier_scene(couche=None, prelire=False, detecteur=None):
        p = point()
        if p is None:
            return None, erreur(400, MESSAGE_POINT)
        *p, zone = p
        try:
            if couche == NOM_MONUMENTS:
                return cache.obtenir_monuments(*p, construire=construire, zone=zone), None
            if couche == NOM_OUVRAGES:
                return cache.obtenir_ouvrages(*p, construire=construire, zone=zone), None
            if couche == COUCHE_PISCINES:
                return cache.obtenir_piscines(*p, construire=construire, zone=zone), None
            if couche == NOM_PANNEAUX:
                return cache.obtenir_panneaux(*p, construire=construire, zone=zone), None
            if couche == COUCHE_VEHICULES:
                return cache.obtenir_vehicules(*p, detecteur, construire=construire,
                                              zone=zone), None
            if prelire:
                # Les couches à part d'abord, en tâche de fond : leurs sources
                # répondent pendant que la scène se construit, et elles sont
                # souvent prêtes quand la page les demande.
                cache.prelire_monuments(*p, zone=zone)
                cache.prelire_ouvrages(*p, zone=zone)
                cache.prelire_vehicules(*p, zone=zone)
                cache.prelire_panneaux(*p, zone=zone)
            return cache.obtenir(*p, construire=construire, zone=zone), None
        except HorsEmprise as exc:
            return None, erreur(422, str(exc))
        except SceneIncomplete as exc:
            # Rien n'a été mis en cache : la prochaine demande réessaiera.
            app.logger.warning("Scène incomplète pour %s : %s", p, exc)
            return None, erreur(503, "Un service de l'IGN n'a pas répondu ("
                                f"{exc}). Rien n'a été mis en cache : réessayez "
                                "dans quelques instants.")
        except MonumentsIndisponibles as exc:
            app.logger.warning("Monuments OSM indisponibles pour %s : %s", p, exc)
            return None, erreur(503, "OpenStreetMap n'a pas répondu ("
                                f"{exc}). Rien n'a été mis en cache : réessayez "
                                "dans quelques instants.")
        except OuvragesIndisponibles as exc:
            app.logger.warning("Ouvrages indisponibles pour %s : %s", p, exc)
            return None, erreur(503, "Un service de l'IGN n'a pas répondu ("
                                f"{exc}). Rien n'a été mis en cache : réessayez "
                                "dans quelques instants.")
        except VehiculesIndisponibles as exc:
            app.logger.warning("Détection indisponible pour %s : %s", p, exc)
            return None, erreur(503, "La détection sur l'orthophoto n'a pas abouti ("
                                f"{exc}). Rien n'a été mis en cache : réessayez "
                                "dans quelques instants.")
        except PanneauxIndisponibles as exc:
            app.logger.warning("Panneaux indisponibles pour %s : %s", p, exc)
            return None, erreur(503, f"Le registre des panneaux solaires n'a pas pu être lu ({exc}). "
                                "Rien n'a été mis en cache : réessayez dans quelques instants.")
        except VehiculesDesactives as exc:
            return None, erreur(400, f"{exc} : détecteurs de ce service : "
                                f"{', '.join(cache.lire_vehicules.detecteurs) or 'aucun'}.")

    def servir_gzip(dossier, nom, a_revalider=False):
        """`a_revalider` : le navigateur redemande à chaque fois, et le nom
        du fichier sert de validateur — la réponse est alors un 304 sans
        corps tant que le fichier est le même."""
        if a_revalider and request.if_none_match.contains(nom):
            reponse = app.response_class(status=304)
        else:
            with open(os.path.join(dossier, nom), "rb") as f:
                charge = f.read()
            # Déjà gzippé : annoncé tel quel, le navigateur le décompresse.
            reponse = app.response_class(charge, mimetype="application/json")
            reponse.headers["Content-Encoding"] = "gzip"
        if a_revalider:
            reponse.set_etag(nom)
            reponse.headers["Cache-Control"] = "no-cache"
        else:
            reponse.headers["Cache-Control"] = "public, max-age=86400"
        return reponse

    @app.get("/")
    def page():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/scene")
    def scene():
        dossier, err = dossier_scene(prelire=True)
        if err:
            return err
        return servir_gzip(dossier, NOM_SCENE)

    @app.get("/api/monuments")
    def monuments():
        """La couche OSM, demandée par la page une fois la scène affichée :
        `null` quand l'emprise n'a aucune partie, le cas de presque partout."""
        dossier, err = dossier_scene(couche=NOM_MONUMENTS)
        if err:
            return err
        return servir_gzip(dossier, NOM_MONUMENTS)

    @app.get("/api/ouvrages")
    def ouvrages():
        """Murs, ponts, voies ferrées et terrains de sport, demandés par la
        page une fois la scène affichée : `null` quand l'emprise n'en a aucun."""
        dossier, err = dossier_scene(couche=NOM_OUVRAGES)
        if err:
            return err
        return servir_gzip(dossier, NOM_OUVRAGES)

    def sans_detecteur(**vide):
        """Réponse des couches de l'orthophoto quand le service n'a pas de
        détecteur : le mode, que la page lit, et rien à dessiner. Ce n'est
        pas une erreur. Jamais gardée : le service peut être relancé avec un
        détecteur."""
        reponse = jsonify({"mode": MODE_PAR_DEFAUT, **vide})
        reponse.headers["Cache-Control"] = "no-store"
        return reponse

    # Les couches de l'orthophoto sont revalidées à chaque demande : à la
    # même adresse, elles changent avec le détecteur du service et avec leur
    # version. Gardée un jour par le navigateur, la couche montrait encore
    # les véhicules sans les piscines après une reconstruction de l'image.
    # Le nom du fichier porte le détecteur et la version, et le fichier ne
    # change jamais une fois écrit.

    @app.get("/api/piscines")
    def piscines():
        """Les piscines de l'orthophoto, demandées par la page une fois la
        scène affichée : une demi-seconde de détection, la première couche
        prête."""
        if not cache.lire_vehicules:
            return sans_detecteur(piscines=[])
        trouve, err = dossier_scene(couche=COUCHE_PISCINES)
        if err:
            return err
        return servir_gzip(*trouve, a_revalider=True)

    @app.get("/api/vehicules")
    def vehicules():
        """Les véhicules vus d'un détecteur (`detecteur=`), demandés par la
        page dans l'ordre que /api/sante lui donne, du rapide au lent : elle
        les réunit à mesure. Sans le paramètre, ou avec un détecteur que le
        service n'a pas : 400."""
        if not cache.lire_vehicules:
            return sans_detecteur(detecteur=None, vehicules=[])
        detecteur = request.args.get("detecteur")
        if not detecteur:
            return erreur(400, "Paramètre detecteur attendu : "
                          f"{', '.join(cache.lire_vehicules.detecteurs)}.")
        trouve, err = dossier_scene(couche=COUCHE_VEHICULES, detecteur=detecteur)
        if err:
            return err
        return servir_gzip(*trouve, a_revalider=True)

    @app.get("/api/panneaux")
    def panneaux():
        """Les panneaux solaires du registre OpenPVMapper, demandés par la
        page une fois la scène affichée. Sans registre, la réponse le dit
        (`actif: false`) : ce n'est pas une erreur."""
        if not cache.lire_panneaux:
            reponse = jsonify({"actif": False, "panneaux": []})
            reponse.headers["Cache-Control"] = "no-store"
            return reponse
        dossier, err = dossier_scene(couche=NOM_PANNEAUX)
        if err:
            return err
        # Revalidée comme les couches de l'orthophoto : à la même adresse, la
        # couche apparaît quand le service est relancé avec le registre.
        return servir_gzip(dossier, NOM_PANNEAUX, a_revalider=True)

    @app.get("/api/ortho")
    def ortho():
        dossier, err = dossier_scene()
        if err:
            return err
        reponse = send_from_directory(dossier, NOM_ORTHO, mimetype="image/jpeg")
        reponse.headers["Cache-Control"] = "public, max-age=86400"
        return reponse

    @app.get("/api/avancement")
    def avancement():
        """Pour la page qui attend sa scène : jamais mis en cache, il change
        d'une seconde à l'autre."""
        p = point()
        if p is None:
            return erreur(400, MESSAGE_POINT)
        try:
            reponse = jsonify(cache.avancement(*p))
        except HorsEmprise as exc:
            return erreur(422, str(exc))
        reponse.headers["Cache-Control"] = "no-store"
        return reponse

    @app.get("/api/sante")
    def sante():
        """Contrôle de vie, et ce que le service sait détecter sur
        l'orthophoto : la page y lit les détecteurs à demander, dans
        l'ordre."""
        lecteur = cache.lire_vehicules
        reponse = jsonify({"ok": True, "vehicules": {
            "mode": lecteur.mode if lecteur else MODE_PAR_DEFAUT,
            "detecteurs": list(lecteur.detecteurs) if lecteur else []},
            "panneaux": {"actif": bool(cache.lire_panneaux),
                         "source": REGISTRE_LICENCE if cache.lire_panneaux else None}})
        reponse.headers["Cache-Control"] = "no-store"
        return reponse

    return app


# Les toitures se calculent sur un bassin de processus (vue3d/toits.py), que
# le service autorise : gunicorn et flask gardent leur script principal, que
# chaque processus du bassin réexécute.
autoriser_bassin()

# Le détecteur de VUE3D_VEHICULES et le registre de VUE3D_PANNEAUX sont
# chargés ici, une fois : demandés sans leur réseau ou leur base, ils
# arrêtent le démarrage, avec la commande qui les produit.
app = creer_app(lire_vehicules=lecteur_vehicules(), lire_panneaux=lecteur_panneaux())
