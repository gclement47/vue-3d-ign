"""Serveur de la vue 3D.

    GET /                          la page ; `?lat=…&lon=…` pour viser un point
    GET /api/scene?lat=…&lon=…     la scène, JSON gzippé (construite au besoin)
    GET /api/ortho?lat=…&lon=…     l'orthophoto de la scène, en JPEG
    GET /api/monuments?lat=…&lon=…   la couche des monuments OSM, JSON gzippé
    GET /api/ouvrages?lat=…&lon=…    la couche des ouvrages BD TOPO, JSON gzippé
    GET /api/avancement?lat=…&lon=…  l'étape de la construction en cours
    GET /api/sante                 contrôle de vie, pour Docker

La première demande d'un point construit sa scène : une vingtaine à une
trentaine de secondes, que la page annonce. Les suivantes la lisent sur disque.
"""

import logging
import os

from flask import Flask, jsonify, request, send_from_directory

from .monuments import fetch_monuments
from .ouvrages import fetch_ouvrages
from .scene import (NOM_MONUMENTS, NOM_ORTHO, NOM_OUVRAGES, NOM_SCENE, Cache,
                    HorsEmprise, MonumentsIndisponibles, OuvragesIndisponibles,
                    SceneIncomplete)
from .scene import construire as construire_scene

logging.basicConfig(level=os.environ.get("VUE3D_LOG", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s : %(message)s")

ICI = os.path.dirname(os.path.abspath(__file__))


def creer_app(dossier_cache=None, construire=construire_scene, lire_monuments=fetch_monuments,
              lire_ouvrages=fetch_ouvrages):
    """`construire`, `lire_monuments` et `lire_ouvrages` sont injectables pour
    les tests, qui n'appellent ni l'IGN ni Overpass."""
    app = Flask(__name__, static_folder=os.path.join(ICI, "static"), static_url_path="/static")
    # Absolu : send_from_directory résout un chemin relatif depuis le dossier
    # de l'application, pas depuis le répertoire courant — avec
    # VUE3D_CACHE=./cache, l'orthophoto répondait 404.
    cache = Cache(os.path.abspath(dossier_cache or os.environ.get("VUE3D_CACHE", "/tmp/vue3d-cache")),
                  lire_monuments=lire_monuments, lire_ouvrages=lire_ouvrages)

    def point():
        try:
            return float(request.args["lat"]), float(request.args["lon"])
        except (KeyError, ValueError):
            return None

    def erreur(code, message):
        return jsonify({"erreur": message}), code

    def dossier_scene(couche=None, prelire=False):
        p = point()
        if p is None:
            return None, erreur(400, "Paramètres lat et lon attendus, en degrés décimaux.")
        try:
            if couche == NOM_MONUMENTS:
                return cache.obtenir_monuments(*p, construire=construire), None
            if couche == NOM_OUVRAGES:
                return cache.obtenir_ouvrages(*p, construire=construire), None
            if prelire:
                # Les couches à part d'abord, en tâche de fond : leurs sources
                # répondent pendant que la scène se construit, et elles sont
                # souvent prêtes quand la page les demande.
                cache.prelire_monuments(*p)
                cache.prelire_ouvrages(*p)
            return cache.obtenir(*p, construire=construire), None
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

    def servir_gzip(dossier, nom):
        with open(os.path.join(dossier, nom), "rb") as f:
            charge = f.read()
        # Déjà gzippé : annoncé tel quel, le navigateur le décompresse.
        reponse = app.response_class(charge, mimetype="application/json")
        reponse.headers["Content-Encoding"] = "gzip"
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
            return erreur(400, "Paramètres lat et lon attendus, en degrés décimaux.")
        try:
            reponse = jsonify(cache.avancement(*p))
        except HorsEmprise as exc:
            return erreur(422, str(exc))
        reponse.headers["Cache-Control"] = "no-store"
        return reponse

    @app.get("/api/sante")
    def sante():
        return {"ok": True}

    return app


app = creer_app()
