"""Serveur de la vue 3D.

    GET /                          la page ; `?lat=…&lon=…` pour viser un point
    GET /api/scene?lat=…&lon=…     la scène, JSON gzippé (construite au besoin)
    GET /api/ortho?lat=…&lon=…     l'orthophoto de la scène, en JPEG
    GET /api/sante                 contrôle de vie, pour Docker

La première demande d'un point construit sa scène : une vingtaine à une
trentaine de secondes, que la page annonce. Les suivantes la lisent sur disque.
"""

import logging
import os

from flask import Flask, jsonify, request, send_from_directory

from .scene import NOM_ORTHO, NOM_SCENE, Cache, HorsEmprise, SceneIncomplete
from .scene import construire as construire_scene

logging.basicConfig(level=os.environ.get("VUE3D_LOG", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s : %(message)s")

ICI = os.path.dirname(os.path.abspath(__file__))


def creer_app(dossier_cache=None, construire=construire_scene):
    """`construire` est injectable pour les tests, qui n'appellent pas l'IGN."""
    app = Flask(__name__, static_folder=os.path.join(ICI, "static"), static_url_path="/static")
    cache = Cache(dossier_cache or os.environ.get("VUE3D_CACHE", "/tmp/vue3d-cache"))

    def point():
        try:
            return float(request.args["lat"]), float(request.args["lon"])
        except (KeyError, ValueError):
            return None

    def erreur(code, message):
        return jsonify({"erreur": message}), code

    def dossier_scene():
        p = point()
        if p is None:
            return None, erreur(400, "Paramètres lat et lon attendus, en degrés décimaux.")
        try:
            return cache.obtenir(*p, construire=construire), None
        except HorsEmprise as exc:
            return None, erreur(422, str(exc))
        except SceneIncomplete as exc:
            # Rien n'a été mis en cache : la prochaine demande réessaiera.
            app.logger.warning("Scène incomplète pour %s : %s", p, exc)
            return None, erreur(503, "Un service de l'IGN n'a pas répondu ("
                                f"{exc}). Rien n'a été mis en cache : réessayez "
                                "dans quelques instants.")

    @app.get("/")
    def page():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/scene")
    def scene():
        dossier, err = dossier_scene()
        if err:
            return err
        with open(os.path.join(dossier, NOM_SCENE), "rb") as f:
            charge = f.read()
        # Déjà gzippée : annoncée telle quelle, le navigateur la décompresse.
        reponse = app.response_class(charge, mimetype="application/json")
        reponse.headers["Content-Encoding"] = "gzip"
        reponse.headers["Cache-Control"] = "public, max-age=86400"
        return reponse

    @app.get("/api/ortho")
    def ortho():
        dossier, err = dossier_scene()
        if err:
            return err
        reponse = send_from_directory(dossier, NOM_ORTHO, mimetype="image/jpeg")
        reponse.headers["Cache-Control"] = "public, max-age=86400"
        return reponse

    @app.get("/api/sante")
    def sante():
        return {"ok": True}

    return app


app = creer_app()
