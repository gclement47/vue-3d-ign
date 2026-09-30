"""Serveur (vue3d/app.py) : paramètres, codes d'erreur, en-têtes."""
import gzip
import json

import pytest

from vue3d import app as module_app
from vue3d.scene import SceneIncomplete


@pytest.fixture
def client(tmp_path):
    scene = {"version": 1, "bbox": [0, 0, 1, 1], "houppiers": []}

    def construire(lat, lon, avancer=None):
        if lat == 45.0:
            raise SceneIncomplete("orthophoto illisible : Read timed out")
        return gzip.compress(json.dumps(scene).encode()), b"\xff\xd8jpeg"

    def lire_monuments(west, south, east, north):
        # Au sud de 46° : Overpass en panne.
        if south < 46:
            raise ConnectionError("Overpass injoignable : 504")
        return {"elements": []}

    def lire_ouvrages(west, south, east, north):
        # Au sud de 46° : l'IGN en panne sur ces couches.
        if south < 46:
            raise ConnectionError("HTTP 504 sur construction_lineaire")
        return {"lineaires": {"features": []}, "surfaciques": {"features": []},
                "voies": {"features": []}, "terrains": {"features": []}}

    appli = module_app.creer_app(str(tmp_path), construire=construire,
                                 lire_monuments=lire_monuments, lire_ouvrages=lire_ouvrages)
    return appli.test_client()


@pytest.fixture
def client_vehicules(tmp_path):
    """Un service lancé avec un détecteur : la lecture rend une boîte au
    centre d'une orthophoto à 0,2 m, sauf au sud de 46° où elle échoue."""
    scene = {"version": 1, "bbox": [0, 0, 1, 1], "batiments": {"features": []}, "eau": None}

    def construire(lat, lon, avancer=None):
        return gzip.compress(json.dumps(scene).encode()), b"\xff\xd8jpeg"

    def lire_vehicules(west, south, east, north):
        if south < 46:
            raise ConnectionError("Read timed out")
        return {"largeur": 1173, "hauteur": 1781,
                "boites": [[586.5, 890.5, 22, 10, 0.0, 0.6, 0, 0xC81E28, "rtmdet"]]}

    lire_vehicules.mode = "rtmdet"
    appli = module_app.creer_app(str(tmp_path), construire=construire,
                                 lire_monuments=lambda *b: {"elements": []},
                                 lire_vehicules=lire_vehicules)
    return appli.test_client()


def test_la_page_est_servie(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Vue 3D IGN" in r.data


def test_la_scene_est_servie_gzippee(client):
    r = client.get("/api/scene?lat=48.8049&lon=2.1204")
    assert r.status_code == 200
    assert r.headers["Content-Encoding"] == "gzip"
    assert json.loads(gzip.decompress(r.data))["version"] == 1


def test_l_orthophoto_est_servie(client):
    r = client.get("/api/ortho?lat=48.8049&lon=2.1204")
    assert r.status_code == 200 and r.mimetype == "image/jpeg"


def test_sans_parametres_400(client):
    assert client.get("/api/scene").status_code == 400
    assert client.get("/api/scene?lat=abc&lon=2").status_code == 400


def test_hors_emprise_422(client):
    r = client.get("/api/scene?lat=40&lon=2")
    assert r.status_code == 422 and "France" in r.get_json()["erreur"]


def test_une_panne_ign_rend_503_et_ne_cache_rien(client):
    r = client.get("/api/scene?lat=45.0&lon=2")
    assert r.status_code == 503
    assert "réessayez" in r.get_json()["erreur"]


def test_l_avancement_dit_si_la_scene_est_prete(client):
    assert client.get("/api/avancement?lat=48.8049&lon=2.1204").get_json() == {"etat": "attente"}
    client.get("/api/scene?lat=48.8049&lon=2.1204")
    r = client.get("/api/avancement?lat=48.8049&lon=2.1204")
    assert r.get_json() == {"etat": "prete"}
    # Il change d'une seconde à l'autre : jamais mis en cache.
    assert r.headers["Cache-Control"] == "no-store"
    assert client.get("/api/avancement").status_code == 400
    assert client.get("/api/avancement?lat=40&lon=2").status_code == 422


def test_la_couche_osm_est_servie_a_part(client):
    """Aucune partie sur l'emprise : la couche vaut null, et se met en cache."""
    r = client.get("/api/monuments?lat=48.8049&lon=2.1204")
    assert r.status_code == 200 and r.headers["Content-Encoding"] == "gzip"
    assert json.loads(gzip.decompress(r.data)) is None


def test_une_panne_osm_rend_503_sans_toucher_la_scene(client):
    """La scène reste servie ; seule la couche OSM attend un nouvel essai."""
    assert client.get("/api/scene?lat=45.5&lon=2").status_code == 200
    r = client.get("/api/monuments?lat=45.5&lon=2")
    assert r.status_code == 503 and "OpenStreetMap" in r.get_json()["erreur"]


def test_la_couche_des_ouvrages_est_servie_a_part(client):
    """Aucun ouvrage sur l'emprise : la couche vaut null, et se met en cache."""
    r = client.get("/api/ouvrages?lat=48.8049&lon=2.1204")
    assert r.status_code == 200 and r.headers["Content-Encoding"] == "gzip"
    assert json.loads(gzip.decompress(r.data)) is None


def test_une_panne_des_ouvrages_rend_503_sans_toucher_la_scene(client):
    """La scène reste servie ; seule la couche attend un nouvel essai."""
    assert client.get("/api/scene?lat=45.5&lon=2").status_code == 200
    r = client.get("/api/ouvrages?lat=45.5&lon=2")
    assert r.status_code == 503 and "IGN" in r.get_json()["erreur"]
    assert client.get("/api/scene?lat=45.5&lon=2").status_code == 200


def test_sans_detecteur_la_couche_des_vehicules_le_dit(client):
    """Ni erreur ni fichier : la page lit le mode et ne montre pas la couche."""
    r = client.get("/api/vehicules?lat=48.8049&lon=2.1204")
    assert r.status_code == 200 and r.get_json() == {"mode": "aucun", "vehicules": []}
    # Le service peut être relancé avec un détecteur : jamais gardée.
    assert r.headers["Cache-Control"] == "no-store"


def test_la_couche_des_vehicules_est_servie_a_part(client_vehicules):
    r = client_vehicules.get("/api/vehicules?lat=48.8049&lon=2.1204")
    assert r.status_code == 200 and r.headers["Content-Encoding"] == "gzip"
    couche = json.loads(gzip.decompress(r.data))
    assert couche["mode"] == "rtmdet"
    (lon, lat, longueur, largeur, cap, couleur), = couche["vehicules"]
    assert (round(lat, 4), round(lon, 4)) == (48.8049, 2.1204) and couleur == 0xC81E28
    assert client_vehicules.get("/api/vehicules").status_code == 400
    assert client_vehicules.get("/api/vehicules?lat=40&lon=2").status_code == 422


def test_une_panne_des_vehicules_rend_503_sans_toucher_la_scene(client_vehicules):
    assert client_vehicules.get("/api/scene?lat=45.5&lon=2").status_code == 200
    r = client_vehicules.get("/api/vehicules?lat=45.5&lon=2")
    assert r.status_code == 503 and "véhicules" in r.get_json()["erreur"]
    assert client_vehicules.get("/api/scene?lat=45.5&lon=2").status_code == 200


def test_sante(client):
    assert client.get("/api/sante").get_json() == {"ok": True}
