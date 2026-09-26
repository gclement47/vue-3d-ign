"""Serveur (vue3d/app.py) : paramètres, codes d'erreur, en-têtes."""
import gzip
import json

import pytest

from vue3d import app as module_app
from vue3d.scene import SceneIncomplete


@pytest.fixture
def client(tmp_path):
    scene = {"version": 1, "bbox": [0, 0, 1, 1], "houppiers": []}

    def construire(lat, lon):
        if lat == 45.0:
            raise SceneIncomplete("orthophoto illisible : Read timed out")
        return gzip.compress(json.dumps(scene).encode()), b"\xff\xd8jpeg"

    appli = module_app.creer_app(str(tmp_path), construire=construire)
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


def test_sante(client):
    assert client.get("/api/sante").get_json() == {"ok": True}
