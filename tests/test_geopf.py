"""Reprise sur incident des services de la Géoplateforme (vue3d/geopf.py)."""
import pytest
import requests

from vue3d import geopf


class _Reponse:
    def __init__(self, code):
        self.status_code = code
        self.ok = code < 400


def test_une_reponse_normale_passe_du_premier_coup(monkeypatch):
    appels = []
    monkeypatch.setattr(geopf.requests, "get", lambda url, timeout: appels.append(url) or _Reponse(200))
    assert geopf.get_avec_reprise("http://x").status_code == 200
    assert len(appels) == 1


def test_un_delai_d_attente_est_retente(monkeypatch):
    appels = []

    def get(url, timeout):
        appels.append(url)
        if len(appels) < 3:
            raise requests.Timeout("Read timed out")
        return _Reponse(200)

    monkeypatch.setattr(geopf.requests, "get", get)
    monkeypatch.setattr(geopf.time, "sleep", lambda s: None)
    assert geopf.get_avec_reprise("http://x").ok
    assert len(appels) == 3


def test_un_echec_persistant_finit_par_remonter(monkeypatch):
    monkeypatch.setattr(geopf.requests, "get",
                        lambda url, timeout: (_ for _ in ()).throw(requests.Timeout("nope")))
    monkeypatch.setattr(geopf.time, "sleep", lambda s: None)
    with pytest.raises(requests.RequestException):
        geopf.get_avec_reprise("http://x")


def test_un_refus_sporadique_est_retente(monkeypatch):
    """Contre-intuitif mais mesuré : une requête de relief rejetée en 400
    pendant un rattrapage, rejouée telle quelle, a répondu 200 six fois de
    suite. Ces refus-là sont passagers comme les délais d'attente."""
    appels = []

    def get(url, timeout):
        appels.append(url)
        return _Reponse(400 if len(appels) == 1 else 200)

    monkeypatch.setattr(geopf.requests, "get", get)
    monkeypatch.setattr(geopf.time, "sleep", lambda s: None)
    assert geopf.get_avec_reprise("http://x").ok
    assert len(appels) == 2


def test_un_refus_persistant_finit_par_remonter(monkeypatch):
    appels = []
    monkeypatch.setattr(geopf.requests, "get",
                        lambda url, timeout: appels.append(url) or _Reponse(400))
    monkeypatch.setattr(geopf.time, "sleep", lambda s: None)
    with pytest.raises(requests.RequestException, match="400"):
        geopf.get_avec_reprise("http://x")
    assert len(appels) == geopf.GEOPF_ESSAIS


def test_une_panne_serveur_est_retentee(monkeypatch):
    appels = []
    monkeypatch.setattr(geopf.requests, "get",
                        lambda url, timeout: appels.append(url) or _Reponse(503))
    monkeypatch.setattr(geopf.time, "sleep", lambda s: None)
    with pytest.raises(requests.RequestException, match="503"):
        geopf.get_avec_reprise("http://x")
    assert len(appels) == geopf.GEOPF_ESSAIS
