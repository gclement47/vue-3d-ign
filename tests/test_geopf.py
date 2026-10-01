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


def test_en_parallele_rend_les_resultats_dans_l_ordre():
    """Lancées ensemble (la barrière ne s'ouvre qu'à trois), rendues dans
    l'ordre où on les a données, pas dans celui où elles finissent."""
    import threading
    import time
    barriere = threading.Barrier(3, timeout=5)

    def lecture(valeur, delai):
        barriere.wait()
        time.sleep(delai)
        return valeur

    assert geopf.en_parallele(lambda: lecture("a", 0.2), lambda: lecture("b", 0),
                              lambda: lecture("c", 0.1)) == ["a", "b", "c"]
    assert geopf.en_parallele(lambda: "seule") == ["seule"]


def test_en_parallele_n_attend_pas_les_autres_pour_echouer():
    import threading
    import time
    libere = threading.Event()

    def lente():
        libere.wait(10)
        return "trop tard"

    def en_panne():
        raise requests.Timeout("Read timed out")

    debut = time.monotonic()
    try:
        with pytest.raises(requests.Timeout):
            geopf.en_parallele(lente, en_panne)
        assert time.monotonic() - debut < 5
    finally:
        libere.set()


def test_les_requetes_simultanees_sont_bornees(monkeypatch):
    """Tout le processus partage GEOPF_SIMULTANEES places : trois fois plus
    de lectures WFS lancées ensemble n'en ont jamais davantage en cours."""
    import threading
    import time

    from vue3d import couches
    en_cours, pic, verrou = [0], [0], threading.Lock()

    class _Reponse:
        text = '{"features": []}'

        def raise_for_status(self):
            pass

    def get(url, timeout=None):
        with verrou:
            en_cours[0] += 1
            pic[0] = max(pic[0], en_cours[0])
        time.sleep(0.05)
        with verrou:
            en_cours[0] -= 1
        return _Reponse()

    monkeypatch.setattr(couches, "get_avec_reprise", get)
    geopf.en_parallele(*(lambda: couches._requete("X", 0, 0, 1, 1)
                         for _ in range(3 * geopf.GEOPF_SIMULTANEES)))
    assert pic[0] == geopf.GEOPF_SIMULTANEES


def test_une_panne_serveur_est_retentee(monkeypatch):
    appels = []
    monkeypatch.setattr(geopf.requests, "get",
                        lambda url, timeout: appels.append(url) or _Reponse(503))
    monkeypatch.setattr(geopf.time, "sleep", lambda s: None)
    with pytest.raises(requests.RequestException, match="503"):
        geopf.get_avec_reprise("http://x")
    assert len(appels) == geopf.GEOPF_ESSAIS
