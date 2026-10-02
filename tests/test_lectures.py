"""Lectures d'une scène (vue3d/scene.py, construire et _lire_ensemble) et
places vers la Géoplateforme (vue3d/geopf.py) : ce qui part, et ce qui ne
part plus, quand une lecture échoue. Le réseau n'est jamais appelé :
requests.get est remplacé par une doublure qui note chaque requête."""
import threading

import pytest
import requests

from vue3d import couches, geopf, scene
from vue3d.scene import SceneIncomplete

_GRILLE = {"couvert": True, "width": 2, "height": 2, "source": "lidar_hd",
           "bbox": [0, 0, 1, 1], "values": [0.0] * 4}


@pytest.fixture(autouse=True)
def places_neuves(monkeypatch):
    """Des places à soi, que les autres tests ne tiennent pas."""
    monkeypatch.setattr(geopf, "_places", geopf._Places(geopf.GEOPF_SIMULTANEES))


class _Reponse:
    ok = True
    status_code = 200
    text = '{"features": []}'

    def raise_for_status(self):
        pass


def _requete(nom):
    """Une requête vers la Géoplateforme, comme les vraies : une place, puis
    get_avec_reprise."""
    with geopf.place():
        return geopf.get_avec_reprise(nom)


def _sans_calcul(monkeypatch):
    """Le calcul de la scène remplacé par un assemblage vide : seules les
    lectures comptent ici."""
    monkeypatch.setattr(scene, "assembler", lambda *a, **k: {"batiments": {}, "houppiers": []})
    monkeypatch.setattr(scene, "lignes_pour_emprise", lambda *a, **k: None)


def _attendre_les_fils(prefixe, delai=5):
    for fil in threading.enumerate():
        if fil.name.startswith(prefixe):
            fil.join(delai)


def test_aucune_requete_ne_part_apres_une_scene_incomplete(monkeypatch):
    """La grille échoue alors que sept lectures tiennent les autres places et
    que huit attendent la leur : les huit ne partent jamais, ni pendant ni
    après SceneIncomplete, et la grille ne réessaie pas plus que prévu.

    Avant le groupe abandonné, chaque lecture ayant son fil, toutes
    partaient quand même : 16 requêtes sur 16, dont 7 après l'échec annoncé
    (même scénario, la grille en échec en 0,05 s et les autres en 1 s)."""
    monkeypatch.setattr(geopf.time, "sleep", lambda s: None)
    _sans_calcul(monkeypatch)
    verrou = threading.Condition()
    en_vol, parties = [0], []
    annoncee, libere, grille_partie = threading.Event(), threading.Event(), threading.Event()

    def get(url, timeout=None):
        with verrou:
            parties.append((url, annoncee.is_set()))
        if url == "grille":
            grille_partie.set()
            # Toutes les autres places prises : huit lectures attendent.
            with verrou:
                verrou.wait_for(lambda: en_vol[0] == geopf.GEOPF_SIMULTANEES - 1, 5)
            raise requests.ConnectionError("panne")
        with verrou:
            en_vol[0] += 1
            verrou.notify_all()
        libere.wait(5)
        return _Reponse()

    def apres_la_grille(lire):
        # La grille prend sa place la première : l'ordre des fils n'y est
        # pour rien.
        def lecture(*a, **k):
            grille_partie.wait(5)
            return lire(*a, **k)
        return lecture

    monkeypatch.setattr(geopf.requests, "get", get)
    monkeypatch.setattr(scene, "fetch_mnh_grid", lambda *a, **k: _requete("grille"))
    for nom in ("fetch_exg_grid", "fetch_sol_grid", "fetch_ortho_jpeg", "fetch_relief",
                "fetch_relief_anneau"):
        monkeypatch.setattr(scene, nom, apres_la_grille(lambda *a, _nom=nom, **k: _requete(_nom)))
    monkeypatch.setattr(scene, "lire_couche", apres_la_grille(couches.lire_couche))
    try:
        with pytest.raises(SceneIncomplete, match="hauteurs du sursol illisible"):
            scene.construire(48.8049, 2.1204)
        annoncee.set()
    finally:
        libere.set()
        _attendre_les_fils("lecture")
    assert [url for url, apres in parties if apres] == []
    assert [url for url, _ in parties].count("grille") == geopf.GEOPF_ESSAIS
    # Au plus une de plus que les sept en vol : entre la place que rend la
    # grille et l'abandon de son groupe, une lecture en attente peut la
    # prendre (geopf.place). Forcé par sys.setswitchinterval(1e-6), cela
    # arrivait 7 fois sur 10 ; jamais dans 300 passages ordinaires.
    assert len(parties) <= geopf.GEOPF_ESSAIS + geopf.GEOPF_SIMULTANEES


def test_l_echec_rapporte_est_celui_de_la_lecture_en_panne(monkeypatch):
    """L'orthophoto, abandonnée parce que les routes ont échoué, finit en
    même temps qu'elles et vient avant dans l'ordre des lectures : le
    message nomme les routes, pas « orthophoto illisible : lecture
    abandonnée »."""
    import concurrent.futures
    _sans_calcul(monkeypatch)
    abandonne = threading.Event()
    abandonner = geopf.Groupe.abandonner

    def abandonner_et_signaler(self, cause=None):
        abandonner(self, cause)
        abandonne.set()

    def couche(nom, *a, **k):
        if nom == scene.COUCHE_ROUTES:
            raise requests.Timeout("Read timed out")
        return {"features": []}

    def orthophoto(*a, **k):
        abandonne.wait(5)
        return _requete("orthophoto")

    monkeypatch.setattr(geopf.Groupe, "abandonner", abandonner_et_signaler)
    monkeypatch.setattr(scene, "lire_couche", couche)
    monkeypatch.setattr(scene, "fetch_exg_grid", orthophoto)
    monkeypatch.setattr(scene, "fetch_mnh_grid", lambda *a, **k: _GRILLE)
    for nom, valeur in (("fetch_sol_grid", None), ("fetch_relief", None),
                        ("fetch_relief_anneau", None), ("fetch_ortho_jpeg", (b"jpeg", None, None))):
        monkeypatch.setattr(scene, nom, lambda *a, _v=valeur, **k: _v)
    # Toutes finies quand _lire_ensemble regarde : l'ordre des fils n'y est
    # pour rien.
    attendre = concurrent.futures.wait
    monkeypatch.setattr(concurrent.futures, "wait",
                        lambda futurs, return_when=None: attendre(futurs, timeout=5))
    with pytest.raises(SceneIncomplete, match="^routes illisible : Read timed out$"):
        scene.construire(48.8049, 2.1204)


def test_le_terrain_devenu_inutile_ne_part_pas_s_il_attend_sa_place(monkeypatch):
    """La grille vient du repli MNS − MNT pendant que le terrain LiDAR, lu
    d'avance, attend encore une place : il ne part jamais (une requête
    d'environ 11 Mo de moins), et le terrain est relu à la bonne source."""
    _sans_calcul(monkeypatch)
    parties = []
    libere, bouchon_parti, terrain_attend = threading.Event(), threading.Event(), threading.Event()

    class Places(geopf._Places):
        """Une seule place, que le bouchon tient : le terrain LiDAR, second
        à la demander, l'attend."""
        demandes = 0

        def prendre(self, groupe):
            Places.demandes += 1
            if Places.demandes == 2:
                terrain_attend.set()
            return super().prendre(groupe)

    def get(url, timeout=None):
        parties.append(url)
        if url == "bouchon":
            bouchon_parti.set()
            libere.wait(5)
        return _Reponse()

    def sol(*a, **k):
        source = a[-1]
        if source == scene.SOURCE_PROBABLE:
            bouchon_parti.wait(5)
        else:
            libere.set()                        # la relecture libère le bouchon
        _requete(f"terrain {source}")
        return [source]

    def grille(*a, **k):
        terrain_attend.wait(5)
        return {**_GRILLE, "source": "mns_mnt"}

    monkeypatch.setattr(geopf, "_places", Places(1))
    monkeypatch.setattr(geopf.requests, "get", get)
    monkeypatch.setattr(scene, "fetch_mnh_grid", grille)
    monkeypatch.setattr(scene, "fetch_sol_grid", sol)
    monkeypatch.setattr(scene, "fetch_exg_grid", lambda *a, **k: _requete("bouchon"))
    for nom, valeur in (("fetch_relief", None), ("fetch_relief_anneau", None),
                        ("fetch_ortho_jpeg", (b"jpeg", None, None)),
                        ("lire_couche", {"features": []})):
        monkeypatch.setattr(scene, nom, lambda *a, _v=valeur, **k: _v)
    vus = {}
    monkeypatch.setattr(scene, "assembler", lambda *a, **k: vus.setdefault("sol", a[12]) and {
        "batiments": {}, "houppiers": []})
    try:
        scene.construire(48.8049, 2.1204)
    finally:
        libere.set()
        _attendre_les_fils("lecture")
    assert parties == ["bouchon", "terrain mns_mnt"]
    assert vus["sol"] == ["mns_mnt"]


def test_un_echec_du_terrain_lu_d_avance_n_arrete_pas_la_scene(monkeypatch):
    """Le terrain LiDAR a son propre groupe : son échec, oublié quand la
    grille a une autre source, n'abandonne pas les autres lectures."""
    _sans_calcul(monkeypatch)
    terrain_echoue = threading.Event()
    parties = []
    monkeypatch.setattr(geopf.time, "sleep", lambda s: None)

    def get(url, timeout=None):
        parties.append(url)
        if url == "terrain lidar_hd":
            raise requests.ConnectionError("panne")
        return _Reponse()

    def sol(*a, **k):
        try:
            _requete(f"terrain {a[-1]}")
        finally:
            if a[-1] == scene.SOURCE_PROBABLE:
                terrain_echoue.set()
        return [a[-1]]

    def apres_le_terrain(nom, valeur):
        def lecture(*a, **k):
            terrain_echoue.wait(5)
            _requete(nom)
            return valeur
        return lecture

    monkeypatch.setattr(geopf.requests, "get", get)
    monkeypatch.setattr(scene, "fetch_sol_grid", sol)
    monkeypatch.setattr(scene, "fetch_mnh_grid",
                        apres_le_terrain("grille", {**_GRILLE, "source": "mns_mnt"}))
    for nom, valeur in (("fetch_exg_grid", None), ("fetch_relief", None),
                        ("fetch_relief_anneau", None), ("fetch_ortho_jpeg", (b"jpeg", None, None)),
                        ("lire_couche", {"features": []})):
        monkeypatch.setattr(scene, nom, apres_le_terrain(nom, valeur))
    scene.construire(48.8049, 2.1204)
    # Toutes parties après l'échec du terrain, aucune abandonnée : ses trois
    # essais, la grille, l'orthophoto, la mosaïque, les deux reliefs, les dix
    # couches WFS et le terrain relu.
    assert len(parties) == geopf.GEOPF_ESSAIS + 1 + 4 + 10 + 1
    assert parties.count("terrain mns_mnt") == 1
