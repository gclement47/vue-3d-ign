"""Toitures plus rapides, mêmes toits au bit près (vue3d/toits.py, vue3d/pans.py).

Chaque raccourci est comparé au calcul d'origine (tests/references_toits.py) :
la scène est mise en cache pour toujours, et une scène reconstruite doit être
celle d'avant, à l'octet. Puis le bassin de processus : mêmes toits, dans le
même ordre, que le calcul dans le processus du service, et repli sur lui
quand le bassin casse.
"""

import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
import json
import logging
import math

import numpy as np
import pytest
import shapely
from shapely import affinity
from shapely.geometry import Point, Polygon, box

from vue3d import pans, toits

from . import references_toits as ref


def _octets(a):
    return np.asarray(a).tobytes()


# --- Raccourcis, comparés au calcul d'origine -------------------------------

def _polygones():
    rect = box(-6.0, -4.0, 6.0, 4.0)
    return [rect, rect.buffer(-1.5), affinity.rotate(rect, 33, origin=(0.3, -0.2)),
            Polygon([(-7, -5), (7, -5), (7, 5), (-7, 5)], [[(-2, -2), (2, -2), (2, 2), (-2, 2)]]),
            affinity.rotate(rect, 33).buffer(-1.0), Polygon()]


def test_contains_xy_est_le_predicat_de_contains_point():
    """profil_toit.contenu teste d'un appel ce qu'il testait cellule par
    cellule, sommets et bords compris (un point du bord n'est pas dedans)."""
    rng = np.random.default_rng(1)
    gx, gy = np.meshgrid(np.arange(-8, 8.01, 0.5), np.arange(-6, 6.01, 0.5))
    for poly in _polygones():
        xs = rng.uniform(-8, 8, 400).tolist() + gx.ravel().tolist()
        ys = rng.uniform(-6, 6, 400).tolist() + gy.ravel().tolist()
        if not poly.is_empty:
            for ring in (poly.exterior, *poly.interiors):
                for (a, b), (c, d) in zip(ring.coords, ring.coords[1:]):
                    xs += [a, (a + c) / 2]
                    ys += [b, (b + d) / 2]
        x, y = np.array(xs), np.array(ys)
        attendu = [poly.contains(Point(a, b)) for a, b in zip(x.tolist(), y.tolist())]
        assert shapely.contains_xy(poly, x, y).tolist() == attendu


def test_profil_toit_lit_un_tableau_comme_une_liste():
    """Les cellules d'une boîte arrivent en tableau ; le profil est le même."""
    cellules = []
    for x in np.arange(-10, 10.01, 0.5):
        for y in np.arange(-6, 6.01, 0.5):
            # Deux corps en L : deux faîtages, donc des corps de toit.
            h = 7.0 - 3.0 * abs(y) / 6 if x < 2 else 6.5 - 2.5 * abs(x - 6) / 4
            cellules.append((float(x), float(y), float(h)))
    emprise = box(-10, -6, 10, 6)
    liste = toits.profil_toit(cellules, emprise)
    tableau = toits.profil_toit(np.array(cellules), emprise)
    assert json.dumps(liste) == json.dumps(tableau)


def test_le_faitage_le_plus_proche_est_celui_de_la_boucle():
    """Mêmes rattachements, égalités comprises : à égale distance de deux
    faîtages, la cellule va au premier, comme le faisait min()."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        k = int(rng.integers(2, 6))
        echant = [[(float(a), float(b)) for a, b in rng.integers(-40, 40, (int(rng.integers(1, 30)), 2)) * 0.5]
                  for _ in range(k)]
        dedans = [(float(x), float(y), 1.0) for x, y in rng.uniform(-25, 25, (300, 2))]
        # Cellules à égale distance de deux points de faîtages différents.
        for (a, b), (c, d) in zip(echant[0], echant[1]):
            dedans.append(((a + c) / 2, (b + d) / 2, 1.0))
        assert toits._faitage_le_plus_proche(dedans, echant) == ref.faitage_le_plus_proche(dedans, echant)
    egalite = [[(0.0, 0.0)], [(2.0, 0.0)]]
    assert toits._faitage_le_plus_proche([(1.0, 0.0, 3.0)], egalite) == [0]


def test_la_mediane_3x3_est_celle_de_nanmedian():
    rng = np.random.default_rng(3)
    for _ in range(40):
        ny, nx = rng.integers(2, 30, 2)
        g = rng.normal(5, 2, (ny, nx)) * 10.0 ** rng.integers(-2, 2, (ny, nx))
        valides = rng.random((ny, nx)) < rng.uniform(0.2, 1.0)
        assert _octets(toits._mediane_3x3(g, valides)) == _octets(ref.mediane_3x3(g, valides))


def test_la_mediane_3x3_tient_sa_precondition():
    """L'égalité au bit près avec nanmedian suppose qu'aucune cellule valide
    ne vaut −0,0 ni NaN : cellules_du_toit n'en retient pas, ni ailleurs
    que dans les valides elles ne changent rien. Hors de là, seul le signe
    d'un zéro diffère."""
    rng = np.random.default_rng(13)
    pieges = np.array([-0.0, 0.0, np.nan, -1.0])
    for h, xs, ys, emprise in _maisons():
        # Hauteurs au décimètre, comme le MNH, et des pièges partout : dans
        # l'emprise comme autour.
        h = np.round(h.astype(np.float64), 1)
        tire = rng.random(h.shape) < 0.15
        h[tire] = rng.choice(pieges, int(tire.sum()))
        c = toits.cellules_du_toit(emprise, h, xs, ys, 0.1)
        fen = h[c["j0"]:c["j1"], c["i0"]:c["i1"]]
        assert tire[c["j0"]:c["j1"], c["i0"]:c["i1"]].any()
        v = fen[c["valides"]]
        assert not np.isnan(v).any() and not (v == 0).any()
        assert _octets(c["lisse"]) == _octets(ref.mediane_3x3(fen, c["valides"]))
    # Des −0,0 parmi les valides : mêmes valeurs, au signe d'un zéro près.
    g = np.array([[-0.0, -0.0, 1.0], [-0.0, 2.0, -0.0], [3.0, -0.0, -0.0]])
    valides = np.ones(g.shape, dtype=bool)
    m, attendu = toits._mediane_3x3(g, valides), ref.mediane_3x3(g, valides)
    assert np.array_equal(m, attendu) and _octets(m) != _octets(attendu)


def test_combler_rend_les_memes_bits():
    """Remplissage de proche en proche, borné aux cellules qui touchent les
    dernières remplies : mêmes moyennes, même ordre des sommes."""
    rng = np.random.default_rng(4)
    for t in range(60):
        ny, nx = rng.integers(2, 40, 2)
        g = rng.normal(8, 3, (ny, nx)) * 10.0 ** rng.integers(-3, 3, (ny, nx))
        g[rng.random((ny, nx)) < rng.uniform(0.1, 0.95)] = np.nan
        if t % 10 == 0:
            g[:] = np.nan                           # rien à propager
        assert _octets(toits._combler(g)) == _octets(ref.combler(g))


def test_proches_est_le_masque_de_la_distance():
    xs = np.arange(-10, 10, 0.5) + 0.25
    X, Y = np.meshgrid(xs, -xs)
    for poly in _polygones()[:5]:
        for rayon in (0.0, 0.3536, 1.0):
            attendu = shapely.distance(shapely.points(X.ravel(), Y.ravel()),
                                       poly).reshape(X.shape) <= rayon
            assert np.array_equal(pans.proches(poly, X, Y, rayon), attendu)


def _grille_maison(fonction, taille=60, pas=0.5):
    xs = (np.arange(taille) + 0.5) * pas - taille * pas / 2
    ys = -xs
    X, Y = np.meshgrid(xs, ys)
    return fonction(X, Y).astype(np.float32), xs, ys


def _maisons():
    """Toits synthétiques bruités : deux pans tournés, marche, croupe, plat."""
    rng = np.random.default_rng(6)

    def deux_pans(X, Y):
        u, v = X * math.cos(0.5) + Y * math.sin(0.5), -X * math.sin(0.5) + Y * math.cos(0.5)
        return np.where((np.abs(u) <= 8) & (np.abs(v) <= 5), 7.0 - 3.0 * np.abs(v) / 5, 0.0)

    def marche(X, Y):
        return np.where((np.abs(X) <= 7) & (np.abs(Y) <= 5), np.where(X < 0, 7.0, 4.0), 0.0)

    def croupe(X, Y):
        d = np.maximum(np.abs(X) / 9, np.abs(Y) / 6)
        return np.where(d <= 1, 8.0 - 4.0 * d, 0.0)

    def plat(X, Y):
        return np.where((np.abs(X) <= 6) & (np.abs(Y) <= 6), 5.0, 0.0)

    emprises = [affinity.rotate(box(-8, -5, 8, 5), math.degrees(0.5), origin=(0, 0)),
                box(-7, -5, 7, 5), box(-9, -6, 9, 6), box(-6, -6, 6, 6)]
    for fonction, emprise in zip((deux_pans, marche, croupe, plat), emprises):
        h, xs, ys = _grille_maison(fonction)
        bruit = (rng.normal(0, 0.05, h.shape) * (h > 0)).astype(np.float32)
        yield h + bruit, xs, ys, emprise


def test_cellules_du_toit_retient_les_memes_cellules():
    rng = np.random.default_rng(7)
    for h, xs, ys, emprise in _maisons():
        vert = np.where(rng.random(h.shape) < 0.1, 20, -20).astype(np.int8)
        for verdure in (None, vert):
            c = toits.cellules_du_toit(emprise, h, xs, ys, 3.0, verdure)
            fen = None if verdure is None else verdure[c["j0"]:c["j1"], c["i0"]:c["i1"]]
            attendu = ref.valides_du_toit(emprise, np.asarray(h[c["j0"]:c["j1"], c["i0"]:c["i1"]],
                                                              dtype=np.float64),
                                          c["X"], c["Y"], 3.0, fen)
            assert np.array_equal(c["valides"], attendu)


def test_segmenter_et_prolonger_rendent_les_memes_etiquettes_et_plans():
    """Croissance des régions en listes Python, lstsq épargnés aux graines
    sans voisine libre : mêmes étiquettes, mêmes plans au bit près."""
    for h, xs, ys, emprise in _maisons():
        c = toits.cellules_du_toit(emprise, h, xs, ys, 3.0)
        z = toits._combler(np.where(c["valides"], c["lisse"], np.nan))
        X, Y, valides = c["X"], c["Y"], c["valides"]
        lab, plans = pans.segmenter(z, valides, X, Y)
        lab_ref, plans_ref = ref.segmenter(z, valides, X, Y)
        assert _octets(lab) == _octets(lab_ref) and lab.dtype == lab_ref.dtype
        assert len(plans) == len(plans_ref)
        for p, q in zip(plans, plans_ref):
            assert (p is None) == (q is None)
            if p is not None:
                assert _octets(p[0]) == _octets(q[0]) and _octets(p[1]) == _octets(q[1])
        zone = pans.proches(emprise, X, Y, pans.PANS_DEBORD_M) | (lab >= 0)
        assert _octets(pans._prolonger(lab, plans, z, zone, X, Y)) == \
            _octets(ref.prolonger(lab, plans, z, zone, X, Y))


# --- Fenêtres et bassin de processus ----------------------------------------

def test_une_fenetre_se_lit_avec_les_index_de_la_grille():
    A = np.arange(20 * 30, dtype=np.float64).reshape(20, 30)
    f = toits._Fenetre(A[5:15, 8:25], 5, 8, A.shape)
    assert np.array_equal(f[6:14, 9:20], A[6:14, 9:20])
    assert np.array_equal(f[5:15, 8:25], A[5:15, 8:25])
    assert f[7, 10] == A[7, 10]
    assert f[np.int64(7):np.int64(9), 10:12].tolist() == A[7:9, 10:12].tolist()
    # Une tranche vide l'est comme dans la grille, même hors de la fenêtre.
    assert f[30:30, 9:12].shape == A[30:30, 9:12].shape
    assert f[9:9, 0:2].size == 0
    for cle in ((slice(4, 8), slice(9, 12)), (slice(6, 8), slice(20, 26)), (4, 10), (7, 25)):
        with pytest.raises(IndexError):
            f[cle]


GEOMETRIES_ILLISIBLES = ["pas une géométrie", [[5.0, 45.0]],
                         {"type": "Polygon", "coordinates": [[[5.0]]]}]


def test_les_bornes_d_une_geometrie_illisible_sont_inconnues():
    """None plutôt qu'une erreur : le bâtiment se calcule alors dans le
    service, où shape() le refuse comme sur un cœur."""
    for geometrie in GEOMETRIES_ILLISIBLES + [
            {"type": "Polygon", "coordinates": [[[5.0, "45"]]]},
            {"type": "Polygon", "coordinates": [[[5.0, 45.0], [float("nan"), 45.1], [5.1, 45.0]]]},
            {"type": "GeometryCollection", "geometries": []}, {"type": "Polygon"}]:
        assert toits._bornes_geojson(geometrie) is None, geometrie
    carre = {"type": "Polygon", "coordinates": [[[5.0, 45.0], [5.1, 45.0], [5.1, 45.2, 30.0],
                                                 [5.0, 45.2], [5.0, 45.0]]]}
    assert toits._bornes_geojson(carre) == (5.0, 45.0, 5.1, 45.2)


def _scene_de_maisons(n_maisons=72):
    """Une scène de 160 m, assez de bâtiments pour le bassin : toits à deux
    pans tournés, marches, plats, quelques arbres, un terrain en pente."""
    rng = np.random.default_rng(8)
    lat0, lon0, pas, n = 45.0, 5.0, 0.5, 320
    m_lon = 111320 * math.cos(math.radians(lat0))
    xs = (np.arange(n) + 0.5) * pas - n * pas / 2
    X, Y = np.meshgrid(xs, -xs)
    h = np.zeros((n, n))
    vert = np.full((n, n), -20, dtype=np.int8)
    features = []
    cote = int(math.ceil(math.sqrt(n_maisons)))
    for k in range(n_maisons):
        cx, cy = (k % cote - (cote - 1) / 2) * 17.0, (k // cote - (cote - 1) / 2) * 17.0
        lx, ly, ang = rng.uniform(4, 7), rng.uniform(3, 5), rng.uniform(0, math.pi)
        u = (X - cx) * math.cos(ang) + (Y - cy) * math.sin(ang)
        v = -(X - cx) * math.sin(ang) + (Y - cy) * math.cos(ang)
        dedans = (np.abs(u) <= lx) & (np.abs(v) <= ly)
        forme = k % 4
        if forme == 0:
            toit = 7.0 - 3.0 * np.abs(v) / ly
        elif forme == 1:
            toit = np.where(u < 0, 7.5, 4.0)
        elif forme == 2:
            toit = np.full(X.shape, 5.0)
        else:
            toit = 6.0 - 2.0 * np.abs(u) / lx + np.where(v > 0, 1.5, 0.0)
        h[dedans] = toit[dedans] + rng.normal(0, 0.05, int(dedans.sum()))
        if k % 9 == 4:
            vert[dedans & (u > 0)] = 20
        coins = [(cx + a * math.cos(ang) - b * math.sin(ang), cy + a * math.sin(ang) + b * math.cos(ang))
                 for a, b in ((-lx, -ly), (lx, -ly), (lx, ly), (-lx, ly), (-lx, -ly))]
        features.append({"type": "Feature", "properties": {"cleabs": f"MAISON{k:03d}"},
                         "geometry": {"type": "Polygon", "coordinates": [[
                             (lon0 + x / m_lon, lat0 + y / 111320) for x, y in coins]]}})
    # Un bâtiment sans identifiant et un autre sans géométrie : sans profil.
    features.insert(5, {"type": "Feature", "properties": {}, "geometry": features[0]["geometry"]})
    features.insert(9, {"type": "Feature", "properties": {"cleabs": "SANS"}, "geometry": None})
    # Des géométries que shape() refuse, ni bornées ni envoyées au bassin :
    # calculées ici, sans profil, comme sur un cœur.
    for k, geometrie in enumerate(GEOMETRIES_ILLISIBLES):
        features.insert(12 + 3 * k, {"type": "Feature", "properties": {"cleabs": f"ILLISIBLE{k}"},
                                     "geometry": geometrie})
    demi = n * pas / 2
    bbox = (lon0 - demi / m_lon, lat0 - demi / 111320, lon0 + demi / m_lon, lat0 + demi / 111320)
    grille = {"bbox": list(bbox), "width": n, "height": n, "couvert": True,
              "source": "lidar_hd", "values": h.ravel().tolist()}
    sol = (200 + 0.05 * X + 0.02 * Y + 0.3 * np.sin(X / 7)).ravel().tolist()
    return bbox, {"features": features}, grille, vert, sol


@pytest.fixture
def bassin_de_deux(monkeypatch):
    """Un bassin de deux processus, arrêté après le test."""
    monkeypatch.setenv("VUE3D_TOITS_PROCESSUS", "2")
    monkeypatch.setattr(toits, "_bassin_casse", 0)
    yield
    with toits._bassin_verrou:
        if toits._bassin is not None:
            toits._bassin.shutdown(wait=True, cancel_futures=True)
        toits._bassin = None


def test_le_bassin_rend_les_memes_toits_dans_le_meme_ordre(bassin_de_deux):
    bbox, bats, grille, vert, sol = _scene_de_maisons()
    assert len(bats["features"]) >= toits.TOITS_PARALLELE_MIN
    ici = toits.toits_pour_emprise(*bbox, bats, grille, vert, sol, processus=1)
    assert toits._bassin is None
    loin = toits.toits_pour_emprise(*bbox, bats, grille, vert, sol, processus=2)
    assert toits._bassin is not None, "le bassin n'a pas servi"
    assert json.dumps(loin) == json.dumps(ici)
    # La scène a de tout : des pans, des surfaces, des profils minimaux.
    profils = ici["toits"].values()
    assert sum("pans" in p for p in profils) >= 5 and sum("surface" in p for p in profils) >= 1
    assert "SANS" not in ici["toits"] and not any(c.startswith("ILLISIBLE") for c in ici["toits"])
    assert len(ici["toits"]) == len(bats["features"]) - 2 - len(GEOMETRIES_ILLISIBLES)
    # Sans orthophoto ni terrain aussi.
    assert json.dumps(toits.toits_pour_emprise(*bbox, bats, grille, None, None, processus=2)) == \
        json.dumps(toits.toits_pour_emprise(*bbox, bats, grille, None, None, processus=1))


class _BassinCasse:
    """Un bassin dont les processus meurent, ou dont le calcul lève."""

    def __init__(self, erreur):
        self.erreur = erreur

    def submit(self, *args, **kwargs):
        f = concurrent.futures.Future()
        f.set_exception(self.erreur)
        return f


@pytest.mark.parametrize("erreur", [BrokenProcessPool("tué"),
                                    MemoryError("plus de place")])
def test_un_bassin_qui_echoue_laisse_le_calcul_au_service(monkeypatch, caplog, erreur):
    """Même résultat, calculé ici, et un avertissement dans le journal."""
    bbox, bats, grille, vert, sol = _scene_de_maisons()
    attendu = json.dumps(toits.toits_pour_emprise(*bbox, bats, grille, vert, sol, processus=1))
    monkeypatch.setattr(toits, "_bassin_de_calcul", lambda: _BassinCasse(erreur))
    with caplog.at_level(logging.WARNING, logger="vue3d.toits"):
        obtenu = toits.toits_pour_emprise(*bbox, bats, grille, vert, sol, processus=2)
    assert json.dumps(obtenu) == attendu
    assert any("processus du service" in r.getMessage() or "repris ici" in r.getMessage()
               for r in caplog.records)


def test_sans_autorisation_le_bassin_ne_sert_pas(monkeypatch):
    """Un script qui importe vue3d (rejeu, mesure) ne lance pas de processus :
    ils réexécuteraient son script principal."""
    monkeypatch.setattr(toits, "_bassin_autorise", False)
    monkeypatch.setattr(toits, "_bassin_de_calcul", lambda: pytest.fail("bassin lancé"))
    bbox, bats, grille, vert, sol = _scene_de_maisons()
    assert toits.toits_pour_emprise(*bbox, bats, grille, vert, sol)["toits"]


_SERVICE_TUE = '''
import os, sys, time
sys.path.insert(0, sys.argv[1])
from vue3d import toits

if __name__ == "__main__":
    os.environ["VUE3D_TOITS_PROCESSUS"] = "2"
    bassin = toits._bassin_de_calcul()
    taches = [bassin.submit(abs, k) for k in range(8)]
    if sys.argv[2] == "pret":
        [t.result() for t in taches]
    # « tot » : les processus démarrent encore quand le service meurt.
    print(" ".join(str(p) for p in bassin._processes), flush=True)
    time.sleep(60)
'''


@pytest.mark.parametrize("moment", ["pret", "tot"])
def test_les_processus_du_bassin_meurent_avec_le_service(tmp_path, moment):
    """Un service tué net ne laisse pas ses processus derrière lui, même
    s'ils démarraient encore."""
    import os
    import signal
    import subprocess
    import sys
    import time
    script = tmp_path / "service.py"
    script.write_text(_SERVICE_TUE)
    racine = os.path.dirname(os.path.dirname(os.path.abspath(toits.__file__)))
    service = subprocess.Popen([sys.executable, str(script), racine, moment],
                               stdout=subprocess.PIPE, text=True)
    try:
        pids = [int(p) for p in service.stdout.readline().split()]
        assert len(pids) == 2
    finally:
        service.send_signal(signal.SIGKILL)
        service.wait()
    def du_bassin(p):
        # Le numéro d'un processus fini peut resservir : on vérifie que
        # c'est bien un processus de multiprocessing en vie (un zombie n'a
        # plus de ligne de commande sous Linux, qui n'a pas toujours ps).
        if os.path.isdir("/proc"):
            try:
                with open(f"/proc/{p}/cmdline", "rb") as f:
                    return b"multiprocessing" in f.read()
            except OSError:
                return False
        r = subprocess.run(["ps", "-o", "stat=,command=", "-p", str(p)],
                           capture_output=True, text=True)
        return "multiprocessing" in r.stdout and not r.stdout.lstrip().startswith("Z")

    fin = time.monotonic() + 5 * toits.TOITS_VEILLE_S + 5
    vivants = pids
    while vivants and time.monotonic() < fin:
        time.sleep(0.2)
        vivants = [p for p in vivants if du_bassin(p)]
    for p in vivants:
        os.kill(p, signal.SIGKILL)
    assert not vivants, "processus du bassin orphelins"


def test_le_nombre_de_processus_se_regle(monkeypatch):
    monkeypatch.setenv("VUE3D_TOITS_PROCESSUS", "3")
    assert toits.processus_toits() == 3
    monkeypatch.setenv("VUE3D_TOITS_PROCESSUS", "0")
    assert toits.processus_toits() == 1
    monkeypatch.setenv("VUE3D_TOITS_PROCESSUS", "beaucoup")
    assert toits.processus_toits() >= 1
    # Sans réglage, les cœurs, plafonnés ; le réglage, lui, n'a pas de plafond.
    import os
    monkeypatch.delenv("VUE3D_TOITS_PROCESSUS")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert toits.processus_toits() == toits.TOITS_PROCESSUS_MAX == 16
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    assert toits.processus_toits() == 3
    monkeypatch.setenv("VUE3D_TOITS_PROCESSUS", "40")
    assert toits.processus_toits() == 40


def _fil_du_bassin():
    import threading
    return [f for f in threading.enumerate() if f.name == "bassin-des-toitures"]


def test_le_service_fait_naitre_le_bassin_des_son_demarrage(bassin_de_deux):
    """autoriser_bassin, au démarrage du service, lance ses processus dans
    un fil à part : la première scène les trouve prêts."""
    assert toits._bassin is None
    toits.autoriser_bassin()
    for fil in _fil_du_bassin():
        fil.join(60)
    assert toits._bassin is not None
    processus = list(toits._bassin._processes.values())
    assert len(processus) == 2 and all(p.is_alive() for p in processus)


def test_sans_processus_le_service_ne_fait_rien_naitre(monkeypatch):
    """VUE3D_TOITS_PROCESSUS=1 (celui de toute la suite, tests/conftest.py) :
    ni fil, ni bassin."""
    monkeypatch.setenv("VUE3D_TOITS_PROCESSUS", "1")
    monkeypatch.setattr(toits, "_bassin_de_calcul", lambda: pytest.fail("bassin lancé"))
    avant = len(_fil_du_bassin())
    toits.autoriser_bassin()
    assert len(_fil_du_bassin()) == avant and toits._bassin_autorise
