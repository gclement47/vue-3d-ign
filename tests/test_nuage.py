"""Nuage de points LiDAR HD (vue3d/nuage.py), sans réseau."""
import base64
import io

import numpy as np
import pytest
import requests

from vue3d import nuage

# (lieu, lon, lat, x, y) : Lambert-93 de pyproj 3.8 (EPSG:4326 -> EPSG:2154).
REFERENCES = [
    ("tour Eiffel", 2.2945, 48.85826, 648237.1625, 6862256.1148),
    ("Gordes", 5.2003, 43.9116, 876728.5214, 6314973.0065),
    ("Brest", -4.486, 48.39, 146636.1395, 6836217.3647),
    ("Ajaccio", 8.7386, 41.9192, 1176667.5614, 6108273.2834),
    ("Dunkerque", 2.3768, 51.0343, 656187.4293, 7104461.7744),
    ("Strasbourg", 7.751, 48.5819, 1050224.8723, 6841837.9889),
]


@pytest.mark.parametrize("lieu,lon,lat,x,y", REFERENCES)
def test_lambert93_au_millimetre_de_pyproj(lieu, lon, lat, x, y):
    X, Y = nuage.vers_lambert93(lon, lat)
    assert abs(X - x) < 1e-3 and abs(Y - y) < 1e-3
    l, b = nuage.depuis_lambert93(x, y)
    # 1e-8 degré : un millimètre.
    assert abs(l - lon) < 1e-8 and abs(b - lat) < 1e-8


def _fichier(taille=1_000_000):
    return bytes(np.random.default_rng(1).integers(0, 256, taille, dtype=np.uint8))


def _lecteur(donnees, bloc=4096):
    lus = []

    def lire(url, debut, fin):
        lus.append((debut, fin))
        return donnees[debut:fin + 1], len(donnees)

    return nuage.LecteurPlages("http://dalle", bloc=bloc, lire=lire), lus


def test_le_lecteur_de_plages_rend_les_octets_du_fichier():
    donnees = _fichier()
    lecteur, lus = _lecteur(donnees)
    assert lecteur.taille == len(donnees)
    for debut, n in [(0, 4), (10, 100), (5000, 20000), (999_990, 50), (123_456, 1)]:
        lecteur.seek(debut)
        assert lecteur.read(n) == donnees[debut:debut + n]
    lecteur.seek(-8, io.SEEK_END)
    assert lecteur.read(100) == donnees[-8:] and lecteur.read(10) == b""


def test_le_lecteur_de_plages_lit_par_blocs():
    """Des lectures voisines tiennent dans le bloc en mémoire : une requête.
    Le premier bloc, lu à l'ouverture pour la taille, sert l'en-tête."""
    lecteur, lus = _lecteur(_fichier(), bloc=4096)
    lecteur.seek(100)
    lecteur.read(375)
    assert len(lus) == 1
    lus.clear()
    lecteur.seek(200_000)
    for _ in range(10):
        lecteur.read(100)
    assert len(lus) == 1
    lecteur.seek(500_000)
    lecteur.read(10_000)          # plus grand que le bloc : lu d'un coup
    assert len(lus) == 2 and lus[-1] == (500_000, 509_999)


class _Reponse:
    def __init__(self, status, contenu, entetes):
        self.status_code, self.content, self.headers = status, contenu, entetes


def test_une_plage_refusee_ne_lit_jamais_le_fichier_entier(monkeypatch):
    """Un 200 sans Content-Range serait la dalle entière, 105 Mo : refusé."""
    monkeypatch.setattr(nuage, "get_avec_reprise",
                        lambda url, timeout, headers: _Reponse(200, b"x" * 1000, {}))
    with pytest.raises(requests.RequestException):
        nuage.lire_plage("http://dalle", 0, 99)


def test_une_plage_incomplete_est_un_echec(monkeypatch):
    monkeypatch.setattr(nuage, "get_avec_reprise", lambda url, timeout, headers: _Reponse(
        206, b"x" * 50, {"Content-Range": "bytes 0-99/1000"}))
    with pytest.raises(requests.RequestException):
        nuage.lire_plage("http://dalle", 0, 99)


def test_une_plage_rendue_donne_ses_octets_et_la_taille(monkeypatch):
    vus = []
    monkeypatch.setattr(nuage, "get_avec_reprise", lambda url, timeout, headers: vus.append(
        headers) or _Reponse(206, b"x" * 100, {"Content-Range": "bytes 0-99/1000"}))
    assert nuage.lire_plage("http://dalle", 0, 99) == (b"x" * 100, 1000)
    assert vus == [{"Range": "bytes=0-99"}]


# --- Ouvrages ajourés ------------------------------------------------------------

def _grilles(sol, bati, cote=60):
    """Grilles de 1 m sur [0, cote]², sol et structure vus là où les masques
    (fonctions de x, y en mètres) sont vrais."""
    x0, y0 = 650_000.0, 6_860_000.0
    X, Y = np.meshgrid(np.arange(cote) + 0.5, np.arange(cote) + 0.5)
    return {"x0": x0, "y0": y0, "pas": 1.0, "sol": sol(X, Y), "bati": bati(X, Y)}


def _batiment(cle, x0, y0, x1, y1, grilles):
    """Emprise GeoJSON en degrés d'un rectangle en mètres dans la grille."""
    gx, gy = grilles["x0"], grilles["y0"]
    xs = np.array([x0, x1, x1, x0, x0]) + gx
    ys = np.array([y0, y0, y1, y1, y0]) + gy
    lon, lat = nuage.depuis_lambert93(xs, ys)
    return {"type": "Feature", "properties": {"cleabs": cle},
            "geometry": {"type": "Polygon", "coordinates": [list(map(list, zip(lon, lat)))]}}


def test_un_batiment_plein_n_est_pas_ajoure():
    """Toit vu partout, sol nulle part : le volume BD TOPO reste."""
    g = _grilles(sol=lambda X, Y: np.zeros_like(X, bool), bati=lambda X, Y: np.ones_like(X, bool))
    cles, _, _ = nuage.ajoures({"features": [_batiment("A", 10, 10, 40, 40, g)]}, g)
    assert cles == []


def test_un_treillis_qui_laisse_voir_le_sol_est_ajoure():
    """Une structure vue partout, le sol entre ses barres une cellule sur deux."""
    g = _grilles(sol=lambda X, Y: (X.astype(int) + Y.astype(int)) % 2 == 0,
                 bati=lambda X, Y: np.ones_like(X, bool))
    cles, polys, _ = nuage.ajoures({"features": [_batiment("TOUR", 10, 10, 40, 40, g)]}, g)
    assert cles == ["TOUR"] and len(polys) == 1


def test_un_batiment_plus_recent_que_le_lidar_n_est_pas_ajoure():
    """Sol vu partout, aucune structure : créé après le relevé, il garde son volume."""
    g = _grilles(sol=lambda X, Y: np.ones_like(X, bool), bati=lambda X, Y: np.zeros_like(X, bool))
    cles, _, _ = nuage.ajoures({"features": [_batiment("NEUF", 10, 10, 40, 40, g)]}, g)
    assert cles == []


def test_le_sol_vu_le_long_des_murs_ne_compte_pas():
    """Le sol vu sur une bande de 1,5 m au bord de l'emprise — sa précision —
    est retiré par l'érosion."""
    g = _grilles(sol=lambda X, Y: ~((X > 11.5) & (X < 38.5) & (Y > 11.5) & (Y < 38.5)),
                 bati=lambda X, Y: np.ones_like(X, bool))
    cles, _, _ = nuage.ajoures({"features": [_batiment("A", 10, 10, 40, 40, g)]}, g)
    assert cles == []


def test_un_etage_qui_chevauche_un_ouvrage_ajoure_l_est_aussi():
    """Le deuxième étage de la tour Eiffel ne voit le sol qu'à 10 % : il est
    ajouré parce que la BD TOPO l'empile sur le premier."""
    g = _grilles(sol=lambda X, Y: ((X < 25) & ((X.astype(int) + Y.astype(int)) % 2 == 0)),
                 bati=lambda X, Y: np.ones_like(X, bool))
    batiments = {"features": [_batiment("BAS", 5, 5, 25, 55, g), _batiment("HAUT", 20, 20, 50, 50, g),
                              _batiment("VOISIN", 52, 5, 58, 55, g)]}
    cles, _, details = nuage.ajoures(batiments, g)
    assert cles == ["BAS", "HAUT"]
    # La fiche dit pourquoi : la mesure du bas, le chevauchement du haut.
    assert details["BAS"]["chevauche"] is None and details["BAS"]["sol"] == pytest.approx(0.5, abs=0.05)
    assert details["HAUT"]["chevauche"] == "BAS" and details["HAUT"]["structure"] == 1.0


def test_un_voxel_ne_garde_qu_un_point():
    x = np.array([0.1, 0.2, 0.3, 0.7, 5.0])
    y = np.zeros(5)
    z = np.array([0.1, 0.1, 0.4, 0.1, 0.1])
    assert nuage.decimer(x, y, z, 0.5).tolist() == [0, 3, 4]


# --- La couche ------------------------------------------------------------------

def _relief(west, south, east, north, altitude=30.0):
    """Relief plat à `altitude`, au format embarqué dans la scène."""
    quant = np.zeros((8, 8), dtype="<i2")
    return {"bbox": [west, south, east, north], "width": 8, "height": 8,
            "zero_m": altitude, "pas_m": 0.1, "altitudes": base64.b64encode(quant.tobytes()).decode()}


def _decoder(couche, cle, type_):
    return np.frombuffer(base64.b64decode(couche[cle]), dtype=type_)


def test_la_couche_rend_les_points_a_leur_hauteur_ajoures_d_abord():
    g = _grilles(sol=lambda X, Y: (X < 30) & ((X.astype(int) + Y.astype(int)) % 2 == 0),
                 bati=lambda X, Y: np.ones_like(X, bool))
    x = g["x0"] + np.array([15.0, 50.0, 20.0, 50.2])     # dans la tour, dehors, dans la tour, dehors
    y = g["y0"] + np.array([20.0, 20.0, 25.0, 20.1])
    z = np.array([130.0, 42.5, 31.0, 42.6])
    brut = {"x": x, "y": y, "z": z, "classe": np.array([6, 6, 64, 6], dtype=np.uint8), "grilles": g}
    lon, lat = nuage.depuis_lambert93(x, y)
    west, south, east, north = lon.min() - 0.001, lat.min() - 0.001, lon.max() + 0.001, lat.max() + 0.001
    tour = _batiment("TOUR", 5, 5, 28, 55, g)
    mlon, mlat = nuage.depuis_lambert93(g["x0"] + np.array([15.0, 50.0]), g["y0"] + np.array([30.0, 30.0]))
    masses = [{"lon": float(mlon[0]), "lat": float(mlat[0])}, {"lon": float(mlon[1]), "lat": float(mlat[1])}]
    couche = nuage.nuage_pour_emprise(west, south, east, north, brut, _relief(west, south, east, north),
                                      {"features": [tour]}, masses, houppiers=masses[::-1])
    assert couche["ajoures"] == ["TOUR"] and couche["n_ajoures"] == 2
    assert couche["ajoures_detail"]["TOUR"]["chevauche"] is None
    # Deux points dehors dans le même voxel de 0,5 m : un seul reste.
    assert couche["n"] == 3
    assert couche["masses_expliquees"] == [0] and couche["houppiers_expliques"] == [1]
    h = _decoder(couche, "h", "<u2") / 100
    assert sorted(h[:2].tolist()) == [1.0, 100.0] and h[2] == pytest.approx(12.5)
    dlon = _decoder(couche, "lon", "<i4") * 1e-7 + west
    assert abs(dlon[2] - lon[1]) < 2e-7
    assert sorted(_decoder(couche, "classe", "u1")[:2].tolist()) == [6, 64]


def test_sans_dalle_ni_relief_pas_de_couche():
    assert nuage.nuage_pour_emprise(0, 0, 1, 1, None, _relief(0, 0, 1, 1)) is None
    brut = {"x": np.zeros(0), "y": np.zeros(0), "z": np.zeros(0), "classe": np.zeros(0, np.uint8),
            "grilles": _grilles(lambda X, Y: np.zeros_like(X, bool), lambda X, Y: np.zeros_like(X, bool))}
    assert nuage.nuage_pour_emprise(0, 0, 1, 1, brut, None) is None


def test_hors_couverture_lidar_hd_la_lecture_rend_none(monkeypatch):
    monkeypatch.setattr(nuage, "lire_couche", lambda *a: {"features": []})
    assert nuage.fetch_nuage(2.29, 48.85, 2.30, 48.86) is None


def test_la_lecture_garde_le_bati_et_note_le_sol(monkeypatch):
    """Sol et structure notés sur la grille, seul le bâti transmis ; une dalle
    listée deux fois n'est lue qu'une fois."""
    url = "https://data.geopf.fr/telechargement/x/LHD_FXX_0648_6863_PTS_LAMB93_IGN69.copc.laz"
    monkeypatch.setattr(nuage, "lire_couche", lambda *a: {"features": [
        {"properties": {"url_npl": url}}, {"properties": {"url_npl": url}}]})
    x0, y0 = nuage.vers_lambert93(2.29, 48.85)
    lues = []

    def lire_points(u, bx0, by0, bx1, by1, lire=None):
        lues.append(u)
        x = np.array([bx0 + 1.2, bx0 + 3.5, bx0 + 3.6])
        y = np.array([by0 + 1.2, by0 + 2.5, by0 + 2.6])
        return (x, y, np.array([40.0, 41.0, 60.0]), np.array([2, 6, 5], dtype=np.uint8)), None

    monkeypatch.setattr(nuage, "lire_points", lire_points)
    brut = nuage.fetch_nuage(2.29, 48.85, 2.30, 48.86)
    assert lues == [url]
    assert brut["classe"].tolist() == [6]
    g = brut["grilles"]
    assert g["sol"][1, 1] and g["sol"].sum() == 1 and g["bati"][2, 3] and g["bati"].sum() == 1
