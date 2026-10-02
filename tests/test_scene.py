"""Cache des scènes (vue3d/scene.py) : clé, complétude, verrou."""
import os
import threading

import pytest

from vue3d import scene
from vue3d.scene import Cache, HorsEmprise, SceneIncomplete, emprise, point_normalise


def test_le_point_est_arrondi_a_une_dizaine_de_metres():
    assert point_normalise(48.80491234, 2.12041234) == (48.8049, 2.1204)
    # Deux demandes voisines partagent donc la même scène.
    assert point_normalise(48.80494, 2.12036) == point_normalise(48.80486, 2.12044)


@pytest.mark.parametrize("lat,lon", [(40.0, 2.0), (48.8, 12.0), (0, 0), (-21.1, 55.5)])
def test_hors_de_france_metropolitaine_est_refuse(lat, lon):
    with pytest.raises(HorsEmprise):
        point_normalise(lat, lon)


def test_l_emprise_est_centree_sur_le_point():
    ouest, sud, est, nord = emprise(48.8, 2.1)
    assert (ouest + est) / 2 == pytest.approx(2.1) and (sud + nord) / 2 == pytest.approx(48.8)
    assert est - ouest == pytest.approx(2 * scene.SCENE_DELTA)


def test_l_anneau_est_carre_en_metres_et_contient_l_emprise():
    import math
    lat, lon = 48.8, 2.1
    ouest, sud, est, nord = scene.emprise_anneau(lat, lon)
    largeur_m = (est - ouest) * 111320 * math.cos(math.radians(lat))
    hauteur_m = (nord - sud) * 111320
    assert largeur_m == pytest.approx(hauteur_m) == pytest.approx(2 * scene.ANNEAU_DEMI_M)
    o, s, e, n = emprise(lat, lon)
    assert ouest < o and est > e and sud < s and nord > n


def test_une_scene_n_est_construite_qu_une_fois(tmp_path):
    appels = []

    def construire(lat, lon, avancer=None):
        appels.append((lat, lon))
        return b"scene", b"jpeg"

    cache = Cache(str(tmp_path))
    d1 = cache.obtenir(48.80491, 2.12041, construire=construire)
    d2 = cache.obtenir(48.80489, 2.12039, construire=construire)
    assert d1 == d2 and len(appels) == 1
    assert open(os.path.join(d1, scene.NOM_SCENE), "rb").read() == b"scene"
    assert open(os.path.join(d1, scene.NOM_ORTHO), "rb").read() == b"jpeg"


def test_le_suivi_suit_la_construction_puis_s_efface(tmp_path):
    """Pendant la construction, l'étape en cours ; après, « prête ». Une
    construction en échec ne reste pas « en cours » pour toujours."""
    cache = Cache(str(tmp_path))
    vus = []

    def construire(lat, lon, avancer):
        for libelle in ("bâtiments", "routes"):
            avancer(libelle)
            vus.append(cache.avancement(lat, lon))
        return b"s", b"j"

    assert cache.avancement(48.8049, 2.1204) == {"etat": "attente"}
    cache.obtenir(48.8049, 2.1204, construire=construire)
    assert [(v["etat"], v["etape"], v["total"], v["libelle"]) for v in vus] == [
        ("construction", 1, scene.ETAPES_SCENE, "bâtiments"),
        ("construction", 2, scene.ETAPES_SCENE, "routes")]
    assert cache.avancement(48.8049, 2.1204) == {"etat": "prete"}

    def en_panne(lat, lon, avancer):
        avancer("bâtiments")
        raise SceneIncomplete("bâtiments illisibles : Read timed out")

    with pytest.raises(SceneIncomplete):
        cache.obtenir(47.0, 2.0, construire=en_panne)
    assert cache.avancement(47.0, 2.0) == {"etat": "attente"}


def test_construire_annonce_chacune_de_ses_etapes(monkeypatch):
    """ETAPES_SCENE suit construire() : chaque lecture et chaque calcul est
    annoncé, et le compte tombe juste — sinon la page afficherait « étape 18
    sur 17 »."""
    grille = {"couvert": True, "width": 2, "height": 2, "source": "lidar_hd",
              "bbox": [0, 0, 1, 1], "values": [0.0] * 4}
    for nom, valeur in {
            "lire_couche": {"features": []}, "fetch_mnh_grid": grille,
            "fetch_exg_grid": None, "fetch_sol_grid": None, "fetch_relief": None,
            "fetch_relief_anneau": None, "lignes_pour_emprise": None,
            "fetch_ortho_jpeg": (b"jpeg", None, None),
            "toits_pour_emprise": {}, "houppiers_pour_emprise": {},
            "eau_pour_emprise": None}.items():
        monkeypatch.setattr(scene, nom, lambda *a, _v=valeur, **k: _v)
    etapes = []
    scene.construire(48.8049, 2.1204, avancer=etapes.append)
    assert len(etapes) == scene.ETAPES_SCENE
    assert etapes[0] == "bâtiments" and etapes[-2:] == ["toitures", "houppiers"]


def _scene_avec_un_batiment(lat, lon, avancer=None):
    """Scène minimale : un bâtiment BD TOPO de 20 m autour du point."""
    import gzip
    import json
    d = 0.0001
    batiment = {"type": "Feature", "properties": {"cleabs": "B1", "hauteur": 12},
                "geometry": {"type": "Polygon", "coordinates": [[
                    [lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d],
                    [lon - d, lat + d], [lon - d, lat - d]]]}}
    scene_ = {"batiments": {"type": "FeatureCollection", "features": [batiment]}}
    return gzip.compress(json.dumps(scene_).encode()), b"jpeg"


def _partie_osm(lat, lon, hauteur="30"):
    d = 0.0001
    anneau = [(lon - d, lat - d), (lon + d, lat - d), (lon + d, lat + d),
              (lon - d, lat + d), (lon - d, lat - d)]
    return {"elements": [{"type": "way", "tags": {"building:part": "yes", "height": hauteur},
                          "geometry": [{"lon": x, "lat": y} for x, y in anneau]}]}


def test_la_couche_osm_se_construit_a_part_et_se_met_en_cache(tmp_path):
    """Elle lit les bâtiments de la scène, et une seule fois Overpass."""
    import gzip
    import json
    appels = []

    def lire(*bbox):
        appels.append(bbox)
        return _partie_osm(48.8049, 2.1204)

    cache = Cache(str(tmp_path), lire_monuments=lire)
    d1 = cache.obtenir_monuments(48.8049, 2.1204, construire=_scene_avec_un_batiment)
    d2 = cache.obtenir_monuments(48.8049, 2.1204, construire=_scene_avec_un_batiment)
    assert d1 == d2 and len(appels) == 1
    couche = json.loads(gzip.decompress(open(os.path.join(d1, scene.NOM_MONUMENTS), "rb").read()))
    assert couche["remplaces"] == ["B1"] and couche["parties"][0]["h"] == 30


def test_une_panne_osm_ne_met_rien_en_cache_et_epargne_la_scene(tmp_path):
    """Overpass en panne : la scène est là, la couche non, et la demande
    suivante réessaie."""
    en_panne = [True]

    def lire(*bbox):
        if en_panne[0]:
            raise ConnectionError("Overpass injoignable : 504")
        return {"elements": []}

    cache = Cache(str(tmp_path), lire_monuments=lire)
    with pytest.raises(scene.MonumentsIndisponibles):
        cache.obtenir_monuments(48.8049, 2.1204, construire=_scene_avec_un_batiment)
    assert cache.present(48.8049, 2.1204)
    assert not os.path.exists(cache.chemin(48.8049, 2.1204, scene.NOM_MONUMENTS))
    en_panne[0] = False
    cache.obtenir_monuments(48.8049, 2.1204, construire=_scene_avec_un_batiment)
    assert os.path.exists(cache.chemin(48.8049, 2.1204, scene.NOM_MONUMENTS))


def test_overpass_repond_pendant_que_la_scene_se_construit(tmp_path):
    """La lecture anticipée tourne en même temps que la construction, et la
    couche OSM reprend sa réponse sans relire Overpass."""
    appels = []
    lu = threading.Event()

    def lire(*bbox):
        appels.append(bbox)
        lu.set()
        return {"elements": []}

    pendant = []

    def construire(lat, lon, avancer=None):
        pendant.append(lu.wait(timeout=5))
        return _scene_avec_un_batiment(lat, lon)

    cache = Cache(str(tmp_path), lire_monuments=lire)
    cache.prelire_monuments(48.8049, 2.1204)
    cache.obtenir_monuments(48.8049, 2.1204, construire=construire)
    assert pendant == [True] and len(appels) == 1


def _scene_avec_relief(lat, lon, avancer=None):
    """Scène minimale : un relief plat à 100 m, une masse de sursol au centre."""
    import gzip
    import json
    import numpy as np
    from vue3d import relief
    g = np.full((8, 8), 100.0)
    scene_ = {"relief": relief._quantifier(g, np.zeros(g.shape, dtype=bool), *emprise(lat, lon)),
              "masses": [{"lon": lon, "lat": lat, "h": 9.0}], "routes": {"features": []}}
    return gzip.compress(json.dumps(scene_).encode()), b"jpeg"


def _mur(lat, lon, z):
    d = 0.001
    mur = {"type": "Feature", "properties": {"nature": "Mur"},
           "geometry": {"type": "LineString", "coordinates": [[lon - d, lat, z], [lon + d, lat, z]]}}
    vide = {"features": []}
    return {"lineaires": {"features": [mur]}, "surfaciques": vide, "voies": vide, "terrains": vide}


def test_la_couche_des_ouvrages_lit_le_relief_et_les_masses_de_la_scene(tmp_path):
    """Un mur à 112 m sur un relief à 100 m : 12 m de haut, et la masse que
    le MNH avait posée dessus est expliquée. Une seule lecture de l'IGN."""
    import gzip
    import json
    appels = []

    def lire(*bbox):
        appels.append(bbox)
        return _mur(48.8049, 2.1204, 112.0)

    cache = Cache(str(tmp_path), lire_ouvrages=lire)
    d1 = cache.obtenir_ouvrages(48.8049, 2.1204, construire=_scene_avec_relief)
    d2 = cache.obtenir_ouvrages(48.8049, 2.1204, construire=_scene_avec_relief)
    assert d1 == d2 and len(appels) == 1
    couche = json.loads(gzip.decompress(open(os.path.join(d1, scene.NOM_OUVRAGES), "rb").read()))
    assert couche["murs"][0]["h"] == 12 and couche["masses_expliquees"] == [0]
    # La version de la couche est dans le nom du fichier, pas dans celui de la scène.
    assert f"v{scene.OUVRAGES_VERSION}" in scene.NOM_OUVRAGES


def test_une_panne_des_ouvrages_ne_met_rien_en_cache_et_epargne_la_scene(tmp_path):
    en_panne = [True]

    def lire(*bbox):
        if en_panne[0]:
            raise ConnectionError("HTTP 504 sur construction_lineaire")
        return _mur(48.8049, 2.1204, 112.0)

    cache = Cache(str(tmp_path), lire_ouvrages=lire)
    with pytest.raises(scene.OuvragesIndisponibles):
        cache.obtenir_ouvrages(48.8049, 2.1204, construire=_scene_avec_relief)
    assert cache.present(48.8049, 2.1204)
    assert not os.path.exists(cache.chemin(48.8049, 2.1204, scene.NOM_OUVRAGES))
    en_panne[0] = False
    cache.obtenir_ouvrages(48.8049, 2.1204, construire=_scene_avec_relief)
    assert os.path.exists(cache.chemin(48.8049, 2.1204, scene.NOM_OUVRAGES))


def test_les_ouvrages_sont_lus_pendant_que_la_scene_se_construit(tmp_path):
    """Lecture anticipée, comme pour Overpass, et sans attendre derrière lui :
    chaque source a ses fils."""
    lu = threading.Event()
    overpass_bloque = threading.Event()

    def lire_ouvrages(*bbox):
        lu.set()
        return _mur(48.8049, 2.1204, 112.0)

    def lire_monuments(*bbox):
        overpass_bloque.wait(5)
        return {"elements": []}

    pendant = []

    def construire(lat, lon, avancer=None):
        pendant.append(lu.wait(timeout=5))
        return _scene_avec_relief(lat, lon)

    cache = Cache(str(tmp_path), lire_monuments=lire_monuments, lire_ouvrages=lire_ouvrages)
    # Overpass tient ses deux fils : les ouvrages passent quand même.
    cache.prelire_monuments(48.8049, 2.1204)
    cache.prelire_monuments(48.9, 2.2)
    cache.prelire_ouvrages(48.8049, 2.1204)
    cache.obtenir_ouvrages(48.8049, 2.1204, construire=construire)
    overpass_bloque.set()
    assert pendant == [True]


def test_une_scene_incomplete_n_est_pas_mise_en_cache(tmp_path):
    """Le cache ne périme pas : une scène figée pendant une panne resterait
    fausse pour toujours. Rien n'est écrit, la demande suivante réessaie."""
    cache = Cache(str(tmp_path))

    def en_panne(lat, lon, avancer=None):
        raise SceneIncomplete("orthophoto illisible : Read timed out")

    with pytest.raises(SceneIncomplete):
        cache.obtenir(48.8049, 2.1204, construire=en_panne)
    assert not cache.present(48.8049, 2.1204)
    # La demande suivante, service revenu, réussit.
    cache.obtenir(48.8049, 2.1204, construire=lambda lat, lon, avancer=None: (b"s", b"j"))
    assert cache.present(48.8049, 2.1204)


def test_deux_demandes_simultanees_ne_construisent_pas_deux_fois(tmp_path):
    appels = []
    depart = threading.Event()

    def lente(lat, lon, avancer=None):
        appels.append(1)
        depart.wait(2)
        return b"s", b"j"

    cache = Cache(str(tmp_path))
    fils = [threading.Thread(target=cache.obtenir, args=(48.8049, 2.1204),
                             kwargs={"construire": lente}) for _ in range(4)]
    for f in fils:
        f.start()
    depart.set()
    for f in fils:
        f.join(5)
    assert len(appels) == 1


def test_assembler_n_embarque_pas_les_grilles():
    """La scène porte les résultats, pas les 2,4 Mo d'entrées de calcul."""
    import json
    import numpy as np
    from tests.test_houppiers import _emprise, _grille
    H = _grille(arbres=[(10, 10, 12, 4)])
    grille = _emprise(H)
    grille.update({"couvert": True, "source": "lidar_hd", "resolution_m": 0.5})
    art = scene.assembler(*grille["bbox"], {"features": []}, {"features": []}, None,
                          {"features": []}, grille, np.full(H.shape, 20, dtype=np.int8), None)
    # Les monuments OSM n'y sont pas : couche à part (Cache.obtenir_monuments).
    assert set(art) == {"version", "bbox", "batiments", "toits", "routes",
                        "constructions", "houppiers", "masses", "vegetation",
                        "relief", "anneau", "eau", "lignes"}
    assert len(art["houppiers"]) == 1
    charge = json.dumps(art)
    assert '"values"' not in charge and '"exg"' not in charge


def test_assembler_ne_plante_pas_d_arbre_dans_l_eau():
    """Entre deux quais, le MNH lit leur hauteur en pleine rivière, et l'eau
    est verte à l'orthophoto : à Notre-Dame, 438 houppiers dans la Seine."""
    import numpy as np
    from tests.test_houppiers import _emprise, _grille, _polygone_m
    H = _grille(arbres=[(20, 20, 8, 4)])
    grille = _emprise(H)
    grille.update({"couvert": True, "source": "lidar_hd", "resolution_m": 0.5})
    vert = np.full(H.shape, 20, dtype=np.int8)
    riviere = dict(_polygone_m(grille, 0, 5, 40, 35), properties={"persistance": "Permanent"})
    args = (*grille["bbox"], {"features": []}, {"features": []}, None, {"features": []},
            grille, vert, None)
    assert len(scene.assembler(*args)["houppiers"]) == 1
    art = scene.assembler(*args, eau=({"features": [riviere]}, {"features": []}))
    assert art["houppiers"] == [] and art["masses"] == []
    # L'eau, elle, est toujours dans la scène.
    assert len(art["eau"]["surfaces"]) == 1


def test_assembler_coupe_les_batiments_au_bord_de_la_scene():
    """Le WFS rend le bâtiment entier : la scène n'en garde que ce qui tient
    dans son emprise (vue3d/batiments.py), et le dit."""
    import numpy as np
    from shapely.geometry import shape
    from tests.test_houppiers import _emprise, _grille
    H = _grille()
    grille = _emprise(H)
    grille.update({"couvert": True, "source": "lidar_hd", "resolution_m": 0.5})
    ouest, sud, est, nord = grille["bbox"]
    milieu, quart = (sud + nord) / 2, (nord - sud) / 4
    deborde = {"type": "Feature", "properties": {"cleabs": "LONG"}, "geometry": {
        "type": "Polygon", "coordinates": [[
            [ouest - 1e-3, milieu - quart], [(ouest + est) / 2, milieu - quart],
            [(ouest + est) / 2, milieu + quart], [ouest - 1e-3, milieu + quart],
            [ouest - 1e-3, milieu - quart]]]}}
    art = scene.assembler(ouest, sud, est, nord, {"features": [deborde]}, {"features": []},
                          None, {"features": []}, grille, np.full(H.shape, 20, dtype=np.int8), None)
    (b,) = art["batiments"]["features"]
    assert shape(b["geometry"]).bounds[0] > ouest and 0 < b["properties"]["coupe"]["part"] < 1
    assert "LONG" in art["toits"]["toits"]


# --- Véhicules : la couche optionnelle --------------------------------------------

def _scene_nue(lat, lon, avancer=None):
    import gzip
    import json
    return gzip.compress(json.dumps({"batiments": {"features": []}, "eau": None}).encode()), b"jpeg"


def _lecteur_vehicules(mode, appels=None, en_panne=None):
    """Doublure de vehicules.Lecteur : `appels` note chaque lecture, par
    fichier ('piscines' ou le détecteur)."""
    import types
    from vue3d.vehicules import MODES

    def lecture(quoi):
        def lire(*bbox):
            if appels is not None:
                appels.append(quoi)
            if en_panne and en_panne[0]:
                raise ConnectionError("Read timed out")
            if quoi == "piscines":
                return {"largeur": 1173, "hauteur": 1781,
                        "piscines": [[300, 400, 50, 25, 0.0, 0.3, 0x5AC8D2, mode]]}
            return {"largeur": 1173, "hauteur": 1781,
                    "boites": [[586.5, 890.5, 22, 10, 0.0, 0.6, 0, 0x808080, quoi]]}
        return lire

    return types.SimpleNamespace(mode=mode, detecteurs=MODES[mode], piscines=lecture("piscines"),
                                 vehicules=lecture)


def test_sans_detecteur_les_couches_de_l_orthophoto_n_existent_pas(tmp_path):
    cache = Cache(str(tmp_path))
    cache.prelire_vehicules(48.8049, 2.1204)              # sans effet, sans erreur
    with pytest.raises(scene.VehiculesDesactives):
        cache.obtenir_vehicules(48.8049, 2.1204, "rtmdet", construire=_scene_nue)
    with pytest.raises(scene.VehiculesDesactives):
        cache.obtenir_piscines(48.8049, 2.1204, construire=_scene_nue)


def test_un_fichier_par_detecteur_et_un_pour_les_piscines(tmp_path):
    """Le nom du fichier porte le détecteur (ou le mode, pour les piscines)
    et la version : relancer le service avec un autre détecteur ne ressert
    jamais la couche du précédent, et chaque fichier n'est lu qu'une fois."""
    import gzip
    import json
    appels = []
    cache = Cache(str(tmp_path), lire_vehicules=_lecteur_vehicules("tous", appels))
    dossier, nom = cache.obtenir_vehicules(48.8049, 2.1204, "rtmdet", construire=_scene_nue)
    assert cache.obtenir_vehicules(48.8049, 2.1204, "rtmdet", construire=_scene_nue) == (dossier, nom)
    assert nom == f"vehicules-rtmdet-v{scene.VEHICULES_VERSION}.json.gz"
    couche = json.loads(gzip.decompress(open(os.path.join(dossier, nom), "rb").read()))
    assert couche["detecteur"] == "rtmdet" and len(couche["vehicules"]) == 1
    _, nom_yolo = cache.obtenir_vehicules(48.8049, 2.1204, "yolo", construire=_scene_nue)
    assert nom_yolo == f"vehicules-yolo-v{scene.VEHICULES_VERSION}.json.gz"
    _, nom_piscines = cache.obtenir_piscines(48.8049, 2.1204, construire=_scene_nue)
    assert nom_piscines == f"piscines-tous-v{scene.PISCINES_VERSION}.json.gz"
    couche = json.loads(gzip.decompress(open(os.path.join(dossier, nom_piscines), "rb").read()))
    assert couche["mode"] == "tous" and len(couche["piscines"]) == 1
    assert appels == ["rtmdet", "yolo", "piscines"]
    # Un détecteur que ce service n'a pas.
    with pytest.raises(scene.VehiculesDesactives):
        cache.obtenir_vehicules(48.8049, 2.1204, "inconnu", construire=_scene_nue)


def test_une_panne_de_detection_ne_met_rien_en_cache_et_epargne_la_scene(tmp_path):
    en_panne = [True]
    cache = Cache(str(tmp_path), lire_vehicules=_lecteur_vehicules("rtmdet", en_panne=en_panne))
    with pytest.raises(scene.VehiculesIndisponibles):
        cache.obtenir_vehicules(48.8049, 2.1204, "rtmdet", construire=_scene_nue)
    with pytest.raises(scene.VehiculesIndisponibles):
        cache.obtenir_piscines(48.8049, 2.1204, construire=_scene_nue)
    assert cache.present(48.8049, 2.1204)
    assert not os.path.exists(cache.chemin(48.8049, 2.1204, scene.nom_vehicules("rtmdet")))
    assert not os.path.exists(cache.chemin(48.8049, 2.1204, scene.nom_piscines("rtmdet")))
    en_panne[0] = False
    cache.obtenir_vehicules(48.8049, 2.1204, "rtmdet", construire=_scene_nue)
    cache.obtenir_piscines(48.8049, 2.1204, construire=_scene_nue)
    assert os.path.exists(cache.chemin(48.8049, 2.1204, scene.nom_vehicules("rtmdet")))
    assert os.path.exists(cache.chemin(48.8049, 2.1204, scene.nom_piscines("rtmdet")))


def test_les_detections_sont_lancees_pendant_que_la_scene_se_construit(tmp_path):
    """La lecture anticipée sert la demande : une seule détection par
    fichier, les piscines d'abord, puis les détecteurs du rapide au lent."""
    appels = []
    cache = Cache(str(tmp_path), lire_vehicules=_lecteur_vehicules("tous", appels))
    cache.prelire_vehicules(48.8049, 2.1204)
    cache.prelire_vehicules(48.8049, 2.1204)              # page rechargée : pas de second calcul
    cache.obtenir_piscines(48.8049, 2.1204, construire=_scene_nue)
    cache.obtenir_vehicules(48.8049, 2.1204, "yolo", construire=_scene_nue)
    cache.obtenir_vehicules(48.8049, 2.1204, "rtmdet", construire=_scene_nue)
    assert appels == ["piscines", "rtmdet", "yolo"]


def _noter_le_fil(lecteur, fils):
    """Le lecteur dont chaque lecture note le nom du fil qui la fait."""
    piscines, vehicules = lecteur.piscines, lecteur.vehicules

    def noter(lire):
        def lire_et_noter(*bbox):
            fils.append(threading.current_thread().name)
            return lire(*bbox)
        return lire_et_noter
    lecteur.piscines = noter(piscines)
    lecteur.vehicules = lambda detecteur: noter(vehicules(detecteur))
    return lecteur


def test_une_detection_sans_tache_de_fond_passe_par_le_fil_des_detections(tmp_path):
    """Une couche demandée sans lecture anticipée (page rechargée après un
    échec, service relancé) n'est pas détectée dans le fil de la requête :
    elle prend le fil des détections, de priorité abaissée, et jamais deux
    détections ne tournent ensemble."""
    fils = []
    cache = Cache(str(tmp_path), lire_vehicules=_noter_le_fil(_lecteur_vehicules("tous"), fils))
    cache.obtenir_piscines(48.8049, 2.1204, construire=_scene_nue)
    cache.obtenir_vehicules(48.8049, 2.1204, "yolo", construire=_scene_nue)
    assert len(fils) == 2 and all(nom.startswith("detection") for nom in fils)
    assert os.path.exists(cache.chemin(48.8049, 2.1204, scene.nom_vehicules("yolo")))


class _Session:
    """Doublure d'une session onnxruntime de rtmdet : aucune boîte."""

    def __init__(self, journal=None):
        self.journal = journal if journal is not None else []

    def get_inputs(self):
        import types
        return [types.SimpleNamespace(name="images")]

    def run(self, sorties, entrees):
        import numpy as np
        self.journal.append("inférence")
        return [np.zeros((1, 0, 20), dtype=np.float32)]


def test_les_detections_cedent_le_processeur_au_calcul_de_la_scene(tmp_path):
    """Pendant les lectures d'une scène, une inférence passe ; pendant ses
    toitures et ses houppiers, elle attend qu'ils soient finis."""
    from vue3d.vehicules import Lecteur
    journal = []
    cache = Cache(str(tmp_path), lire_vehicules=Lecteur("rtmdet", {"rtmdet": _Session(journal)}))
    session = cache.lire_vehicules.sessions["rtmdet"]
    lectures, lue, calcul, fin = (threading.Event() for _ in range(4))

    def construire(lat, lon, avancer=None):
        avancer("bâtiments")
        lectures.set()
        lue.wait(5)
        avancer("toitures")
        calcul.set()
        fin.wait(5)
        avancer("houppiers")
        journal.append("scène")
        return b"s", b"j"

    fil_scene = threading.Thread(target=cache.obtenir, args=(48.8049, 2.1204),
                                 kwargs={"construire": construire})
    fil_scene.start()
    lectures.wait(5)
    session.run(None, {})                       # pendant les lectures : passe
    lue.set()
    calcul.wait(5)
    fil_detection = threading.Thread(target=session.run, args=(None, {}))
    fil_detection.start()
    fil_detection.join(0.3)
    assert fil_detection.is_alive() and journal == ["inférence"]
    fin.set()
    fil_scene.join(5)
    fil_detection.join(5)
    assert journal == ["inférence", "scène", "inférence"]


def test_seules_les_sessions_sur_le_processeur_cedent(tmp_path):
    """CoreML calcule sur le GPU : sa session n'attend pas la scène."""
    from vue3d.vehicules import Lecteur

    class Moteur(_Session):
        def __init__(self, moteurs):
            super().__init__()
            self.moteurs = moteurs

        def get_providers(self):
            return self.moteurs

    coreml = Moteur(["CoreMLExecutionProvider", "CPUExecutionProvider"])
    processeur = Moteur(["CPUExecutionProvider"])
    cache = Cache(str(tmp_path), lire_vehicules=Lecteur("tous", {"rtmdet": processeur, "yolo": coreml}))
    assert isinstance(cache.lire_vehicules.sessions["rtmdet"], scene.SessionQuiCede)
    assert cache.lire_vehicules.sessions["yolo"] is coreml


def test_une_scene_en_echec_ne_retient_pas_les_detections(tmp_path):
    from vue3d.vehicules import Lecteur
    cache = Cache(str(tmp_path), lire_vehicules=Lecteur("rtmdet", {"rtmdet": _Session()}))

    def en_panne(lat, lon, avancer=None):
        avancer("toitures")
        raise SceneIncomplete("hauteurs du sursol illisibles : Read timed out")

    with pytest.raises(SceneIncomplete):
        cache.obtenir(48.8049, 2.1204, construire=en_panne)
    fil = threading.Thread(target=cache.lire_vehicules.sessions["rtmdet"].run, args=(None, {}))
    fil.start()
    fil.join(2)
    assert not fil.is_alive()


def test_le_lecteur_des_vehicules_detecte_par_les_sessions_qui_cedent(tmp_path, monkeypatch):
    """Le vrai vehicules.Lecteur, orthophoto et réseau doublés : ses
    détections passent par les sessions du cache, qui attendent les scènes ;
    le lecteur reçu, lui, n'est pas touché."""
    import numpy as np
    from vue3d import vehicules
    monkeypatch.setattr(vehicules.Lecteur, "_orthophoto",
                        staticmethod(lambda *bbox: np.zeros((300, 200, 3), dtype=np.uint8)))
    originale = _Session()
    lecteur = vehicules.Lecteur("rtmdet", {"rtmdet": originale})
    cache = Cache(str(tmp_path), lire_vehicules=lecteur)
    attentes = []
    cede = cache.lire_vehicules.sessions["rtmdet"]
    cede._attendre = lambda: attentes.append(1)
    cache.prelire_vehicules(48.8049, 2.1204)
    dossier, nom = cache.obtenir_piscines(48.8049, 2.1204, construire=_scene_nue)
    cache.obtenir_vehicules(48.8049, 2.1204, "rtmdet", construire=_scene_nue)
    assert os.path.exists(os.path.join(dossier, nom))
    assert attentes and len(attentes) == len(originale.journal)
    assert lecteur.sessions["rtmdet"] is originale and cede.get_inputs()[0].name == "images"


# --- Panneaux solaires : le registre ------------------------------------------------

def _installation(west, south, east, north):
    lon, lat, d = (west + east) / 2, (south + north) / 2, 5 / 111320
    return [{"contour": [[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d]],
             "surface": 100, "kwp": 12, "annee": 2023}]


def test_sans_registre_la_couche_des_panneaux_n_existe_pas(tmp_path):
    cache = Cache(str(tmp_path))
    cache.prelire_panneaux(48.8049, 2.1204)              # sans effet, sans erreur
    with pytest.raises(scene.PanneauxDesactives):
        cache.obtenir_panneaux(48.8049, 2.1204, construire=_scene_nue)


def test_la_couche_des_panneaux_est_lue_une_fois_et_versionnee(tmp_path):
    import gzip
    import json
    appels = []

    def lire(*bbox):
        appels.append(bbox)
        return _installation(*bbox)

    cache = Cache(str(tmp_path), lire_panneaux=lire)
    cache.prelire_panneaux(48.8049, 2.1204)
    d1 = cache.obtenir_panneaux(48.8049, 2.1204, construire=_scene_nue)
    d2 = cache.obtenir_panneaux(48.8049, 2.1204, construire=_scene_nue)
    assert d1 == d2 and len(appels) == 1
    assert scene.NOM_PANNEAUX == f"panneaux-v{scene.PANNEAUX_VERSION}.json.gz"
    couche = json.loads(gzip.decompress(open(os.path.join(d1, scene.NOM_PANNEAUX), "rb").read()))
    assert len(couche["panneaux"]) == 1 and couche["panneaux"][0]["kwp"] == 12


def test_une_base_illisible_ne_met_rien_en_cache(tmp_path):
    en_panne = [True]

    def lire(*bbox):
        if en_panne[0]:
            raise OSError("disk I/O error")
        return _installation(*bbox)

    cache = Cache(str(tmp_path), lire_panneaux=lire)
    with pytest.raises(scene.PanneauxIndisponibles):
        cache.obtenir_panneaux(48.8049, 2.1204, construire=_scene_nue)
    assert not os.path.exists(cache.chemin(48.8049, 2.1204, scene.NOM_PANNEAUX))
    en_panne[0] = False
    cache.obtenir_panneaux(48.8049, 2.1204, construire=_scene_nue)
    assert os.path.exists(cache.chemin(48.8049, 2.1204, scene.NOM_PANNEAUX))


@pytest.mark.parametrize("brute,attendue", [(None, None), ("", None), ("500", 500),
                                            (512, 500), (90, 150), (5000, 1000)])
def test_la_zone_est_arrondie_et_bornee(brute, attendue):
    assert scene.zone_normalisee(brute) == attendue


@pytest.mark.parametrize("brute", ["grand", "nan"])
def test_une_zone_illisible_est_refusee(brute):
    with pytest.raises(ValueError):
        scene.zone_normalisee(brute)


def test_la_zone_fixe_le_cote_nord_sud_et_agrandit_l_anneau():
    lat, lon = 48.8, 2.1
    o, s, e, n = emprise(lat, lon, 1000)
    assert (n - s) * 111320 == pytest.approx(1000)
    assert e - o == pytest.approx(n - s)
    ao, as_, ae, an = scene.emprise_anneau(lat, lon, zone=1000)
    assert (an - as_) * 111320 == pytest.approx(2 * scene.ANNEAU_DEMI_M * 1000 / scene.SCENE_COTE_M)
    # Une zone plus petite que la scène par défaut garde l'anneau d'un kilomètre.
    assert scene.emprise_anneau(lat, lon, zone=200) == scene.emprise_anneau(lat, lon)


def test_chaque_zone_a_sa_scene(tmp_path):
    """La scène par défaut garde son dossier d'avant les zones ; une zone en a
    un autre, construite avec elle."""
    appels = []

    def construire(lat, lon, avancer=None, zone=None):
        appels.append(zone)
        return b"scene", b"jpeg"

    cache = Cache(str(tmp_path))
    defaut = cache.obtenir(48.8049, 2.1204, construire=construire)
    large = cache.obtenir(48.8049, 2.1204, construire=construire, zone="700")
    assert cache.obtenir(48.8049, 2.1204, construire=construire, zone=690) == large
    assert appels == [None, 700]
    assert os.path.basename(defaut) == "48.8049_2.1204"
    assert os.path.basename(large) == "48.8049_2.1204_z700"
    assert cache.avancement(48.8049, 2.1204, "700") == {"etat": "prete"}
    assert cache.avancement(48.8049, 2.1204, "500") == {"etat": "attente"}
