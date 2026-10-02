"""Indice de verdure de l'orthophoto IGN, sur la grille du MNH.

Le MNH LiDAR HD ne distingue pas un toit d'un feuillage : sous un arbre, une
annexe de 12 m² lit 12 m de « toit » parfaitement plat, et passe tous les
tests de forme. L'orthophoto est le second avis : là où elle est verte, le
LiDAR mesure une canopée, pas une toiture.

L'indice est ExG = 2G − R − B, mesuré sur un site réel avec les zones de
végétation BD TOPO pour vérité terrain : végétation à +19 de médiane,
toitures à −16 ; au seuil 4, 93 % de la végétation est reconnue et 2 % des
toitures passent pour du vert. L'orthophoto n'a pas de proche infrarouge, un
NDVI est donc hors de portée ; ExG suffit pour trancher feuillage / toiture.

La grille est demandée à la taille exacte de la grille MNH : une cellule de
hauteur et une cellule de verdure se correspondent alors par index.
"""

import io

import numpy as np
import requests

from .geopf import get_avec_reprise, place

ORTHO_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"
# Même seuil que le classement de la végétation côté rendu.
EXG_SEUIL = 4


def exg_depuis_image(contenu):
    """Décode une image RGB en grille ExG (int8, bornée à ±127)."""
    from PIL import Image
    img = Image.open(io.BytesIO(contenu)).convert("RGB")
    rgb = np.asarray(img, dtype=np.int16)
    exg = 2 * rgb[:, :, 1] - rgb[:, :, 0] - rgb[:, :, 2]
    return np.clip(exg, -127, 127).astype(np.int8)


def _image_wms(west, south, east, north, largeur, hauteur):
    # WMS 1.3.0 + EPSG:4326 => BBOX en lat,lon, comme pour le MNH.
    url = (
        "https://data.geopf.fr/wms-r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={ORTHO_LAYER}&STYLES=&CRS=EPSG:4326"
        f"&BBOX={south},{west},{north},{east}"
        f"&WIDTH={largeur}&HEIGHT={hauteur}&FORMAT=image/jpeg"
    )
    # Le WMS de l'IGN renvoie sporadiquement un 400 sur une requête valide
    # (constaté sur une image de 516 × 712, servie en 200 l'instant d'après) :
    # c'est get_avec_reprise qui absorbe ces refus.
    with place():
        reponse = get_avec_reprise(url)
    reponse.raise_for_status()
    if "image" not in reponse.headers.get("Content-Type", ""):
        raise requests.RequestException(
            f"Réponse orthophoto inattendue ({reponse.headers.get('Content-Type')}): "
            f"{reponse.text[:200]}")
    return reponse.content


# Résolution de la mosaïque servie avec la scène. Les seize tuiles WMTS que
# la vue assemblait au zoom 18 valent 0,39 m par pixel à nos latitudes : on
# reprend cet ordre de grandeur, pour une seule requête au lieu de seize.
ORTHO_MOSAIQUE_RESOLUTION_M = 0.4


def fetch_ortho_jpeg(west, south, east, north, resolution_m=ORTHO_MOSAIQUE_RESOLUTION_M,
                     max_pixels=2048):
    """Orthophoto de l'emprise en une seule image JPEG.

    L'image est demandée en EPSG:4326, donc en projection géographique : ses
    colonnes suivent la longitude et ses lignes la latitude. Sa largeur et sa
    hauteur doivent donc suivre l'étendue en MÈTRES, sans quoi elle sort
    étirée — à 49° de latitude, un degré de longitude vaut deux tiers d'un
    degré de latitude.
    """
    import math
    west, south, east, north = map(float, (west, south, east, north))
    lat_moy = (south + north) / 2
    largeur_m = (east - west) * 111320 * math.cos(math.radians(lat_moy))
    hauteur_m = (north - south) * 111320
    # Plafonnés ensemble, d'un même facteur : bornés chacun de son côté, le
    # plus long seul raccourcirait et l'image sortirait étirée — le cas des
    # zones de plus de 820 m (2 048 px à 0,4 m).
    reduction = min(1.0, max_pixels / max(largeur_m / resolution_m, hauteur_m / resolution_m))
    largeur = int(max(largeur_m / resolution_m * reduction, 64))
    hauteur = int(max(hauteur_m / resolution_m * reduction, 64))
    return _image_wms(west, south, east, north, largeur, hauteur), largeur, hauteur


# Tuiles de l'orthophoto des détections lues en même temps. Une zone de
# 1 000 m en compte six (5 000 px sur 3 600 à 3 800, en tuiles de 2 048),
# l'emprise par défaut une seule. Mesuré le 2026-10-01 sur les six de Gordes,
# trois fois en alternant, médianes : 11,4 s une à une, 7,4 s à deux, 5,1 s à
# trois, 4,8 s à six. Les octets rendus sont ceux de la lecture une à une.
# Sur une centaine de tuiles lues ainsi, une a été refusée en 400 trois fois
# de suite (à deux fils, pendant qu'un autre essai sondait le service) :
# l'image entière échoue alors, comme avant pour une tuile, et la couche
# n'est pas écrite.
ORTHO_RGB_FILS = 6


def fetch_ortho_rgb(west, south, east, north, resolution_m, tuile_max=2048, fils=ORTHO_RGB_FILS):
    """Orthophoto de l'emprise à `resolution_m` exactement, en tableau RGB
    (hauteur, largeur, 3), assemblée de tuiles d'au plus `tuile_max` pixels,
    lues `fils` à la fois.

    Pour la détection des véhicules (0,2 m), qui ne supporte pas d'image plus
    grossière : ses réseaux et ses filtres de taille comptent en pixels de
    0,2 m. Une zone de 1 000 m en demande 5 000 de côté ; plafonnée à 2 048
    comme la mosaïque, l'image revenait à 0,49 m, les voitures y étaient 2,5
    fois trop petites et presque toutes écartées.

    En EPSG:4326, pixels et degrés sont proportionnels : chaque tuile est la
    portion exacte de l'emprise qui correspond à ses pixels. Le découpage ne
    dépend pas de `fils` : une tuile coupée autrement serait compressée
    autrement par le serveur, et l'image ne serait plus la même au pixel près.

    Raises:
        requests.RequestException si une tuile n'a pas pu être lue.
    """
    import concurrent.futures
    import math
    from PIL import Image
    west, south, east, north = map(float, (west, south, east, north))
    lat_moy = (south + north) / 2
    largeur = int(max((east - west) * 111320 * math.cos(math.radians(lat_moy)) / resolution_m, 64))
    hauteur = int(max((north - south) * 111320 / resolution_m, 64))
    dlon, dlat = (east - west) / largeur, (north - south) / hauteur
    rgb = np.zeros((hauteur, largeur, 3), dtype=np.uint8)

    def lire(y0, x0):
        """Une tuile, lue, décodée et posée à sa place : chaque fil écrit sa
        propre portion du tableau."""
        l, h = min(tuile_max, largeur - x0), min(tuile_max, hauteur - y0)
        # Lignes depuis le nord : la tuile y0 commence à north - y0 * dlat.
        contenu = _image_wms(west + x0 * dlon, north - (y0 + h) * dlat,
                             west + (x0 + l) * dlon, north - y0 * dlat, l, h)
        morceau = np.asarray(Image.open(io.BytesIO(contenu)).convert("RGB"))
        if morceau.shape[:2] != (h, l):
            raise requests.RequestException(
                f"tuile d'orthophoto de {morceau.shape[1]} × {morceau.shape[0]} px "
                f"pour {l} × {h} demandés")
        rgb[y0:y0 + h, x0:x0 + l] = morceau

    tuiles = [(y0, x0) for y0 in range(0, hauteur, tuile_max) for x0 in range(0, largeur, tuile_max)]
    if fils <= 1 or len(tuiles) == 1:
        for t in tuiles:
            lire(*t)
        return rgb
    with concurrent.futures.ThreadPoolExecutor(min(fils, len(tuiles)),
                                               thread_name_prefix="orthophoto") as bassin:
        taches = [bassin.submit(lire, *t) for t in tuiles]
        try:
            for tache in taches:
                tache.result()
        except BaseException:
            # Une tuile illisible rend l'image fausse : les tuiles pas encore
            # parties ne le sont jamais.
            for tache in taches:
                tache.cancel()
            raise
    return rgb


def fetch_exg_grid(west, south, east, north, largeur, hauteur):
    """Grille ExG (numpy int8, hauteur × largeur) de l'orthophoto sur l'emprise.

    Ordonnée par lignes depuis le nord-ouest, comme la grille MNH : une cellule
    de hauteur et une cellule de verdure se correspondent par index.
    """
    west, south, east, north = map(float, (west, south, east, north))
    exg = exg_depuis_image(_image_wms(west, south, east, north, largeur, hauteur))
    if exg.shape != (hauteur, largeur):
        # Le service peut renvoyer une image d'une autre taille s'il bride la
        # requête : on ré-échantillonne au plus proche pour garder l'alignement.
        from PIL import Image
        img = Image.fromarray(exg.astype(np.float32))
        exg = np.asarray(img.resize((largeur, hauteur), Image.NEAREST)).astype(np.int8)
    return exg
