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

from .geopf import get_avec_reprise

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
    largeur = int(min(max(largeur_m / resolution_m, 64), max_pixels))
    hauteur = int(min(max(hauteur_m / resolution_m, 64), max_pixels))
    return _image_wms(west, south, east, north, largeur, hauteur), largeur, hauteur


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
