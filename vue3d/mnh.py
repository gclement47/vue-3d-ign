"""Modèle Numérique de Hauteur (MNH) LiDAR HD de l'IGN.

Le MNH donne la hauteur du sursol au-dessus du terrain, en mètres : arbres,
haies, mais aussi bâtiments et tout ce qui dépasse. C'est la seule source
ouverte qui chiffre la hauteur de la végétation — la couche vectorielle
`BDTOPO_V3:zone_de_vegetation` ne porte que la nature (Haie, Bois, Forêt…) et
la géométrie, jamais de hauteur (vérifié via DescribeFeatureType).

Deux particularités du service, toutes deux constatées contre le service
réel :

1. `GetFeatureInfo` est refusé sur cette couche (`LayerNotQueryable`). Le seul
   moyen d'obtenir des valeurs et non des pixels colorés est un `GetMap` au
   format `image/x-bil;bits=32`, qui renvoie les hauteurs brutes en float32
   little-endian, ligne par ligne depuis le nord-ouest.
2. En WMS 1.3.0 et EPSG:4326, la BBOX s'écrit `lat,lon` — l'ordre inverse de
   celui des requêtes WFS voisines. Passée en `lon,lat`, la réponse est une
   dalle vide, sans erreur.

Couverture : le LiDAR HD est un programme en cours de déploiement. Mesuré sur
un échantillon de 60 bâtiments tirés au hasard en France, 77 % des points
sont couverts ; les trous observés se concentrent sur le Nord et la Bretagne.
D'où `couvert` dans la réponse, pour que l'appelant distingue « pas de
végétation » de « pas de donnée ».
"""

import struct

import logging

import requests

from .geopf import get_avec_reprise

MNH_LAYER = "IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.WGS84G"
# Repli hors couverture LiDAR HD : modèle numérique de surface photogrammétrique
# moins modèle de terrain, tous deux nationaux. Moins fin que le LiDAR — il
# lisse les pics (mesuré : 7,6 m contre 8,8 m sur le même arbre) — mais là où le
# LiDAR renvoie -9999, il donne des hauteurs plausibles (11,6 m sur une zone urbaine du Nord).
MNS_LAYER = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES.MNS"
MNT_LAYER = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"

# Résolution par défaut : 2 m. À 1 m la grille quadruple pour un gain nul à
# l'écran (les volumes sont déjà rendus en pavés), et la réponse JSON dépasse
# le demi-mégaoctet.
MNH_RESOLUTION_M = 2.0
# Garde-fou : le service refuse au-delà de 2048, et une grille plus fine ne
# passerait de toute façon pas confortablement dans la page.
MNH_MAX_PIXELS = 512
# En deçà, c'est du sol, du mobilier urbain ou du bruit de mesure : inutile de
# le transporter jusqu'au navigateur.
MNH_SEUIL_M = 2.0


def _grille_depuis_bil(contenu, largeur, hauteur):
    """Décode le BIL32 en liste de hauteurs, nettoyée des valeurs aberrantes."""
    attendu = largeur * hauteur * 4
    if len(contenu) < attendu:
        raise ValueError(f"BIL tronqué : {len(contenu)} octets pour {attendu} attendus")
    brut = struct.unpack(f"<{largeur * hauteur}f", contenu[:attendu])
    # Le service signale l'absence de donnée par de grandes valeurs négatives ;
    # les négatifs résiduels sont du bruit autour de zéro.
    return [0.0 if (v < 0 or v > 200 or v != v) else round(v, 1) for v in brut]


def _grille_wms(layer, west, south, east, north, largeur, hauteur, brut=False):
    """GetMap BIL32 sur une couche altimétrique, décodé en liste de floats.

    brut=True conserve les valeurs telles quelles (altitudes, nodata négatif) :
    c'est l'appelant qui soustrait MNS et MNT avant de nettoyer.
    """
    # WMS 1.3.0 + EPSG:4326 => BBOX en lat,lon.
    url = (
        "https://data.geopf.fr/wms-r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={layer}&STYLES=&CRS=EPSG:4326"
        f"&BBOX={south},{west},{north},{east}"
        f"&WIDTH={largeur}&HEIGHT={hauteur}"
        "&FORMAT=image/x-bil;bits=32"
    )
    reponse = get_avec_reprise(url)
    reponse.raise_for_status()
    if "bil" not in reponse.headers.get("Content-Type", ""):
        # Le service répond en XML quand il refuse la requête.
        raise requests.RequestException(
            f"Réponse {layer} inattendue ({reponse.headers.get('Content-Type')}): "
            f"{reponse.text[:200]}")
    if brut:
        attendu = largeur * hauteur * 4
        if len(reponse.content) < attendu:
            raise requests.RequestException(f"BIL tronqué sur {layer}")
        return list(struct.unpack(f"<{largeur * hauteur}f", reponse.content[:attendu]))
    return _grille_depuis_bil(reponse.content, largeur, hauteur)


journal = logging.getLogger(__name__)


def fetch_mnh_grid(west, south, east, north, resolution_m=MNH_RESOLUTION_M,
                   max_pixels=MNH_MAX_PIXELS):
    """Grille de hauteurs du sursol sur une emprise WGS84.

    Returns:
        dict: width, height, bbox, resolution_m, couvert, values (m, pas de pas
        de grille constant en degrés). `values` est ordonné par lignes depuis
        le nord-ouest, comme l'image source.
    """
    west, south = float(west), float(south)
    east, north = float(east), float(north)

    # Pas de grille en degrés, déduit de la résolution voulue au centre.
    import math
    lat_moy = (south + north) / 2
    m_par_deg_lon = 111320 * math.cos(math.radians(lat_moy))
    # max_pixels : 512 pour la végétation (payload navigateur) ; les toits
    # demandent 0,5 m et restent côté serveur, donc jusqu'à 2048 (plafond WMS).
    largeur = int(min(max((east - west) * m_par_deg_lon / resolution_m, 8), max_pixels))
    hauteur = int(min(max((north - south) * 110540 / resolution_m, 8), max_pixels))

    valeurs = _grille_wms(MNH_LAYER, west, south, east, north, largeur, hauteur)
    source = "lidar_hd"
    # Une dalle hors couverture LiDAR revient entièrement à zéro (nodata). Le
    # seuil évite de confondre ce cas avec une emprise réellement rase.
    if max(valeurs) <= 0.5:
        try:
            mns = _grille_wms(MNS_LAYER, west, south, east, north, largeur, hauteur, brut=True)
            mnt = _grille_wms(MNT_LAYER, west, south, east, north, largeur, hauteur, brut=True)
            repli = [round(min(max(a - b, 0.0), 200.0), 1) if (a > -1000 and b > -1000) else 0.0
                     for a, b in zip(mns, mnt)]
            if max(repli) > 0.5:
                valeurs, source = repli, "mns_mnt"
        except requests.RequestException as exc:
            # Le repli est un bonus : son échec ne doit pas faire perdre la
            # réponse LiDAR, même vide.
            journal.warning("Repli MNS-MNT indisponible: %s", exc)

    data = {
        "width": largeur,
        "height": hauteur,
        "bbox": [west, south, east, north],
        "resolution_m": resolution_m,
        "couvert": max(valeurs) > 0.5,
        # lidar_hd : hauteurs mesurées. mns_mnt : estimées par photogrammétrie,
        # à afficher comme telles.
        "source": source if max(valeurs) > 0.5 else None,
        "seuil_m": MNH_SEUIL_M,
        "values": valeurs,
    }
    return data
