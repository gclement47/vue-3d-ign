"""Relief RGE ALTI, embarqué dans la scène plutôt que téléchargé par la vue.

La vue demandait elle-même au WMS de l'IGN une grille BIL float32 de 256 points
de côté, soit 263 Ko à chaque ouverture, sur le même hôte que les seize tuiles
d'orthophoto — un navigateur n'ouvrant que six connexions par hôte, tout cela
se met en file.

La même grille, quantifiée en décimètres sur des entiers 16 bits et rapportée à
son minimum, tient dans une dizaine de kilo-octets une fois la scène
compressé : le relief est lisse, donc ses différences le sont aussi. Mesuré sur
un site de plaine, 2,4 m de dénivelé sur 356 m.

La résolution est celle que la vue employait déjà, et l'erreur d'interpolation
en découle. Mesurée contre une référence à 512 points de côté sur la même
emprise :

    grille      maille     erreur médiane   p95      compressé
     96          3,7 m        0,03 m       0,27 m       2 Ko
    128          2,8 m        0,02 m       0,19 m       4 Ko
    178          2,0 m        0,01 m       0,10 m       7 Ko
    256          1,4 m        0,00 m       0,07 m      10 Ko

Le maillage du terrain place un sommet tous les 2 m : 256 le nourrit largement,
et c'est le même coût qu'une grille plus grossière une fois compressée.

Hors couverture (étranger, outre-mer) le service rend −99999 partout. Au-delà
de quelques pour cent de trous on est en bord de couverture : mieux vaut
laisser la vue basculer sur son repli mondial que rapiécer.
"""

import base64
import struct

import numpy as np
import requests

from .geopf import get_avec_reprise

RELIEF_LAYER = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"
RELIEF_TAILLE = 256
RELIEF_NODATA = -99999
# Au-delà, l'emprise est en bord de couverture : on ne renvoie rien.
RELIEF_PART_TROUS_MAX = 0.02
# Pas de quantification : le décimètre, très en deçà de l'erreur propre du RGE
# ALTI (0,54 à 0,72 m de médiane selon les sites).
RELIEF_PAS_M = 0.1
RELIEF_SENTINELLE = -32768


def _bil(west, south, east, north, largeur, hauteur):
    """GetMap BIL32 sur la couche d'altitude, décodé en grille de flottants.

    Trois pièges, tous vérifiés contre le service : en WMS 1.3.0 avec EPSG:4326
    la BBOX s'écrit lat,lon ; le corps est du float32 petit-boutiste brut,
    première ligne au nord ; une erreur arrive en XML avec un code 200, c'est
    donc le type de contenu qui tranche.
    """
    url = (
        "https://data.geopf.fr/wms-r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={RELIEF_LAYER}&STYLES=&CRS=EPSG:4326"
        "&FORMAT=image/x-bil;bits=32"
        f"&WIDTH={largeur}&HEIGHT={hauteur}"
        f"&BBOX={south},{west},{north},{east}"
    )
    reponse = get_avec_reprise(url)
    reponse.raise_for_status()
    if "bil" not in reponse.headers.get("Content-Type", ""):
        raise requests.RequestException(
            f"Réponse relief inattendue ({reponse.headers.get('Content-Type')}): "
            f"{reponse.text[:200]}")
    attendu = largeur * hauteur * 4
    if len(reponse.content) < attendu:
        raise requests.RequestException("BIL relief tronqué")
    return np.frombuffer(reponse.content[:attendu], dtype="<f4").reshape(hauteur, largeur)


def _quantifier(grille, trous, west, south, east, north):
    """Grille d'altitudes en décimètres au-dessus de son minimum, en base64."""
    hauteur, largeur = grille.shape
    zero = float(np.floor(grille[~trous].min()))
    quant = np.rint((grille - zero) / RELIEF_PAS_M)
    quant = np.clip(quant, RELIEF_SENTINELLE + 1, 32767)
    quant[trous] = RELIEF_SENTINELLE
    return {
        "width": largeur, "height": hauteur,
        "bbox": [west, south, east, north],
        "zero_m": round(zero, 1),
        "pas_m": RELIEF_PAS_M,
        "source": "RGE ALTI (IGN)",
        "precision": "1 m, modèle de terrain",
        "altitudes": base64.b64encode(quant.astype("<i2").tobytes()).decode("ascii"),
    }


def fetch_relief(west, south, east, north, taille=RELIEF_TAILLE):
    """Relief de l'emprise, prêt à embarquer dans la scène.

    Returns:
        dict(width, height, bbox, zero_m, pas_m, altitudes) où `altitudes` est
        un tableau d'entiers 16 bits encodé en base64, en pas de 0,1 m au-dessus
        de `zero_m`, ligne 0 au nord. Les trous valent RELIEF_SENTINELLE.
        None si l'emprise n'est pas couverte.
    """
    west, south, east, north = map(float, (west, south, east, north))
    grille = _bil(west, south, east, north, taille, taille).astype(np.float64)
    trous = grille <= RELIEF_NODATA + 1
    if trous.mean() > RELIEF_PART_TROUS_MAX:
        return None
    if trous.all():
        return None
    return _quantifier(grille, trous, west, south, east, north)


# --- Anneau de contexte ---------------------------------------------------
# Autour de la scène, un relief grossier évite l'effet de maquette posée dans
# le vide. Il se fond dans le brouillard de la vue : sa précision compte peu,
# son poids beaucoup moins encore. Mesuré sur 2 km de côté contre une
# référence à 512 points, pour la même emprise :
#
#                      grille 64   96     128    192
#   maille               32 m     21 m   16 m   10 m
#   Versailles  méd.     0,38     0,24   0,18   0,12 m    (34 m de dénivelé)
#               p95      2,2      1,5    1,2    0,7 m
#   Gordes      méd.     2,3      1,4    1,1    0,7 m     (242 m)
#               p95     10,5      6,9    5,2    3,5 m
#   Chamonix    p95     27,9     18,9   15,0    9,1 m     (671 m)
#   compressé           3 Ko     7 Ko  12 Ko   26 Ko
#
# 128 : un mètre d'erreur médiane même en pente, pour 12 Ko au plus.
ANNEAU_TAILLE = 128
# L'anneau touche la mer ou la frontière bien plus souvent que la scène : un
# quart de trous à Saint-Malo comme à Menton. Il n'est donc pas refusé pour
# autant, ses trous restent simplement vides.
#
# Mais le service rééchantillonne SANS masquer ses −99 999 : il les mélange à
# leurs voisins, et sort des altitudes de −8 700 m ou −61 m à plusieurs pixels
# du trou. Mesuré à Saint-Malo et Menton, un seuil à −20 m suivi d'un
# élargissement de deux pixels (31 m) est le premier qui ne laisse que des
# altitudes plausibles (−5 m au plus bas, sur l'estran).
ANNEAU_TROU_SOUS_M = -20
ANNEAU_ELARGISSEMENT = 2


def _elargir(masque, pas):
    """Dilatation en croix de `pas` pixels, sans dépendance à scipy."""
    for _ in range(pas):
        m = masque.copy()
        m[1:] |= masque[:-1]
        m[:-1] |= masque[1:]
        m[:, 1:] |= masque[:, :-1]
        m[:, :-1] |= masque[:, 1:]
        masque = m
    return masque


def fetch_relief_anneau(west, south, east, north, taille=ANNEAU_TAILLE):
    """Relief grossier autour de la scène, trous compris.

    Même format que `fetch_relief`. None seulement si rien n'est couvert.
    """
    west, south, east, north = map(float, (west, south, east, north))
    grille = _bil(west, south, east, north, taille, taille).astype(np.float64)
    trous = _elargir(grille < ANNEAU_TROU_SOUS_M, ANNEAU_ELARGISSEMENT)
    if trous.all():
        return None
    return _quantifier(grille, trous, west, south, east, north)


def echantillonneur(relief):
    """Altitude du relief embarqué en un point, telle que la vue la lit.

    Même interpolation bilinéaire que la page (nœuds de la grille aux bords de
    l'emprise, ligne 0 au nord) : une hauteur comptée ici au-dessus du relief
    est celle que la vue dessinera.

    Returns:
        fonction (lon, lat) -> altitude en mètres, ou None hors de la grille
        et près d'un trou ; None si la scène n'a pas de relief.
    """
    if not relief or not relief.get("altitudes"):
        return None
    largeur, hauteur = relief["width"], relief["height"]
    quant = np.frombuffer(base64.b64decode(relief["altitudes"]), dtype="<i2")
    if quant.size != largeur * hauteur:
        return None
    alt = relief["zero_m"] + quant.reshape(hauteur, largeur).astype(np.float64) * relief["pas_m"]
    alt[quant.reshape(hauteur, largeur) == RELIEF_SENTINELLE] = np.nan
    west, south, east, north = relief["bbox"]

    def altitude(lon, lat):
        fx = (lon - west) / (east - west) * (largeur - 1)
        fy = (north - lat) / (north - south) * (hauteur - 1)
        if not (0 <= fx <= largeur - 1 and 0 <= fy <= hauteur - 1):
            return None
        x0, y0 = int(fx), int(fy)
        x1, y1 = min(x0 + 1, largeur - 1), min(y0 + 1, hauteur - 1)
        ax, ay = fx - x0, fy - y0
        a = ((alt[y0, x0] * (1 - ax) + alt[y0, x1] * ax) * (1 - ay)
             + (alt[y1, x0] * (1 - ax) + alt[y1, x1] * ax) * ay)
        return None if np.isnan(a) else float(a)

    return altitude
