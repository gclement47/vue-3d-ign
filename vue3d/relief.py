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
    valides = grille[~trous]
    if not valides.size:
        return None
    zero = float(np.floor(valides.min()))
    quant = np.rint((grille - zero) / RELIEF_PAS_M)
    quant = np.clip(quant, RELIEF_SENTINELLE + 1, 32767)
    quant[trous] = RELIEF_SENTINELLE
    return {
        "width": taille, "height": taille,
        "bbox": [west, south, east, north],
        "zero_m": round(zero, 1),
        "pas_m": RELIEF_PAS_M,
        "source": "RGE ALTI (IGN)",
        "precision": "1 m, modèle de terrain",
        "altitudes": base64.b64encode(quant.astype("<i2").tobytes()).decode("ascii"),
    }
