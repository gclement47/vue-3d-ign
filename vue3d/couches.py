"""Couches vectorielles de la Géoplateforme IGN (WFS), sans clé d'API.

Quatre couches servent à la scène :

- `BDTOPO_V3:batiment` : emprises, hauteurs et altitudes de toit déclarées ;
- `BDTOPO_V3:zone_de_vegetation` : OÙ est la végétation (haie, bois, forêt…),
  jamais sa hauteur ;
- `LANDCOVER.FORESTINVENTORY.V2:formation_vegetale` (BD Forêt v2) : l'essence
  dominante d'un massif, jamais celle d'un arbre ;
- `BDTOPO_V3:troncon_de_route` : pour poser un point de vue Street View sur la
  rue qui dessert un bâtiment.

Le serveur WFS de l'IGN renvoie parfois une erreur Java avec un code 200
(« Unable to obtain connection ») : c'est le contenu qui tranche, pas le code.

Il ne rend pas plus de 5 000 objets par requête, sans erreur : au centre de
Paris, 5 000 bâtiments sur les 9 948 d'une emprise de 2 × 2 km. Une emprise
élargie peut passer ce seuil. Les pages (STARTINDEX) ne conviennent pas : non
triées, deux pages s'y recouvrent (3 doublons, donc 3 objets jamais rendus,
sur ces 9 948). L'emprise est donc coupée en quatre jusqu'à ce que chaque
morceau tienne en une réponse, et les objets à cheval sur deux morceaux,
rendus deux fois, ne sont gardés qu'une.
"""

import json

import requests

from .geopf import en_parallele, get_avec_reprise, place

WFS_URL = "https://data.geopf.fr/wfs/ows"
COUCHE_BATIMENTS = "BDTOPO_V3:batiment"
COUCHE_VEGETATION = "BDTOPO_V3:zone_de_vegetation"
COUCHE_FORET = "LANDCOVER.FORESTINVENTORY.V2:formation_vegetale"
COUCHE_ROUTES = "BDTOPO_V3:troncon_de_route"
# Plafond du service, constaté (numberMatched 9 948, numberReturned 5 000).
WFS_PAGE = 5000
# Une zone de 1 km tient en une réponse presque partout ; quatre découpes
# (256 morceaux) laissent une marge très au-delà de ce qu'une scène demande.
WFS_DECOUPES_MAX = 4


def _requete(typenames, west, south, east, north):
    """Une réponse du service, au plus WFS_PAGE objets."""
    url = (f"{WFS_URL}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
           f"&TYPENAMES={typenames}&OUTPUTFORMAT=application/json&SRSNAME=EPSG:4326"
           f"&BBOX={west},{south},{east},{north},EPSG:4326")
    with place():
        reponse = get_avec_reprise(url, timeout=30)
    reponse.raise_for_status()
    texte = reponse.text
    if "java.lang.RuntimeException" in texte or "Unable to obtain connection" in texte:
        raise requests.RequestException(f"erreur serveur IGN sur {typenames}")
    try:
        return json.loads(texte)
    except ValueError as exc:
        raise requests.RequestException(f"réponse illisible sur {typenames}") from exc


def lire_couche(typenames, west, south, east, north, profondeur=0):
    """GeoJSON d'une couche WFS sur une emprise WGS84, sans objet manquant.

    Raises:
        requests.RequestException si le service ne répond pas, ou répond une
        erreur déguisée en succès.
    """
    geojson = _requete(typenames, west, south, east, north)
    attendus = geojson.get("numberMatched")
    if not isinstance(attendus, int) or len(geojson["features"]) >= attendus:
        return geojson
    if profondeur >= WFS_DECOUPES_MAX:
        raise requests.RequestException(
            f"{typenames} : {len(geojson['features'])} objets rendus sur {attendus}")
    milieu_lon, milieu_lat = (west + east) / 2, (south + north) / 2
    quarts = ((west, south, milieu_lon, milieu_lat), (milieu_lon, south, east, milieu_lat),
              (west, milieu_lat, milieu_lon, north), (milieu_lon, milieu_lat, east, north))
    # Les quatre lus ensemble, réunis dans le même ordre qu'un à un : un objet
    # à cheval reste celui du premier quart qui le rend.
    reponses = en_parallele(*(lambda q=q: lire_couche(typenames, *q, profondeur + 1)
                              for q in quarts))
    vus, objets = set(), []
    for reponse in reponses:
        for objet in reponse["features"]:
            if objet.get("id") not in vus:
                vus.add(objet.get("id"))
                objets.append(objet)
    geojson["features"] = objets
    geojson["numberMatched"] = geojson["numberReturned"] = len(objets)
    return geojson
