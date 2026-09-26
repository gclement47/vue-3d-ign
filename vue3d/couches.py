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
"""

import json

import requests

from .geopf import get_avec_reprise

WFS_URL = "https://data.geopf.fr/wfs/ows"
COUCHE_BATIMENTS = "BDTOPO_V3:batiment"
COUCHE_VEGETATION = "BDTOPO_V3:zone_de_vegetation"
COUCHE_FORET = "LANDCOVER.FORESTINVENTORY.V2:formation_vegetale"
COUCHE_ROUTES = "BDTOPO_V3:troncon_de_route"


def lire_couche(typenames, west, south, east, north):
    """GeoJSON d'une couche WFS sur une emprise WGS84.

    Raises:
        requests.RequestException si le service ne répond pas, ou répond une
        erreur déguisée en succès.
    """
    url = (f"{WFS_URL}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
           f"&TYPENAMES={typenames}&OUTPUTFORMAT=application/json&SRSNAME=EPSG:4326"
           f"&BBOX={west},{south},{east},{north},EPSG:4326")
    reponse = get_avec_reprise(url, timeout=30)
    reponse.raise_for_status()
    texte = reponse.text
    if "java.lang.RuntimeException" in texte or "Unable to obtain connection" in texte:
        raise requests.RequestException(f"erreur serveur IGN sur {typenames}")
    try:
        return json.loads(texte)
    except ValueError as exc:
        raise requests.RequestException(f"réponse illisible sur {typenames}") from exc
