"""Accès aux services de la Géoplateforme IGN, avec reprise sur incident.

Le service tombe régulièrement en délai d'attente, et d'autant plus qu'on lui
demande de gros rasters : la construction d'une scène lui réclame une
grille de 2 048 pixels de côté, une orthophoto de même taille et une mosaïque.
Constaté en traitant plusieurs centaines de sites d'affilée : des « Read timed
out » isolés, dont un qui avait privé une scène de son orthophoto — donc d'une
végétation correctement classée — et l'avait figée ainsi dans le cache.

Ses refus sont sporadiques eux aussi, ce qui est contre-intuitif : une requête
de relief rejetée en 400 pendant un traitement par lots a été rejouée telle quelle six
fois de suite, six fois en 200. D'où la reprise sur TOUT échec, refus compris,
et non sur les seules erreurs de transport. Ces lectures sont des GET
idempotents sur des URL dont la forme est vérifiée par ailleurs : réessayer ne
coûte qu'un aller-retour, publier une scène amputée coûte bien plus.
"""

import time

import requests

# Deux essais de plus, espacés : au-delà on s'acharne sur un service qui a
# manifestement un problème, et l'appelant réessaiera plus tard.
GEOPF_ESSAIS = 3
GEOPF_ATTENTE_S = 3


def get_avec_reprise(url, timeout=30, essais=GEOPF_ESSAIS, attente=GEOPF_ATTENTE_S):
    """GET sur la Géoplateforme, en réessayant les incidents de transport.

    Raises:
        requests.RequestException si tous les essais échouent.
    """
    dernier = None
    for essai in range(essais):
        try:
            reponse = requests.get(url, timeout=timeout)
        except requests.RequestException as exc:
            dernier = exc
        else:
            if reponse.ok:
                return reponse
            dernier = requests.RequestException(
                f"HTTP {reponse.status_code} sur {url[:80]}")
        if essai < essais - 1:
            time.sleep(attente)
    raise dernier
