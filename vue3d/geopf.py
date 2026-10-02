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

import concurrent.futures
import contextlib
import threading
import time

import requests

# Deux essais de plus, espacés : au-delà on s'acharne sur un service qui a
# manifestement un problème, et l'appelant réessaiera plus tard.
GEOPF_ESSAIS = 3
GEOPF_ATTENTE_S = 3

# Requêtes simultanées vers la Géoplateforme, pour tout le processus : les
# lectures d'une scène, celles d'une autre construite en même temps, les
# ouvrages et l'orthophoto des véhicules lus en tâche de fond se partagent ces
# places. Mesuré le 1er octobre 2026 sur les 16 requêtes d'une zone de
# 1 000 m à Gordes, les plus longues lancées d'abord, médianes de 4 tours :
# 10,7 s une à une, 2,7 s à 4 à la fois, 2,9 s à 8, 3,4 s à 16. Une scène
# ne gagne rien au-delà de 4 : il reste la plus longue requête, l'orthophoto
# que le serveur met 1,5 à 3,5 s à rendre. Deux scènes à la fois (32
# requêtes, 5 tours) : 4,6 s à 8, 3,6 s à 16, 4,1 s à 32. 8 laisse de la
# marge aux couches lues en tâche de fond sans charger un service public.
#
# Les refus sporadiques (400 « LayerNotDefined » d'un serveur qui ne connaît
# pas la couche, 502) ne touchent que le WMS, et pas davantage en parallèle :
# 15 % des requêtes WMS une à une (39 sur 264), 12 % à 4 à 32 à la fois (75
# sur 612), sans reprise. get_avec_reprise les absorbe ; en parallèle, leurs
# attentes se recouvrent au lieu de s'additionner.
GEOPF_SIMULTANEES = 8
_places = threading.BoundedSemaphore(GEOPF_SIMULTANEES)


@contextlib.contextmanager
def place():
    """Une des GEOPF_SIMULTANEES places, le temps d'une requête et de ses
    reprises. À prendre autour de `get_avec_reprise` seulement, jamais autour
    d'une attente sur d'autres lectures : un fil qui garde sa place en
    attendant celles qu'il a lancées peut bloquer tout le processus."""
    with _places:
        yield


def en_parallele(*lectures):
    """Résultats de lectures (fonctions sans argument) lancées ensemble, dans
    l'ordre où elles sont données.

    Un fil par lecture : elles attendent le réseau, pas un cœur, et c'est
    `place()` qui borne les requêtes. Des fils neufs plutôt qu'un bassin
    partagé : une lecture peut en lancer d'autres (les quarts d'une couche
    WFS), qui n'ont pas à attendre qu'un fil du même bassin se libère.

    Raises:
        l'exception de la première lecture en échec, dès qu'elle échoue :
        les lectures pas encore commencées sont abandonnées, celles en cours
        finissent sans que personne ne les attende.
    """
    if len(lectures) <= 1:
        return [lecture() for lecture in lectures]
    bassin = concurrent.futures.ThreadPoolExecutor(max_workers=len(lectures),
                                                   thread_name_prefix="geopf")
    try:
        futurs = [bassin.submit(lecture) for lecture in lectures]
        faits, _ = concurrent.futures.wait(futurs, return_when=concurrent.futures.FIRST_EXCEPTION)
        for futur in futurs:
            if futur in faits and futur.exception() is not None:
                raise futur.exception()
        return [futur.result() for futur in futurs]
    finally:
        bassin.shutdown(wait=False, cancel_futures=True)


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
