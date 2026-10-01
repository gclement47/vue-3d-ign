"""Lecture WFS (vue3d/couches.py) : réponses plafonnées du service."""
import json
import re

import pytest
import requests

from vue3d import couches


class _Reponse:
    def __init__(self, corps):
        self.text = json.dumps(corps)

    def raise_for_status(self):
        pass


def _service(objets, plafond):
    """Faux WFS : rend au plus `plafond` des objets (id, lon, lat) de la
    BBOX, et annonce combien il y en avait."""
    def get(url, timeout=None):
        o, s, e, n = map(float, re.search(r"BBOX=([^,]+),([^,]+),([^,]+),([^,]+),", url).groups())
        dedans = [{"id": i, "geometry": {"type": "Point", "coordinates": [x, y]}}
                  for i, x, y in objets if o <= x <= e and s <= y <= n]
        return _Reponse({"features": dedans[:plafond], "numberMatched": len(dedans),
                         "numberReturned": min(plafond, len(dedans))})
    return get


def test_une_reponse_plafonnee_est_completee_par_quarts(monkeypatch):
    # Un objet sur la ligne de partage : rendu par deux quarts, gardé une fois.
    objets = [(f"b.{i}", 0.1 + 0.2 * (i % 5), 0.1 + 0.2 * (i // 5)) for i in range(25)]
    objets.append(("b.milieu", 0.5, 0.5))
    monkeypatch.setattr(couches, "get_avec_reprise", _service(objets, plafond=10))
    geojson = couches.lire_couche("X", 0, 0, 1, 1)
    ids = [f["id"] for f in geojson["features"]]
    assert sorted(ids) == sorted(i for i, _, _ in objets)
    assert geojson["numberReturned"] == len(objets)


def test_les_quarts_sont_lus_ensemble(monkeypatch):
    """Les quatre quarts partent en même temps (la barrière ne s'ouvre qu'à
    quatre), et le résultat est celui de la lecture un à un : mêmes objets,
    dans le même ordre."""
    import threading
    objets = [(f"b.{i}", 0.1 + 0.2 * (i % 5), 0.1 + 0.2 * (i // 5)) for i in range(25)]
    objets.append(("b.milieu", 0.5, 0.5))
    service = _service(objets, plafond=10)
    barriere = threading.Barrier(4, timeout=5)
    appels = []

    def get(url, timeout=None):
        appels.append(url)
        if len(appels) > 1:
            barriere.wait()
        return service(url, timeout)

    monkeypatch.setattr(couches, "get_avec_reprise", get)
    geojson = couches.lire_couche("X", 0, 0, 1, 1)
    # Un à un : sud-ouest, sud-est, nord-ouest, nord-est, chacun dans l'ordre
    # du service ; l'objet du milieu, à cheval, vient du premier quart.
    attendus = []
    for o, s, e, n in ((0, 0, .5, .5), (.5, 0, 1, .5), (0, .5, .5, 1), (.5, .5, 1, 1)):
        for i, x, y in objets:
            if o <= x <= e and s <= y <= n and i not in attendus:
                attendus.append(i)
    assert [f["id"] for f in geojson["features"]] == attendus


def test_une_couche_trop_dense_est_une_lecture_en_echec(monkeypatch):
    # Cinq objets au même point : aucun découpage ne les sépare.
    objets = [(f"b.{i}", 0.3, 0.3) for i in range(5)]
    monkeypatch.setattr(couches, "get_avec_reprise", _service(objets, plafond=2))
    with pytest.raises(requests.RequestException):
        couches.lire_couche("X", 0, 0, 1, 1)
