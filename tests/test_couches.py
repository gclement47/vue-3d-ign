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


def test_une_couche_trop_dense_est_une_lecture_en_echec(monkeypatch):
    # Cinq objets au même point : aucun découpage ne les sépare.
    objets = [(f"b.{i}", 0.3, 0.3) for i in range(5)]
    monkeypatch.setattr(couches, "get_avec_reprise", _service(objets, plafond=2))
    with pytest.raises(requests.RequestException):
        couches.lire_couche("X", 0, 0, 1, 1)
