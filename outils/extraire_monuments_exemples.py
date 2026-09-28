"""Extrait OpenStreetMap des monuments des lieux d'exemple du README.

La requête Overpass est l'étape la plus lente et la moins fiable d'une
construction. Mesuré le 2026-09-28 sur les trois instances publiques, deux
essais par instance : de 0,6 s à plus de 100 s pour la même requête, avec des
504 — et une emprise sans aucune partie coûte autant qu'une pleine, le temps
se passant dans la file du serveur. Quand les trois échouent, la scène entière
échoue, alors que toutes les données IGN sont arrivées.

Pour les lieux d'exemple du README, la première impression du projet, la
réponse est donc lue dans un extrait embarqué, vue3d/donnees/
monuments_exemples.json.gz : pas d'attente, et pas d'échec dû à Overpass.

Les lieux sont relus dans les liens du README, source unique : un exemple
ajouté au README se rattrape en relançant cet outil, et un test le vérifie.

Usage : . .venv/bin/activate && python outils/extraire_monuments_exemples.py
Réseau : Overpass en direct, une requête par lieu.
"""

import datetime
import gzip
import json
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from vue3d.monuments import EXTRAIT_EXEMPLES, cle_emprise, fetch_monuments
from vue3d.scene import emprise, point_normalise

LIEN = re.compile(r"localhost:8080/\?lat=(-?[\d.]+)&lon=(-?[\d.]+)")


def lieux_du_readme():
    """Points des liens du README, normalisés comme les clés du cache."""
    texte = (RACINE / "README.md").read_text(encoding="utf-8")
    return sorted({point_normalise(float(a), float(b)) for a, b in LIEN.findall(texte)})


def allege(element):
    """L'élément tel qu'Overpass le rend, sans ce que la vue n'utilise pas
    (identifiants, références de nœuds, boîtes englobantes)."""
    garde = {"type": element["type"], "tags": element.get("tags") or {}}
    if element.get("geometry"):
        garde["geometry"] = element["geometry"]
    if element.get("members"):
        garde["members"] = [{"role": m.get("role"), "geometry": m["geometry"]}
                            for m in element["members"] if m.get("geometry")]
    return garde


def main():
    emprises = {}
    for lat, lon in lieux_du_readme():
        bbox = emprise(lat, lon)
        elements = [allege(e) for e in fetch_monuments(*bbox, extrait=False).get("elements", [])
                    if e.get("type") in ("way", "relation")]
        emprises[cle_emprise(*bbox)] = {"lieu": f"{lat},{lon}", "elements": elements}
        print(f"{lat:.4f}, {lon:.4f} : {len(elements)} partie(s)")
    extrait = {
        "date": datetime.date.today().isoformat(),
        "source": "OpenStreetMap, building:part par l'API Overpass",
        "licence": "ODbL 1.0 — © contributeurs OpenStreetMap",
        "emprises": emprises,
    }
    octets = gzip.compress(json.dumps(extrait, ensure_ascii=False,
                                      separators=(",", ":")).encode(), 9, mtime=0)
    Path(EXTRAIT_EXEMPLES).write_bytes(octets)
    print(f"{len(emprises)} lieux, {len(octets) / 1024:.0f} Ko -> {EXTRAIT_EXEMPLES}")


if __name__ == "__main__":
    main()
