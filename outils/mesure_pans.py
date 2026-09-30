"""Mesure des toits en pans (vue3d/pans.py) sur des lieux réels.

Exécute le vrai chemin de la scène — toits_pour_emprise, sur les grilles
MNH, ExG et terrain de la Géoplateforme — et compte, parmi les toits dont le
résumé s'écarte du LiDAR (ceux qui reçoivent pans ou surface) : combien
reçoivent des pans, combien retombent sur la surface, le poids transmis et
le temps. C'est la mesure à relancer avant de toucher une constante de
pans.py, ou pour éprouver un nouveau lieu.

Usage : . .venv/bin/activate && python outils/mesure_pans.py [lat lon ...]
Sans argument : Gordes et Strasbourg.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vue3d.batiments import decouper_batiments
from vue3d.couches import COUCHE_BATIMENTS, lire_couche
from vue3d.mnh import fetch_mnh_grid, fetch_sol_grid
from vue3d.ortho import fetch_exg_grid
from vue3d.scene import emprise
from vue3d.toits import TOITS_RESOLUTION_M, toits_pour_emprise

LIEUX = {"Gordes": (43.9116, 5.2003), "Strasbourg": (48.5819, 7.7510)}


def mesurer(nom, lat, lon):
    west, south, east, north = emprise(lat, lon)
    bats = decouper_batiments(lire_couche(COUCHE_BATIMENTS, west, south, east, north),
                              west, south, east, north)
    grille = fetch_mnh_grid(west, south, east, north,
                            resolution_m=TOITS_RESOLUTION_M, max_pixels=2048)
    exg = fetch_exg_grid(west, south, east, north, grille["width"], grille["height"])
    sol = fetch_sol_grid(west, south, east, north, grille["width"], grille["height"],
                         grille["source"])
    debut = time.perf_counter()
    toits = toits_pour_emprise(west, south, east, north, bats, grille, exg, sol)["toits"]
    duree = time.perf_counter() - debut
    pans = [p["pans"] for p in toits.values() if "pans" in p]
    surfaces = [p["surface"] for p in toits.values() if "surface" in p]
    candidats = len(pans) + len(surfaces)
    poids = lambda liste: len(json.dumps(liste, separators=(",", ":"))) / 1024
    print(f"\n## {nom} ({lat}, {lon})")
    print(f"- toits à forme mesurée : {candidats} ; en pans : {len(pans)} "
          f"({100 * len(pans) / max(candidats, 1):.0f} %), surface en repli : {len(surfaces)}")
    if pans:
        n = sorted(p["n_pans"] for p in pans)
        e = sorted(p["ecart_m"] for p in pans)
        t = sorted(len(p["triangles"]) // 3 for p in pans)
        print(f"- pans par toit : médiane {n[len(n) // 2]}, max {n[-1]} ; écart médian au "
              f"LiDAR : médiane {e[len(e) // 2]:.2f} m, max {e[-1]:.2f} m")
        print(f"- triangles par toit : médiane {t[len(t) // 2]}, max {t[-1]}")
    print(f"- poids JSON : pans {poids(pans):.0f} Ko, surfaces {poids(surfaces):.0f} Ko")
    print(f"- toits_pour_emprise : {duree:.1f} s")


if __name__ == "__main__":
    args = sys.argv[1:]
    lieux = ({f"{args[k]}, {args[k + 1]}": (float(args[k]), float(args[k + 1]))
              for k in range(0, len(args) - 1, 2)} if args else LIEUX)
    for nom, (lat, lon) in lieux.items():
        mesurer(nom, lat, lon)
