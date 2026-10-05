"""Mesure du nuage LiDAR HD sur des lieux réels : lecture, ouvrages ajourés, volume.

Exécute le vrai chemin de la couche (vue3d/nuage.py) sur la Géoplateforme.
Pour chaque lieu, sur l'emprise par défaut :

- ce que coûte la lecture des dalles COPC : requêtes, octets, secondes, selon
  la taille des blocs lus (--blocs) ;
- pour chaque bâtiment de la scène, la part de son emprise érodée où le
  LiDAR voit le sol (part_sol_vu), et les bâtiments au-delà de 5 % ;
- les points du bâti transmis, et ceux des ouvrages ajourés.

C'est la mesure à relancer avant de toucher une constante de nuage.py.

Usage : python outils/mesure_nuage.py [--blocs 65536,262144,1048576] [--zone 1000] [lat lon ...]
Sans lieu : la tour Eiffel, Notre-Dame de Paris, le Grand Palais (verrière),
la gare de l'Est, la cathédrale de Strasbourg et le village de Gordes.
"""

import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vue3d import nuage
from vue3d.batiments import decouper_batiments
from vue3d.couches import COUCHE_BATIMENTS, lire_couche
from vue3d.relief import fetch_relief
from vue3d.scene import emprise

LIEUX = {"Tour Eiffel": (48.8584, 2.2945), "Notre-Dame": (48.8530, 2.3499),
         "Grand Palais": (48.8661, 2.3125), "Gare de l'Est": (48.8768, 2.3590),
         "Strasbourg": (48.5819, 7.7510), "Gordes": (43.9116, 5.2003)}


def mesurer(nom, lat, lon, blocs, zone=None):
    bbox = emprise(lat, lon, zone)
    print(f"\n## {nom} ({lat}, {lon})")
    brut = None
    for bloc in blocs:
        nuage.BLOC_OCTETS = bloc
        lus = []
        lire = lambda url, debut, fin: lus.append(fin - debut + 1) or nuage.lire_plage(url, debut, fin)
        t0 = time.time()
        brut = nuage.fetch_nuage(*bbox, lire=lire)
        if brut is None:
            print("- hors couverture LiDAR HD")
            return
        print(f"- blocs de {bloc // 1024} Ko : {len(lus)} requêtes, {sum(lus) / 1e6:.1f} Mo demandés, "
              f"{time.time() - t0:.1f} s ; {len(brut['x'])} points du bâti")
    batiments = decouper_batiments(lire_couche(COUCHE_BATIMENTS, *bbox), *bbox)
    parts = []
    for f in batiments["features"]:
        p = f["properties"]
        for poly in nuage._polygones_l93(f["geometry"]):
            part, bati, n = nuage.parts_vues(poly, brut["grilles"])
            if part is not None:
                parts.append((part, n, p, bati))
    bornes = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01)
    compte = collections.Counter(next(f"{a:.0%}-{b:.0%}" for a, b in zip(bornes, bornes[1:]) if a <= part < b)
                                 for part, _, _, _ in parts)
    print(f"- {len(parts)} emprises jugées ; part du sol vu : "
          + ", ".join(f"{k} : {compte[k]}" for k in (f"{a:.0%}-{b:.0%}" for a, b in zip(bornes, bornes[1:]))))
    for part, n, p, bati in sorted(parts, key=lambda t: -t[0]):
        if part < 0.05:
            break
        print(f"    sol vu {part:5.0%}, structure vue {bati:5.0%} sur {n:5d} m²  {p.get('cleabs')}  "
              f"{p.get('nature')} / {p.get('usage_1')}, {p.get('hauteur')} m, créé {str(p.get('date_creation'))[:10]}")
    print(f"    ajourés : {nuage.ajoures(batiments, brut['grilles'])[0]}")
    couche = nuage.nuage_pour_emprise(*bbox, brut, fetch_relief(*bbox), batiments, [])
    if couche:
        print(f"- couche : {couche['n']} points, dont {couche['n_ajoures']} d'ouvrages ajourés "
              f"({len(couche['ajoures'])} bâtiments) ; "
              f"{sum(len(couche[k]) for k in ('lon', 'lat', 'h', 'classe')) / 1e6:.1f} Mo en base64")


if __name__ == "__main__":
    args = sys.argv[1:]
    blocs, zone = [nuage.BLOC_OCTETS], None
    while args[:1] in (["--blocs"], ["--zone"]):
        if args[0] == "--blocs":
            blocs = [int(b) for b in args[1].split(",")]
        else:
            zone = int(args[1])
        args = args[2:]
    lieux = ({f"{args[k]}, {args[k + 1]}": (float(args[k]), float(args[k + 1]))
              for k in range(0, len(args) - 1, 2)} if args else LIEUX)
    for nom, (lat, lon) in lieux.items():
        mesurer(nom, lat, lon, blocs, zone)
