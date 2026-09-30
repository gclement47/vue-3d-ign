"""Mesure des réservoirs, constructions ponctuelles et ouvrages sur des lieux réels.

Exécute le vrai chemin de la scène — constructions_pour_emprise, puis
houppiers_pour_emprise avec et sans leur masque — et celui de la couche des
ouvrages, sur les données de la Géoplateforme. Pour chaque lieu : ce que la
BD TOPO et le LiDAR disent des hauteurs, combien de houppiers et de masses le
masque retire selon la dilatation des réservoirs, et ce que la couche des
ouvrages dessine et explique. C'est la mesure à relancer avant de toucher une
constante de constructions.py ou d'ouvrages.py, ou pour éprouver un nouveau
lieu.

Usage : . .venv/bin/activate && python outils/mesure_constructions.py [lat lon ...]
Sans argument : quatre parcs de stockage (Lavéra, Feyzin, Donges,
Gonfreville), la Cité de Carcassonne et le Pont du Gard.
"""

import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vue3d import constructions
from vue3d.constructions import (COUCHE_PONCTUELLES, COUCHE_RESERVOIRS,
                                 constructions_pour_emprise)
from vue3d.couches import (COUCHE_BATIMENTS, COUCHE_FORET, COUCHE_ROUTES,
                           COUCHE_VEGETATION, lire_couche)
from vue3d.houppiers import houppiers_pour_emprise
from vue3d.mnh import fetch_mnh_grid
from vue3d.ortho import fetch_exg_grid
from vue3d.ouvrages import fetch_ouvrages, ouvrages_pour_emprise
from vue3d.relief import fetch_relief
from vue3d.scene import emprise
from vue3d.toits import TOITS_RESOLUTION_M

LIEUX = {"Lavéra": (43.3900, 5.0230), "Feyzin": (45.6734, 4.8409),
         "Donges": (47.3071, -2.0675), "Gonfreville": (49.4734, 0.2402),
         "Carcassonne": (43.2065, 2.3640), "Pont du Gard": (43.9475, 4.5350)}
DILATATIONS_M = (0.0, 1.0, 2.0, 3.0, 4.0)


def mesurer(nom, lat, lon):
    bbox = emprise(lat, lon)
    lire = lambda couche: lire_couche(couche, *bbox)
    batiments, vegetation, forets = lire(COUCHE_BATIMENTS), lire(COUCHE_VEGETATION), lire(COUCHE_FORET)
    reservoirs, ponctuelles = lire(COUCHE_RESERVOIRS), lire(COUCHE_PONCTUELLES)
    grille = fetch_mnh_grid(*bbox, resolution_m=TOITS_RESOLUTION_M, max_pixels=2048)
    exg = fetch_exg_grid(*bbox, grille["width"], grille["height"])

    def sursol(masque=None):
        bati = {"features": batiments["features"] + (masque["features"] if masque else [])}
        return houppiers_pour_emprise(*bbox, bati, vegetation, forets, grille, exg)

    print(f"\n## {nom} ({lat}, {lon}) — MNH {grille.get('source')}")
    construits, masque = constructions_pour_emprise(*bbox, reservoirs, ponctuelles,
                                                   batiments, grille)
    cuves = construits["reservoirs"]
    sources = collections.Counter(c["source"] for c in cuves)
    print(f"- réservoirs : {len(reservoirs['features'])} lus, {len(cuves)} morceau(x) dessiné(s) ; "
          f"hauteur {dict(sources)}")
    natures = collections.Counter(f["properties"].get("nature") for f in ponctuelles["features"])
    print(f"- constructions ponctuelles lues : {dict(natures)}")
    for p in construits["ponctuelles"]:
        print(f"    {p['detail'] or p['nature']} : {p['h']} m ({p['source']}), "
              f"rayon {p['r'] if p['r'] is not None else 'non mesuré'}")
    sans = sursol()
    print(f"- sans masque : {len(sans['houppiers'])} houppier(s), {len(sans['masses'])} masse(s)")
    if reservoirs["features"]:
        retenue = constructions.MASQUE_RESERVOIR_M
        for d in DILATATIONS_M:
            constructions.MASQUE_RESERVOIR_M = d
            _, m = constructions_pour_emprise(*bbox, reservoirs, ponctuelles, batiments, grille)
            avec = sursol(m)
            print(f"    réservoirs dilatés de {d:.0f} m : {len(avec['houppiers'])} houppier(s), "
                  f"{len(avec['masses'])} masse(s)" + ("   <- retenu" if d == retenue else ""))
        constructions.MASQUE_RESERVOIR_M = retenue
    avec = sursol(masque)

    couche = ouvrages_pour_emprise(*bbox, fetch_ouvrages(*bbox), fetch_relief(*bbox),
                                   avec["masses"], lire(COUCHE_ROUTES))
    if couche is None:
        print("- ouvrages : aucun")
        return
    murs = sorted(m["h"] for m in couche["murs"])
    print(f"- murs : {len(murs)}" + (f", de {murs[0]} à {murs[-1]} m, médiane "
                                     f"{statistics.median(murs):.1f} m" if murs else ""))
    for p in couche["ponts"]:
        print(f"    pont {p['detail'] or ''} : tablier à {p['h']} m au plus"
              + (f", large de {p['largeur_m']} m ({'chaussée' if p['largeur_mesuree'] else 'convention'})"
                 if "ligne" in p else ", en surface"))
    voies = collections.Counter((v["nature"], "au sol" if v["au_sol"] else "sur ouvrage")
                                for v in couche["voies"])
    print(f"- voies ferrées : {dict(voies)} ; terrains de sport : "
          f"{dict(collections.Counter(t['nature'] for t in couche['terrains']))}")
    print(f"- masses expliquées par un mur ou un pont : {len(couche['masses_expliquees'])} "
          f"sur {len(avec['masses'])}")


if __name__ == "__main__":
    args = sys.argv[1:]
    lieux = ({f"{args[k]}, {args[k + 1]}": (float(args[k]), float(args[k + 1]))
              for k in range(0, len(args) - 1, 2)} if args else LIEUX)
    for nom, (lat, lon) in lieux.items():
        mesurer(nom, lat, lon)
