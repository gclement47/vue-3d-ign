"""Mesure de la couche des véhicules sur des lieux réels.

Exécute le vrai chemin de la couche — detecter, puis vehicules_pour_emprise —
sur l'orthophoto de la Géoplateforme. Pour chaque lieu et chaque détecteur :
le temps de calcul, le nombre de véhicules selon le seuil et le recouvrement
toléré entre deux boîtes, ce que les filtres écartent, le gabarit des boîtes,
et l'accord entre les deux détecteurs. C'est la mesure à relancer avant de
toucher une constante de vue3d/vehicules.py, ou pour éprouver un nouveau lieu.

Il n'y a pas de vérité terrain annotée : les comptes se jugent sur l'image
annotée (--images), à l'œil.

Usage :
    . .venv/bin/activate && pip install -r requirements-vehicules.txt
    python outils/mesure_vehicules.py modeles/ [--images dossier] [--balayage] [lat lon ...]

`modeles/` contient les réseaux exportés par outils/exporter_vehicules.py.
Sans coordonnées : le village de Gordes et la ville basse de Carcassonne.
--balayage ajoute celui des tailles de tuile et de leur recouvrement :
plusieurs minutes par lieu.
"""

import io
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vue3d import vehicules
from vue3d.couches import COUCHE_BATIMENTS, lire_couche
from vue3d.eau import COUCHE_COURS_EAU, COUCHE_SURFACES_EAU, eau_pour_emprise
from vue3d.ortho import fetch_ortho_jpeg
from vue3d.scene import emprise

LIEUX = {"Gordes": (43.9116, 5.2003), "Carcassonne": (43.2075, 2.3680)}
SEUILS = {"rtmdet": (0.1, 0.15, 0.2, 0.3), "yolo": (0.15, 0.25, 0.35, 0.5)}
TUILES = {"rtmdet": (1024, 768, 640, 512, 400, 320), "yolo": (320, 256, 200, 160, 128)}
RECOUVREMENTS = {"rtmdet": (65, 128, 256, 384), "yolo": (32, 80)}
IOUS = (0.1, 0.3, 0.5)


def avec(nom, **valeurs):
    """Les détecteurs, celui-là modifié."""
    return {**vehicules.DETECTEURS, nom: {**vehicules.DETECTEURS[nom], **valeurs}}


def mesurer(nom_lieu, lat, lon, sessions, images, balayage):
    from PIL import Image, ImageDraw
    bbox = emprise(lat, lon)
    contenu, _, _ = fetch_ortho_jpeg(*bbox, resolution_m=vehicules.RESOLUTION_M)
    image = Image.open(io.BytesIO(contenu)).convert("RGB")
    rgb = np.asarray(image)
    batiments = lire_couche(COUCHE_BATIMENTS, *bbox)
    eau = eau_pour_emprise(*bbox, lire_couche(COUCHE_SURFACES_EAU, *bbox),
                           lire_couche(COUCHE_COURS_EAU, *bbox))
    print(f"\n## {nom_lieu} ({lat}, {lon}) — orthophoto {rgb.shape[1]} × {rgb.shape[0]} px")

    def couche(boites, mode):
        brut = {"largeur": rgb.shape[1], "hauteur": rgb.shape[0], "boites": boites}
        c = vehicules.vehicules_pour_emprise(*bbox, brut, mode, batiments, eau)
        return c["vehicules"] if c else []

    par_detecteur = {}
    for nom, session in sessions.items():
        seul = {nom: session}
        t0 = time.time()
        boites = vehicules.detecter(rgb, seul)
        duree = time.time() - t0
        gardes = couche(boites, nom)
        par_detecteur[nom] = boites
        print(f"- {nom} : {len(boites)} boîte(s) en {duree:.1f} s, {len(gardes)} véhicule(s) gardé(s)")
        if gardes:
            lo = sorted(v[2] for v in gardes)
            la = sorted(v[3] for v in gardes)
            dec = lambda t, q: t[min(int(q * len(t)), len(t) - 1)]      # noqa: E731
            print(f"  gabarit : longueur {lo[0]} / {statistics.median(lo)} / {dec(lo, 0.95)} / {lo[-1]} m "
                  f"(min, médiane, 95 %, max), largeur {la[0]} / {statistics.median(la)} / "
                  f"{dec(la, 0.95)} / {la[-1]} m ; "
                  f"{sum(b[6] for b in boites)} boîte(s) de classe « gros véhicule »")
        brutes = vehicules.detecter(rgb, seul, avec(nom, seuil=min(SEUILS[nom])))
        print("  selon le seuil : " + ", ".join(
            f"{s} → {len(couche([b for b in brutes if b[5] >= s], nom))}" for s in SEUILS[nom]))
        iou_d, centres_d = vehicules.DOUBLON_IOU, vehicules.DOUBLON_CENTRES_M
        comptes = []
        for iou in IOUS:
            vehicules.DOUBLON_IOU = iou
            comptes.append(f"{iou} → {len(vehicules.detecter(rgb, seul))}")
        vehicules.DOUBLON_IOU = iou_d
        print("  boîtes selon le recouvrement toléré : " + ", ".join(comptes))
        if balayage:
            for libelle, cle, valeurs in (("la tuile", "tuile_px", TUILES[nom]),
                                          ("le recouvrement", "recouvrement_px", RECOUVREMENTS[nom])):
                lignes = []
                for v in valeurs:
                    t0 = time.time()
                    n = len(couche(vehicules.detecter(rgb, seul, avec(nom, **{cle: v})), nom))
                    lignes.append(f"{v} px → {n} ({time.time() - t0:.0f} s)")
                print(f"  selon {libelle} : " + ", ".join(lignes))
        if images:
            annotee = image.copy()
            dessin = ImageDraw.Draw(annotee)
            for b in boites:
                dessin.polygon(vehicules._coins(*b[:5]), outline=(255, 0, 255))
            chemin = Path(images) / f"{nom_lieu.lower()}-{nom}.jpg"
            annotee.save(chemin, quality=88)
            print(f"  image : {chemin}")

    if len(par_detecteur) == 2:
        (na, a), (nb, b) = par_detecteur.items()
        rayon2 = (vehicules.DOUBLON_ENTRE_DETECTEURS_M / vehicules.RESOLUTION_M) ** 2
        proche = lambda v, autres: any(                                   # noqa: E731
            (v[0] - o[0]) ** 2 + (v[1] - o[1]) ** 2 < rayon2 for o in autres)
        communs = sum(proche(v, b) for v in a)
        t0 = time.time()
        union = vehicules.detecter(rgb, sessions)
        print(f"- accord : {communs} commun(s), {len(a) - communs} vu(s) de {na} seul, "
              f"{len(b) - sum(proche(v, a) for v in b)} de {nb} seul ; "
              f"union {len(union)} boîte(s), {len(couche(union, 'tous'))} gardée(s), "
              f"en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    args = sys.argv[1:]
    images = None
    if "--images" in args:
        i = args.index("--images")
        images = args[i + 1]
        Path(images).mkdir(parents=True, exist_ok=True)
        del args[i:i + 2]
    balayage = "--balayage" in args
    args = [a for a in args if a != "--balayage"]
    if not args:
        sys.exit(__doc__)
    sessions = vehicules.charger("tous", args[0])
    coords = [float(a) for a in args[1:]]
    lieux = ({f"{la}, {lo}": (la, lo) for la, lo in zip(coords[::2], coords[1::2])}
             if coords else LIEUX)
    for nom_lieu, (lat, lon) in lieux.items():
        mesurer(nom_lieu, lat, lon, sessions, images, balayage)
