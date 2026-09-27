"""Mesure : part du terrain dans les défauts du mode « toits mesurés ».

Hypothèse à trancher : les cascades déchiquetées vues à Gordes viennent du
redressement `sol − sol_bas` ajouté par surface_toit, pas du grain du MNH.

Trois mesures, sur Gordes (terrain fort) et Strasbourg (témoin plat) :
1. par bâtiment fiable : dénivelé du toit vs étendue du terrain sous l'emprise ;
2. par bâtiment à surface : marche max entre cellules voisines dans le MNH
   seul, le terrain seul, et la surface redressée — qui fabrique les falaises ;
3. essai à blanc : redressement par plan ajusté (moindres carrés) sur le
   terrain, au lieu de la grille brute ; marches et amplitudes avant/après.

Script de mesure : importe vue3d en lecture, ne modifie rien au projet.
Réseau : Géoplateforme en direct (les tests pytest, eux, restent sans réseau).
Usage : . .venv/bin/activate && python outils/mesure_redressement.py
Sorties dans cache/mesures/, ignoré par git.

Résultats du 2026-09-28 — hypothèse RÉFUTÉE. Marche max entre cellules
voisines des surfaces embarquées (médiane Gordes / Strasbourg) : MNH seul
2,4 / 3,0 m, terrain seul 0,7 / 0,2 m, surface redressée 2,4 / 3,0 m — et le
plan ajusté ne change rien. Les cascades sont dans le MNH lui-même (marches
réelles de 7 à 10 m, identiques sur terrain plat) ; le terrain, lui, fait les
tours démesurées : il dépasse le dénivelé du toit sur 86 % des bâtiments de
Gordes (11,3 m au pire) contre 6 % à Strasbourg. Suite donnée : segmentation
de plans (prototype_plans.py) plutôt que correctif du redressement.
"""

import math
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
DOSSIER = RACINE / "cache" / "mesures"

import numpy as np
import shapely
from shapely.geometry import shape
from shapely.ops import transform

from vue3d.scene import emprise
from vue3d.couches import lire_couche, COUCHE_BATIMENTS
from vue3d.mnh import fetch_mnh_grid, fetch_sol_grid
from vue3d.ortho import fetch_exg_grid
from vue3d import toits
from vue3d.toits import (TOITS_RESOLUTION_M, SURFACE_ECART_RESUME_M,
                         _cellules_locales, _verdure_locale, _combler,
                         _morceaux, _sol_bas, profil_toit, part_verte,
                         qualifier_couvert, cellules_du_toit, ecart_au_resume)

SITES = {
    "Gordes": (43.9116, 5.2003),
    "Strasbourg": (48.5819, 7.7510),
}


def percentile(vals, t):
    return float(np.percentile(np.asarray(vals, dtype=np.float64), t)) if vals else float("nan")


def marche_max(g, masque):
    """Plus grande marche entre deux cellules voisines toutes deux du masque."""
    pire = 0.0
    dh = np.abs(np.diff(g, axis=1))
    mh = masque[:, 1:] & masque[:, :-1]
    if mh.any():
        pire = max(pire, float(np.nanmax(dh[mh])))
    dv = np.abs(np.diff(g, axis=0))
    mv = masque[1:, :] & masque[:-1, :]
    if mv.any():
        pire = max(pire, float(np.nanmax(dv[mv])))
    return pire


def amplitude(g, masque):
    v = g[masque]
    return float(np.nanmax(v) - np.nanmin(v)) if v.size else float("nan")


def etendue_sol_sous(poly, sol_arr, xs, ys):
    """max − min du terrain aux cellules dont le centre est dans l'emprise."""
    minx, miny, maxx, maxy = poly.bounds
    i0 = max(int(np.searchsorted(xs, minx)) - 1, 0)
    i1 = min(int(np.searchsorted(xs, maxx, side="right")) + 1, len(xs))
    j0 = max(int(np.searchsorted(-ys, -maxy)) - 1, 0)
    j1 = min(int(np.searchsorted(-ys, -miny, side="right")) + 1, len(ys))
    if i0 >= i1 or j0 >= j1:
        return None
    X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1])
    dedans = shapely.contains_xy(poly, X, Y)
    v = sol_arr[j0:j1, i0:i1][dedans]
    v = v[np.isfinite(v)]
    if v.size < 3:
        return None
    return float(v.max() - v.min())


def plan_sol(sol_win, X, Y, masque):
    """Plan a·x + b·y + c ajusté par moindres carrés sur le terrain du masque."""
    m = masque & np.isfinite(sol_win)
    if m.sum() < 3:
        return None
    A = np.c_[X[m], Y[m], np.ones(int(m.sum()))]
    coef, *_ = np.linalg.lstsq(A, sol_win[m], rcond=None)
    return coef


def mesurer_site(nom, lat, lon):
    west, south, east, north = emprise(lat, lon)
    print(f"\n=== {nom} ({lat}, {lon}) — emprise {west:.4f},{south:.4f},{east:.4f},{north:.4f}")
    batiments = lire_couche(COUCHE_BATIMENTS, west, south, east, north)
    grille = fetch_mnh_grid(west, south, east, north,
                            resolution_m=TOITS_RESOLUTION_M, max_pixels=2048)
    if not grille.get("couvert"):
        print("  hors couverture, site ignoré")
        return None
    exg = fetch_exg_grid(west, south, east, north, grille["width"], grille["height"])
    sol = fetch_sol_grid(west, south, east, north, grille["width"], grille["height"],
                         grille["source"])
    print(f"  grille {grille['width']}x{grille['height']} source={grille['source']}, "
          f"{len(batiments.get('features', []))} bâtiments")

    # Même chemin de calcul que toits_pour_emprise.
    lon0, lat0 = (west + east) / 2, (south + north) / 2
    m_lon = 111320 * math.cos(math.radians(lat0))
    cellules = _cellules_locales(grille, lon0, lat0)
    verdure = _verdure_locale(exg, grille["bbox"], lon0, lat0)
    hauteurs = np.asarray(grille["values"], dtype=np.float32).reshape(
        grille["height"], grille["width"])
    xs, ys = _verdure_locale(hauteurs, grille["bbox"], lon0, lat0)[1:]
    sol_arr = np.asarray(sol, dtype=np.float64).reshape(hauteurs.shape)
    demi_diag = 0.5 * math.hypot(xs[1] - xs[0], ys[0] - ys[1]) + 1e-6

    fiables, candidats = [], []
    for f in batiments.get("features", []):
        cleabs = (f.get("properties") or {}).get("cleabs")
        if not cleabs or not f.get("geometry"):
            continue
        try:
            geom = shape(f["geometry"])
        except Exception:
            continue
        poly = transform(lambda x, y, z=None: ((x - lon0) * m_lon, (y - lat0) * 111320), geom)
        profil = profil_toit(cellules, poly)
        if not profil:
            continue
        profil = qualifier_couvert(profil, part_verte(poly, verdure))
        if not profil["fiable"]:
            continue

        # --- Mesure 1 : dénivelé du toit vs étendue du terrain sous l'emprise.
        etendue = etendue_sol_sous(poly, sol_arr, xs, ys)
        if etendue is None:
            continue
        denivele = profil["denivele"]
        fiables.append({"cleabs": cleabs, "denivele": denivele, "etendue_sol": etendue})

        # --- Mesures 2 et 3 : mêmes conditions que l'embarquement d'une surface.
        if profil["mode_bas"] or profil["hauteur_inconnue"] or len(_morceaux(geom)) != 1:
            continue
        poly_seul = _morceaux(poly)[0]
        fenetre = cellules_du_toit(poly_seul, hauteurs, xs, ys, profil["gouttiere"], exg)
        if fenetre is None:
            continue
        ecart = ecart_au_resume(profil, poly_seul, fenetre)
        if ecart <= SURFACE_ECART_RESUME_M:
            continue          # pas de surface embarquée : le résumé suffit
        i0, i1, j0, j1 = fenetre["i0"], fenetre["i1"], fenetre["j0"], fenetre["j1"]
        X, Y = fenetre["X"], fenetre["Y"]
        z_mnh = _combler(fenetre["lisse"])
        sol_win = sol_arr[j0:j1, i0:i1]
        bas = _sol_bas(sol_arr, geom, xs, ys, lon0, lat0, m_lon)
        if bas is None:
            continue
        rel = np.nan_to_num(sol_win - bas)
        z_red = z_mnh + rel                      # la surface telle qu'embarquée
        # masque « utile » de surface_toit : le carré de la cellule touche l'emprise
        utile = shapely.distance(shapely.points(X.ravel(), Y.ravel()),
                                 poly_seul).reshape(X.shape) <= demi_diag

        # --- Essai à blanc : redressement par plan ajusté sur le terrain.
        coef = plan_sol(sol_win, X, Y, utile)
        if coef is None:
            continue
        plan = coef[0] * X + coef[1] * Y + coef[2]
        plan_bas = min(coef[0] * (lo - lon0) * m_lon + coef[1] * (la - lat0) * 111320 + coef[2]
                       for a in _morceaux(geom) for lo, la, *_ in a.exterior.coords)
        z_plan = z_mnh + (plan - plan_bas)

        candidats.append({
            "cleabs": cleabs, "denivele": denivele, "etendue_sol": etendue,
            "ecart_resume": round(ecart, 2),
            "marche_mnh": marche_max(z_mnh, utile),
            "marche_sol": marche_max(rel, utile),
            "marche_red": marche_max(z_red, utile),
            "marche_plan": marche_max(z_plan, utile),
            "ampli_red": amplitude(z_red, utile),
            "ampli_plan": amplitude(z_plan, utile),
            "ecart_plan_red": float(np.median(np.abs(z_plan - z_red)[utile])),
        })

    return {"nom": nom, "fiables": fiables, "candidats": candidats}


def resumer(r, sortie):
    def p(ligne=""):
        print(ligne)
        sortie.append(ligne)

    fiables, candidats = r["fiables"], r["candidats"]
    p(f"\n## {r['nom']} — {len(fiables)} toits fiables, {len(candidats)} surfaces embarquées")

    dens = [b["denivele"] for b in fiables]
    sols = [b["etendue_sol"] for b in fiables]
    ratio = [b["etendue_sol"] / max(b["denivele"], 0.1) for b in fiables]
    domine = sum(1 for b in fiables if b["etendue_sol"] > b["denivele"])
    p("\n### Mesure 1 — terrain sous l'emprise (toits fiables)")
    p("| | p50 | p90 | max |")
    p("|---|---|---|---|")
    p(f"| dénivelé du toit (m) | {percentile(dens,50):.1f} | {percentile(dens,90):.1f} | {max(dens):.1f} |")
    p(f"| étendue du terrain (m) | {percentile(sols,50):.1f} | {percentile(sols,90):.1f} | {max(sols):.1f} |")
    p(f"| ratio terrain/toit | {percentile(ratio,50):.1f} | {percentile(ratio,90):.1f} | {max(ratio):.1f} |")
    p(f"\nTerrain > dénivelé du toit : {domine}/{len(fiables)} bâtiments "
      f"({100*domine/max(len(fiables),1):.0f} %)")

    if not candidats:
        p("\n(aucune surface embarquée)")
        return
    p("\n### Mesure 2 et 3 — marches max entre cellules voisines (surfaces embarquées)")
    for cle in ("marche_mnh", "marche_sol", "marche_red", "marche_plan"):
        v = [c[cle] for c in candidats]
        p(f"- {cle} : p50 {percentile(v,50):.2f} m, p90 {percentile(v,90):.2f} m, max {max(v):.2f} m")
    sol_dom = sum(1 for c in candidats if c["marche_sol"] > c["marche_mnh"])
    p(f"- marche du terrain > marche du MNH : {sol_dom}/{len(candidats)} surfaces")
    v = [c["ecart_plan_red"] for c in candidats]
    p(f"- écart médian plan vs actuel : p50 {percentile(v,50):.2f} m, max {max(v):.2f} m")

    p("\n### Pires surfaces (par marche de la surface redressée)")
    p("| cleabs | dénivelé | étendue sol | écart résumé | marche MNH | marche sol | marche actuelle | marche plan | ampli actuelle | ampli plan |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(candidats, key=lambda c: -c["marche_red"])[:4]:
        p(f"| {c['cleabs']} | {c['denivele']:.1f} | {c['etendue_sol']:.1f} | {c['ecart_resume']:.2f} "
          f"| {c['marche_mnh']:.2f} | {c['marche_sol']:.2f} | {c['marche_red']:.2f} "
          f"| {c['marche_plan']:.2f} | {c['ampli_red']:.1f} | {c['ampli_plan']:.1f} |")


def main():
    sortie = ["# Mesure : redressement terrain des surfaces de toit",
              f"Sites : {', '.join(SITES)} — grille MNH 0,5 m, chemin de calcul de toits_pour_emprise."]
    for nom, (lat, lon) in SITES.items():
        r = mesurer_site(nom, lat, lon)
        if r:
            resumer(r, sortie)
    DOSSIER.mkdir(parents=True, exist_ok=True)
    chemin = DOSSIER / "resultats-redressement.md"
    with open(chemin, "w") as f:
        f.write("\n".join(sortie) + "\n")
    print(f"\nRésultats écrits dans {chemin}")


if __name__ == "__main__":
    main()
