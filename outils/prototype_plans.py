"""Prototype go/no-go : segmentation de plans sur la grille MNH à 0,5 m.

Question à trancher avant tout LoD2 : sur les toits réels de Gordes et de
Strasbourg, un region growing déterministe trouve-t-il des plans propres ?

Méthode, par bâtiment fiable d'un seul tenant à fenêtre lisible :
- surface redressée z = médiane 3x3 du MNH + (sol − sol_bas), celle que la vue
  dessine — c'est elle qu'un LoD2 remplacerait ;
- normales par gradient central (maille 0,5 m), courbure = |laplacien| ;
- graines triées par courbure croissante (départ sur le plat des pans),
  propagation 4-voisins si l'angle des normales et le résidu au plan tiennent
  dans les tolérances ; plan réajusté (moindres carrés) toutes les 20 cellules ;
- plans quasi verticaux rejetés, petites régions rejetées puis absorbées par
  résidu, régions coplanaires adjacentes fusionnées.

Métriques : couverture cumulée des plans validés (critère Niveau 0 : >= 80 %),
résidu médian |z − plan|, nombre de plans, temps. Trois jeux de tolérances
pour la calibration. Aucun code du projet modifié.
Usage : . .venv/bin/activate && python outils/prototype_plans.py
Sorties dans cache/mesures/, ignoré par git.

Résultats du 2026-09-28, config « large » (25°, 0,40 m), toits fiables :
couverture >= 80 % pour 91 % des bâtiments de Gordes (87) et 85 % de
Strasbourg (78) ; résidu médian 0,07 / 0,12 m ; 2 à 10 plans par toit ;
0,2-0,3 s par site entier. Le réglage « manuel » (12°, 0,15 m) s'effondre sur
les toits de plus de 35° de pente — 30 % de couverture contre 95 % sur les
doux : à maille fixe, le bruit de normale croît avec la pente. D'où le
lissage 3×3 des normales et l'amorce de chaque région sur un plan ajusté du
voisinage 3×3, qui ont plus pesé que les seuils eux-mêmes.
"""

import math
import sys
import time
from collections import deque
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
from vue3d.toits import (TOITS_RESOLUTION_M, SURFACE_ECART_RESUME_M,
                         _cellules_locales, _verdure_locale, _combler,
                         _morceaux, _sol_bas, profil_toit, part_verte,
                         qualifier_couvert, cellules_du_toit, ecart_au_resume)

SITES = {"Gordes": (43.9116, 5.2003), "Strasbourg": (48.5819, 7.7510)}
PAS = TOITS_RESOLUTION_M

# Jeux de tolérances : (angle max entre normales, résidu max au plan).
CONFIGS = {"strict (12°, 0,15 m)": (12.0, 0.15), "souple (20°, 0,25 m)": (20.0, 0.25),
           "large (25°, 0,40 m)": (25.0, 0.40)}
# Taille minimale d'un pan : 3 m² (12 cellules), ou 4 % des cellules valides.
PAN_MIN_CELLULES = 12
PAN_MIN_PART = 0.04
# Normale quasi horizontale : façade résiduelle, pas un pan.
PAN_NZ_MIN = 0.2
# Fusion de deux pans adjacents : normales à moins de 5°, plans à moins de
# 0,15 m l'un de l'autre au centre commun.
FUSION_ANGLE_DEG = 5.0
FUSION_DIST_M = 0.15


def _ajuster_plan(xs, ys, zs):
    """Plan z = a·x + b·y + c par moindres carrés ; normale unitaire (-a,-b,1)/n."""
    A = np.c_[xs, ys, np.ones(len(xs))]
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    n = np.array([-coef[0], -coef[1], 1.0])
    return coef, n / np.linalg.norm(n)


def segmenter(z, masque, angle_deg, dist_m):
    """Region growing déterministe sur la grille z (NaN hors masque).

    Returns:
        (labels, plans) : labels >= 0 par cellule assignée (-1 sinon),
        plans = liste de dict(coef, normale, n).
    """
    ny, nx = z.shape
    plein = _combler(np.where(masque, z, np.nan))
    gy, gx = np.gradient(plein, PAS)
    normes = np.sqrt(gx * gx + gy * gy + 1.0)
    nrm = np.stack([-gx / normes, -gy / normes, 1.0 / normes], axis=-1)
    # Normales lissées 3x3 : sur un versant raide, le bruit du MNH par cellule
    # fait battre la normale locale et fragmente la région.
    p = np.pad(nrm, ((1, 1), (1, 1), (0, 0)), mode="edge")
    nrm = sum(p[dj:dj + ny, di:di + nx] for dj in range(3) for di in range(3)) / 9.0
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    lap = np.abs(4 * plein
                 - np.roll(plein, 1, 0) - np.roll(plein, -1, 0)
                 - np.roll(plein, 1, 1) - np.roll(plein, -1, 1))

    ordre = sorted(((lap[j, i], j, i) for j, i in zip(*np.nonzero(masque))))
    labels = np.full((ny, nx), -1, dtype=np.int32)
    plans = []
    cos_min = math.cos(math.radians(angle_deg))
    X = np.arange(nx)[None, :] * PAS * np.ones((ny, 1))
    Y = np.arange(ny)[:, None] * PAS * np.ones((1, nx))

    for _, j0, i0 in ordre:
        if labels[j0, i0] != -1:
            continue
        k = len(plans)
        region = [(j0, i0)]
        labels[j0, i0] = k
        # Amorce sur le voisinage 3x3 : un plan ajusté sur une seule cellule
        # n'existe pas, et la normale d'une cellule seule est trop bruitée.
        vois = [(jj, ii) for jj in range(max(j0 - 1, 0), min(j0 + 2, ny))
                for ii in range(max(i0 - 1, 0), min(i0 + 2, nx)) if masque[jj, ii]]
        if len(vois) >= 3:
            js, is_ = np.array(vois).T
            coef, npl = _ajuster_plan(X[js, is_], Y[js, is_], z[js, is_])
        else:
            npl = nrm[j0, i0]
            coef = np.array([-npl[0] / npl[2], -npl[1] / npl[2],
                             z[j0, i0] + (npl[0] * X[j0, i0] + npl[1] * Y[j0, i0]) / npl[2]])
        file = deque(region)
        depuis_ajust = 0
        while file:
            j, i = file.popleft()
            for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                jj, ii = j + dj, i + di
                if not (0 <= jj < ny and 0 <= ii < nx):
                    continue
                if labels[jj, ii] != -1 or not masque[jj, ii]:
                    continue
                if float(nrm[jj, ii] @ npl) < cos_min:
                    continue
                if abs(z[jj, ii] - (coef[0] * X[jj, ii] + coef[1] * Y[jj, ii] + coef[2])) > dist_m:
                    continue
                labels[jj, ii] = k
                region.append((jj, ii))
                file.append((jj, ii))
                depuis_ajust += 1
                if depuis_ajust >= 20:
                    depuis_ajust = 0
                    js, is_ = np.array(region).T
                    coef, npl = _ajuster_plan(X[js, is_], Y[js, is_], z[js, is_])
        js, is_ = np.array(region).T
        if len(region) >= 3:
            coef, npl = _ajuster_plan(X[js, is_], Y[js, is_], z[js, is_])
        plans.append({"coef": coef, "normale": npl, "n": len(region)})

    # Validation : taille et normale. Les cellules des régions rejetées
    # retombent à -1, puis sont absorbées par le pan voisin le plus proche
    # en résidu (cheminées arasées, bords de pan).
    n_masque = int(masque.sum())
    seuil = max(PAN_MIN_CELLULES, int(PAN_MIN_PART * n_masque))
    garde = {k for k, p in enumerate(plans)
             if p["n"] >= seuil and p["normale"][2] >= PAN_NZ_MIN}
    labels = np.where(np.isin(labels, list(garde)) if garde else False, labels, -1)

    # Fusion des pans coplanaires adjacents (sur-segmentation).
    cos_fusion = math.cos(math.radians(FUSION_ANGLE_DEG))
    change = True
    while change and len(garde) > 1:
        change = False
        for a in sorted(garde):
            for b in sorted(garde):
                if b <= a:
                    continue
                va, vb = labels == a, labels == b
                adj = ((np.roll(va, 1, 0) | np.roll(va, -1, 0)
                        | np.roll(va, 1, 1) | np.roll(va, -1, 1)) & vb).any()
                if not adj:
                    continue
                pa, pb = plans[a], plans[b]
                if float(pa["normale"] @ pb["normale"]) < cos_fusion:
                    continue
                js, is_ = np.nonzero(va | vb)
                za = pa["coef"][0] * X[js, is_] + pa["coef"][1] * Y[js, is_] + pa["coef"][2]
                zb = pb["coef"][0] * X[js, is_] + pb["coef"][1] * Y[js, is_] + pb["coef"][2]
                if float(np.median(np.abs(za - zb))) > FUSION_DIST_M:
                    continue
                labels[vb] = a
                coef, npl = _ajuster_plan(X[js, is_], Y[js, is_],
                                          z[js, is_])
                plans[a] = {"coef": coef, "normale": npl, "n": len(js)}
                garde.discard(b)
                change = True
                break
            if change:
                break

    # Absorption des cellules orphelines par le pan adjacent le plus proche.
    for _ in range(8):
        js, is_ = np.nonzero(masque & (labels == -1))
        if not len(js):
            break
        bouge = False
        for j, i in zip(js, is_):
            mieux, res_min = -1, 2 * dist_m
            for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                jj, ii = j + dj, i + di
                if 0 <= jj < ny and 0 <= ii < nx and labels[jj, ii] >= 0:
                    c = plans[labels[jj, ii]]["coef"]
                    r = abs(z[j, i] - (c[0] * X[j, i] + c[1] * Y[j, i] + c[2]))
                    if r < res_min:
                        mieux, res_min = labels[jj, ii], r
            if mieux >= 0:
                labels[j, i] = mieux
                bouge = True
        if not bouge:
            break

    return labels, plans, X, Y


def mesurer_site(nom, lat, lon):
    west, south, east, north = emprise(lat, lon)
    print(f"\n=== {nom}")
    batiments = lire_couche(COUCHE_BATIMENTS, west, south, east, north)
    grille = fetch_mnh_grid(west, south, east, north,
                            resolution_m=TOITS_RESOLUTION_M, max_pixels=2048)
    exg = fetch_exg_grid(west, south, east, north, grille["width"], grille["height"])
    sol = fetch_sol_grid(west, south, east, north, grille["width"], grille["height"],
                         grille["source"])
    lon0, lat0 = (west + east) / 2, (south + north) / 2
    m_lon = 111320 * math.cos(math.radians(lat0))
    cellules = _cellules_locales(grille, lon0, lat0)
    verdure = _verdure_locale(exg, grille["bbox"], lon0, lat0)
    hauteurs = np.asarray(grille["values"], dtype=np.float32).reshape(
        grille["height"], grille["width"])
    xs, ys = _verdure_locale(hauteurs, grille["bbox"], lon0, lat0)[1:]
    sol_arr = np.asarray(sol, dtype=np.float64).reshape(hauteurs.shape)

    lignes = {c: [] for c in CONFIGS}
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
        if (not profil["fiable"] or profil["mode_bas"] or profil["hauteur_inconnue"]
                or len(_morceaux(geom)) != 1):
            continue
        poly_seul = _morceaux(poly)[0]
        fenetre = cellules_du_toit(poly_seul, hauteurs, xs, ys, profil["gouttiere"], exg)
        if fenetre is None:
            continue
        ecart = ecart_au_resume(profil, poly_seul, fenetre)
        i0, i1, j0, j1 = fenetre["i0"], fenetre["i1"], fenetre["j0"], fenetre["j1"]
        bas = _sol_bas(sol_arr, geom, xs, ys, lon0, lat0, m_lon)
        if bas is None:
            continue
        masque = fenetre["valides"] & np.isfinite(fenetre["lisse"])
        z = np.where(masque,
                     fenetre["lisse"] + np.nan_to_num(sol_arr[j0:j1, i0:i1] - bas),
                     np.nan)
        # Pente p90 du toit, pour corréler l'échec éventuel à la raideur.
        gy_, gx_ = np.gradient(_combler(np.where(masque, z, np.nan)), PAS)
        pente90 = float(np.percentile(
            np.degrees(np.arctan(np.hypot(gx_, gy_)))[masque], 90))
        for cfg, (ang, dist) in CONFIGS.items():
            debut = time.perf_counter()
            labels, plans, X, Y = segmenter(z, masque, ang, dist)
            duree = time.perf_counter() - debut
            assigne = labels >= 0
            couverture = float(assigne.sum()) / float(masque.sum())
            if assigne.any():
                js, is_ = np.nonzero(assigne)
                zp = np.array([plans[labels[j, i]]["coef"] @ (X[j, i], Y[j, i], 1.0)
                               for j, i in zip(js, is_)])
                residu = float(np.median(np.abs(z[js, is_] - zp)))
            else:
                residu = float("nan")
            lignes[cfg].append({
                "cleabs": cleabs, "n": int(masque.sum()),
                "candidat": ecart > SURFACE_ECART_RESUME_M,
                "ecart_resume": round(ecart, 2),
                "plans": len({int(k) for k in np.unique(labels) if k >= 0}),
                "couverture": couverture, "residu": residu, "ms": duree * 1000,
                "pente90": pente90,
            })
    return lignes


def pct(vals, t):
    return float(np.percentile(np.asarray(vals, dtype=np.float64), t))


def resumer(nom, lignes, sortie):
    def p(s=""):
        print(s)
        sortie.append(s)

    for cfg, rows in lignes.items():
        if not rows:
            continue
        for etiquette, sel in (("tous toits fiables", rows),
                               ("candidats surface (écart > 0,7 m)",
                                [r for r in rows if r["candidat"]])):
            if not sel:
                continue
            couv = [r["couverture"] for r in sel]
            res = [r["residu"] for r in sel if np.isfinite(r["residu"])]
            npl = [r["plans"] for r in sel]
            ok80 = sum(1 for c in couv if c >= 0.8)
            p(f"\n### {nom} — {cfg} — {etiquette} ({len(sel)} bâtiments)")
            p(f"- couverture cumulée : p50 {pct(couv,50):.0%}, p10 {pct(couv,10):.0%} ; "
              f">= 80 % : {ok80}/{len(sel)} ({100*ok80/len(sel):.0f} %)")
            p(f"- résidu médian au plan : p50 {pct(res,50):.2f} m, p90 {pct(res,90):.2f} m")
            p(f"- nombre de plans : p50 {pct(npl,50):.0f}, max {max(npl)}")
            p(f"- temps : total {sum(r['ms'] for r in sel)/1000:.1f} s, "
              f"pire bâtiment {max(r['ms'] for r in sel):.0f} ms")
            doux = [r["couverture"] for r in sel if r["pente90"] <= 35]
            raide = [r["couverture"] for r in sel if r["pente90"] > 35]
            if doux and raide:
                p(f"- couverture selon la pente p90 : <= 35° {pct(doux,50):.0%} "
                  f"({len(doux)} bât.), > 35° {pct(raide,50):.0%} ({len(raide)} bât.)")


PIRES_GORDES = {"BATIMENT0000000296689006", "BATIMENT0000000296689042",
                "BATIMENT0000000296689169", "BATIMENT0000000296689045"}


def main():
    sortie = ["# Prototype : segmentation de plans sur la grille MNH 0,5 m"]
    for nom, (lat, lon) in SITES.items():
        lignes = mesurer_site(nom, lat, lon)
        resumer(nom, lignes, sortie)
        if nom == "Gordes":
            sortie.append("\n### Pires cascades de la mesure précédente")
            print("\n### Pires cascades de la mesure précédente")
            for cfg, rows in lignes.items():
                for r in rows:
                    if r["cleabs"] in PIRES_GORDES:
                        s = (f"- {cfg} {r['cleabs']} : {r['plans']} plans, "
                             f"couverture {r['couverture']:.0%}, résidu {r['residu']:.2f} m")
                        print(s)
                        sortie.append(s)
    DOSSIER.mkdir(parents=True, exist_ok=True)
    chemin = DOSSIER / "resultats-plans.md"
    with open(chemin, "w") as f:
        f.write("\n".join(sortie) + "\n")
    print(f"\nRésultats écrits dans {chemin}")


if __name__ == "__main__":
    main()
