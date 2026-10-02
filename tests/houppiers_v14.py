"""Segmentation des houppiers telle qu'elle était en scène v14 (bc716ba) :
l'oracle des tests d'équivalence de test_houppiers.py.

vue3d/houppiers.py a été réécrit pour aller plus vite — descente par front,
sommets par couronnes, masques testés aux seules cellules qui comptent,
arrondis en tableau — sans changer un octet de la scène. Ce module garde le
code d'avant, plus lent mais simple à relire, pour que les tests le
vérifient sur des grilles tirées au hasard. Il n'est importé que par les
tests ; les commentaires et la justification des constantes sont dans
vue3d/houppiers.py.
"""

import math

import numpy as np
import shapely
from shapely.geometry import shape

from vue3d.houppiers import (ALLONGEMENT_MAX, ETENDUE_FACTEUR, HOUPPIERS_MIN_CELLULES,
                             NATURES_FORET, PROFIL_BINS, RAYON_MAX_MASSE_M,
                             RAYON_MAX_VEGETATION_M, SURSOL_HAUTEUR_MAX_M)
from vue3d.mnh import MNH_SEUIL_M
from vue3d.ortho import EXG_SEUIL
from vue3d.houppiers import _fenetre, lisser, rayon_cellules


def _dilater(grille):
    p = np.pad(grille, 1, mode="edge")
    out = grille.copy()
    for dj in range(3):
        for di in range(3):
            np.maximum(out, p[dj:dj + grille.shape[0], di:di + grille.shape[1]], out=out)
    return out


def _decales(grille, remplissage):
    p = np.pad(grille, 1, mode="constant", constant_values=remplissage)
    ny, nx = grille.shape
    return [p[1 + dj:1 + dj + ny, 1 + di:1 + di + nx]
            for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]


def sommets(lisse, masque, rayons):
    base = np.where(masque, lisse, -1.0).astype(np.float32)
    courant = base.copy()
    marqueurs = np.zeros(lisse.shape, dtype=bool)
    for k in range(1, int(rayons.max()) + 1):
        courant = _dilater(courant)
        marqueurs |= (rayons == k) & masque & (base >= courant)
    return marqueurs


def segmenter(lisse, masque, rayons, etendue_max, stats=None):
    ny, nx = lisse.shape
    labels_bordes = np.zeros((ny + 2, nx + 2), dtype=np.int32)
    labels = labels_bordes[1:-1, 1:-1]
    marqueurs = sommets(lisse, masque, rayons)
    apex = [(0, 0)]

    def poser_marqueurs(sel):
        js, is_ = np.nonzero(sel)
        for k, (j, i) in enumerate(zip(js, is_), start=len(apex)):
            labels[j, i] = k
            apex.append((int(j), int(i)))

    poser_marqueurs(marqueurs)
    if stats is not None:
        stats["marqueurs"] = len(apex) - 1
    restants = masque & (labels == 0)
    hauteur_ref = np.where(masque, lisse, -1.0).astype(np.float32)
    hauteur_bordee = np.pad(hauteur_ref, 1, mode="constant", constant_values=-1.0)
    voisins = [(dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]
    for _ in range(64):
        if not restants.any():
            break
        progres = False
        aj = np.array([a[0] for a in apex], dtype=np.int32)
        ai = np.array([a[1] for a in apex], dtype=np.int32)
        ray = np.minimum(rayons[aj, ai] * ETENDUE_FACTEUR, etendue_max).astype(np.float32)
        ray[0] = 0
        for _ in range(64):
            j, i = np.nonzero(restants & _dilater(labels > 0))
            h = hauteur_ref[j, i]
            meilleur_h = np.full(len(j), -1.0)
            meilleur_l = np.zeros(len(j), dtype=np.int32)
            for dj, di in voisins:
                vl = labels_bordes[j + 1 + dj, i + 1 + di]
                vh = hauteur_bordee[j + 1 + dj, i + 1 + di]
                dist = np.hypot(j - aj[vl], i - ai[vl])
                score = vh + np.where(vh >= h, 1000.0, 0.0)
                ok = (vl > 0) & (dist <= ray[vl]) & (score > meilleur_h)
                meilleur_h = np.where(ok, score, meilleur_h)
                meilleur_l = np.where(ok, vl, meilleur_l)
            pris = meilleur_l > 0
            if not pris.any():
                break
            labels[j[pris], i[pris]] = meilleur_l[pris]
            restants[j[pris], i[pris]] = False
            progres = True
        if not restants.any():
            break
        h_orph = np.where(restants, hauteur_ref, -1.0).astype(np.float32)
        nouveaux = restants & (h_orph >= _dilater(h_orph))
        if not nouveaux.any() and not progres:
            break
        poser_marqueurs(nouveaux)
        restants &= ~nouveaux
    if stats is not None:
        stats["orphelins"] = len(apex) - 1 - stats["marqueurs"]
    _fondre_petits(labels, hauteur_ref, len(apex))
    return labels, apex


def _fondre_petits(labels, hauteur_ref, n_labels):
    for _ in range(3):
        n = np.bincount(labels.ravel(), minlength=n_labels)
        petits = np.zeros(n_labels, dtype=bool)
        petits[1:] = n[1:] < HOUPPIERS_MIN_CELLULES
        cibles = petits[labels]
        if not cibles.any():
            return
        meilleur_h = np.full(labels.shape, -1.0, dtype=np.float32)
        meilleur_l = np.zeros(labels.shape, dtype=np.int32)
        for vl, vh in zip(_decales(labels, 0), _decales(hauteur_ref, -1.0)):
            ok = cibles & (vl > 0) & ~petits[vl] & (vh > meilleur_h)
            meilleur_h = np.where(ok, vh, meilleur_h)
            meilleur_l = np.where(ok, vl, meilleur_l)
        labels[cibles] = meilleur_l[cibles]


def _essences_par_cellule(features, lons, lats, cle):
    valeurs = np.full((len(lats), len(lons)), -1, dtype=np.int32)
    noms = []
    for f in features or []:
        try:
            geom = shape(f["geometry"])
        except Exception:
            continue
        nom = (f.get("properties") or {}).get(cle)
        if nom not in noms:
            noms.append(nom)
        fenetre = _fenetre(geom, lons, lats)
        dedans = shapely.contains_xy(geom, *np.meshgrid(lons[fenetre[1]], lats[fenetre[0]]))
        vue = valeurs[fenetre]
        vue[dedans & (vue < 0)] = noms.index(nom)
    return valeurs, noms


def decrire(labels, apex, hauteur, xs, ys, pas, classe, natures, noms_nature,
            essences, noms_essence, lons, lats):
    n_labels = len(apex)
    flat = labels.ravel()
    ok = flat > 0
    lab = flat[ok]
    h = hauteur.ravel()[ok].astype(np.float64)
    x = xs.ravel()[ok].astype(np.float64)
    y = ys.ravel()[ok].astype(np.float64)
    n = np.bincount(lab, minlength=n_labels)
    aj = np.array([a[0] for a in apex]); ai = np.array([a[1] for a in apex])
    h_max = np.zeros(n_labels); np.maximum.at(h_max, lab, h)
    ax = xs[aj, ai]; ay = ys[aj, ai]
    d = np.hypot(x - ax[lab], y - ay[lab])
    d_max = np.zeros(n_labels); np.maximum.at(d_max, lab, d)
    part = np.clip(np.floor(d / np.maximum(d_max[lab], pas) * PROFIL_BINS), 0, PROFIL_BINS - 1).astype(np.int64)
    cle = lab * PROFIL_BINS + part
    somme_h = np.bincount(cle, weights=h, minlength=n_labels * PROFIL_BINS)
    nb = np.bincount(cle, minlength=n_labels * PROFIL_BINS)
    sx = np.bincount(lab, weights=x, minlength=n_labels)
    sy = np.bincount(lab, weights=y, minlength=n_labels)
    sxx = np.bincount(lab, weights=x * x, minlength=n_labels)
    syy = np.bincount(lab, weights=y * y, minlength=n_labels)
    sxy = np.bincount(lab, weights=x * y, minlength=n_labels)

    out = []
    for k in range(1, n_labels):
        if n[k] < HOUPPIERS_MIN_CELLULES:
            continue
        mx, my = sx[k] / n[k], sy[k] / n[k]
        cxx = sxx[k] / n[k] - mx * mx + pas * pas / 12
        cyy = syy[k] / n[k] - my * my + pas * pas / 12
        cxy = sxy[k] / n[k] - mx * my
        tr, det = cxx + cyy, cxx * cyy - cxy * cxy
        disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-9)
        allongement = min(math.sqrt(l1 / l2), ALLONGEMENT_MAX)
        axe = math.degrees(0.5 * math.atan2(2 * cxy, cxx - cyy)) % 180
        profil = []
        dernier = 1.0
        for b in range(PROFIL_BINS):
            c = nb[k * PROFIL_BINS + b]
            if c:
                dernier = somme_h[k * PROFIL_BINS + b] / c / max(h_max[k], 0.1)
            profil.append(round(float(min(dernier, 1.0)), 2))
        j, i = apex[k]
        nature = natures[j, i]
        essence = essences[j, i]
        out.append({
            "lon": round(float(lons[j, i]), 7), "lat": round(float(lats[j, i]), 7),
            "h": round(float(h_max[k]), 1),
            "r": round(math.sqrt(n[k] * pas * pas / math.pi), 2),
            "r_max": round(float(d_max[k]) + pas / 2, 2),
            "profil": profil,
            "allongement": round(allongement, 2),
            "axe_deg": round(axe, 1),
            "n": int(n[k]),
            "classe": classe,
            "nature": noms_nature[nature] if nature >= 0 else None,
            "essence": noms_essence[essence] if essence >= 0 else None,
        })
    return out


def segmenter_emprise(grille, exg, batiments, vegetation, forets):
    west, south, east, north = grille["bbox"]
    nx, ny = grille["width"], grille["height"]
    H = np.asarray(grille["values"], dtype=np.float32).reshape(ny, nx)
    seuil = float(grille.get("seuil_m") or MNH_SEUIL_M)
    lat0 = (south + north) / 2
    m_lon = 111320 * math.cos(math.radians(lat0))
    pas = (east - west) * m_lon / nx
    lons = west + (east - west) * (np.arange(nx) + 0.5) / nx
    lats = north - (north - south) * (np.arange(ny) + 0.5) / ny
    LON, LAT = np.meshgrid(lons, lats)
    xs = ((LON - (west + east) / 2) * m_lon).astype(np.float32)
    ys = ((LAT - lat0) * 111320).astype(np.float32)

    sursol = H >= seuil
    bati = np.zeros((ny, nx), dtype=bool)
    for f in (batiments or {}).get("features", []):
        try:
            geom = shape(f["geometry"])
            fenetre = _fenetre(geom, lons, lats)
            bati[fenetre] |= shapely.contains_xy(geom, LON[fenetre], LAT[fenetre])
        except Exception:
            continue
    natures, noms_nature = _essences_par_cellule(
        (vegetation or {}).get("features", []), lons, lats, "nature")
    essences, noms_essence = _essences_par_cellule(
        (forets or {}).get("features", []), lons, lats, "essence")
    vegetal = natures >= 0
    en_foret = np.isin(natures, [k for k, nom in enumerate(noms_nature)
                                 if nom and nom.startswith(NATURES_FORET)])
    trop_haut = ~en_foret & (H > SURSOL_HAUTEUR_MAX_M)
    nb_ortho = 0
    if exg is not None:
        par_ortho = ~vegetal & (exg >= EXG_SEUIL)
        nb_ortho = int((par_ortho & sursol & ~bati & ~trop_haut).sum())
        vegetal |= par_ortho
    candidats = sursol & ~bati & ~trop_haut
    masque_veg = candidats & vegetal
    masque_masse = candidats & ~vegetal

    lisse = lisser(H)
    out = {"hauteur_max": round(float(H[candidats].max()), 1) if candidats.any() else 0.0,
           "nb_ortho": nb_ortho}
    for cle, masque, rayon_max in (("houppiers", masque_veg, RAYON_MAX_VEGETATION_M),
                                   ("masses", masque_masse, RAYON_MAX_MASSE_M)):
        if not masque.any():
            out[cle] = []
            continue
        rayons = rayon_cellules(lisse, pas, rayon_max)
        labels, apex = segmenter(lisse, masque, rayons, rayon_max / pas)
        out[cle] = decrire(labels, apex, H, xs, ys, pas,
                           "vegetation" if cle == "houppiers" else "sursol",
                           natures, noms_nature, essences, noms_essence, LON, LAT)
    return out
