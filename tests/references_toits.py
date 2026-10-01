"""Calculs d'origine des toitures, gardés pour les tests d'équivalence.

Les versions rapides de vue3d/toits.py et vue3d/pans.py doivent rendre les
mêmes nombres au bit près : la scène est mise en cache pour toujours, et une
scène reconstruite doit être identique à celle d'avant. Ces fonctions sont
celles du commit bc716ba (scène v14), recopiées telles quelles, sans autre
usage que de servir de référence.
"""

import math
import warnings
from collections import deque

import numpy as np
import shapely
from shapely.geometry import Point

from vue3d.ortho import EXG_SEUIL
from vue3d.pans import (PANS_ANGLE_DEG, PANS_ECART_M, PANS_FUSION_DEG, PANS_FUSION_M,
                        PANS_MIN_CELLULES, PANS_MIN_PART, PANS_NZ_MIN, PANS_REAJUSTEMENT)
from vue3d.toits import SURFACE_RETRAIT_M, SURFACE_SOL_PART


def contenu(cellules, inner):
    """profil_toit.contenu : les cellules (x, y, h) dont le centre est dans inner."""
    if inner.is_empty:
        return []
    minx, miny, maxx, maxy = inner.bounds
    return [(x, y, h) for x, y, h in cellules
            if minx <= x <= maxx and miny <= y <= maxy and inner.contains(Point(x, y))]


def faitage_le_plus_proche(dedans, echant):
    """corps_de_toit.plus_proche, cellule par cellule."""
    return [min(range(len(echant)),
                key=lambda k: min((x - a) ** 2 + (y - b) ** 2 for a, b in echant[k]))
            for x, y, *_ in dedans]


def mediane_3x3(grille, valides):
    g = np.where(valides, grille, np.nan)
    p = np.pad(g, 1, constant_values=np.nan)
    ny, nx = g.shape
    pile = np.stack([p[dj:dj + ny, di:di + nx] for dj in range(3) for di in range(3)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmedian(pile, axis=0)
    return np.where(valides, m, np.nan)


def combler(grille):
    g = grille.copy()
    ny, nx = g.shape
    while np.isnan(g).any():
        p = np.pad(g, 1, constant_values=np.nan)
        voisins = np.stack([p[dj:dj + ny, di:di + nx]
                            for dj in range(3) for di in range(3) if (dj, di) != (1, 1)])
        connus = ~np.isnan(voisins)
        n = connus.sum(axis=0)
        somme = np.where(connus, voisins, 0).sum(axis=0)
        a_remplir = np.isnan(g) & (n > 0)
        if not a_remplir.any():
            break
        g[a_remplir] = somme[a_remplir] / n[a_remplir]
    return g


def valides_du_toit(polygone_m, h, X, Y, gouttiere, verdure=None):
    """Le masque `valides` de cellules_du_toit, sur sa fenêtre."""
    dedans = shapely.contains_xy(polygone_m, X, Y)
    loin_du_bord = dedans & (shapely.distance(
        shapely.points(X.ravel(), Y.ravel()), polygone_m.boundary).reshape(X.shape)
        >= SURFACE_RETRAIT_M)
    valides = loin_du_bord & (h >= SURFACE_SOL_PART * gouttiere)
    if verdure is not None:
        valides &= verdure < EXG_SEUIL
    return valides


def _plan(xs, ys, zs):
    A = np.c_[xs, ys, np.ones(len(xs))]
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    n = np.array([-coef[0], -coef[1], 1.0])
    return coef, n / np.linalg.norm(n)


def _voisins4(j, i, ny, nx):
    for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        jj, ii = j + dj, i + di
        if 0 <= jj < ny and 0 <= ii < nx:
            yield jj, ii


def segmenter(z, valides, X, Y):
    ny, nx = z.shape
    dx, dy = X[0, 1] - X[0, 0], Y[0, 0] - Y[1, 0]
    dz_sud, dz_est = np.gradient(z, dy, dx)
    n = np.stack([-dz_est, dz_sud, np.ones_like(z)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    p = np.pad(n, ((1, 1), (1, 1), (0, 0)), mode="edge")
    n = sum(p[a:a + ny, b:b + nx] for a in range(3) for b in range(3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    zp = np.pad(z, 1, mode="edge")
    courbure = np.abs(4 * z - zp[:-2, 1:-1] - zp[2:, 1:-1] - zp[1:-1, :-2] - zp[1:-1, 2:])

    js, is_ = np.nonzero(valides)
    ordre = np.lexsort((is_, js, courbure[js, is_]))
    etiquettes = np.full((ny, nx), -1, dtype=np.int32)
    plans = []
    cos_min = math.cos(math.radians(PANS_ANGLE_DEG))
    for o in ordre:
        j0, i0 = int(js[o]), int(is_[o])
        if etiquettes[j0, i0] != -1:
            continue
        k = len(plans)
        etiquettes[j0, i0] = k
        region = [(j0, i0)]
        vois = [(a, b) for a in range(max(j0 - 1, 0), min(j0 + 2, ny))
                for b in range(max(i0 - 1, 0), min(i0 + 2, nx)) if valides[a, b]]
        a_, b_ = np.array(vois).T
        coef, npl = _plan(X[a_, b_], Y[a_, b_], z[a_, b_])
        file, depuis = deque(region), 0
        while file:
            j, i = file.popleft()
            for jj, ii in _voisins4(j, i, ny, nx):
                if etiquettes[jj, ii] != -1 or not valides[jj, ii]:
                    continue
                if float(n[jj, ii] @ npl) < cos_min:
                    continue
                if abs(z[jj, ii] - (coef[0] * X[jj, ii] + coef[1] * Y[jj, ii] + coef[2])) > PANS_ECART_M:
                    continue
                etiquettes[jj, ii] = k
                region.append((jj, ii))
                file.append((jj, ii))
                depuis += 1
                if depuis >= PANS_REAJUSTEMENT:
                    depuis = 0
                    a_, b_ = np.array(region).T
                    coef, npl = _plan(X[a_, b_], Y[a_, b_], z[a_, b_])
        a_, b_ = np.array(region).T
        plans.append(_plan(X[a_, b_], Y[a_, b_], z[a_, b_]) if len(region) >= 3 else (coef, npl))

    seuil = max(PANS_MIN_CELLULES, int(PANS_MIN_PART * valides.sum()))
    tailles = np.bincount(etiquettes[etiquettes >= 0], minlength=len(plans))
    for k, (coef, npl) in enumerate(plans):
        if tailles[k] < seuil or npl[2] < PANS_NZ_MIN:
            plans[k] = None
            etiquettes[etiquettes == k] = -1

    cos_fusion = math.cos(math.radians(PANS_FUSION_DEG))
    while True:
        paires = set()
        for a, b in ((etiquettes[:, :-1], etiquettes[:, 1:]), (etiquettes[:-1, :], etiquettes[1:, :])):
            m = (a >= 0) & (b >= 0) & (a != b)
            paires |= {(min(u, v), max(u, v)) for u, v in zip(a[m].tolist(), b[m].tolist())}
        fusion = None
        for a, b in sorted(paires):
            if float(plans[a][1] @ plans[b][1]) < cos_fusion:
                continue
            m = (etiquettes == a) | (etiquettes == b)
            ca, cb = plans[a][0], plans[b][0]
            ecart = np.abs((ca[0] - cb[0]) * X[m] + (ca[1] - cb[1]) * Y[m] + ca[2] - cb[2])
            if float(np.median(ecart)) <= PANS_FUSION_M:
                fusion = (a, b, m)
                break
        if fusion is None:
            break
        a, b, m = fusion
        etiquettes[etiquettes == b] = a
        plans[a] = _plan(X[m], Y[m], z[m])
        plans[b] = None

    for _ in range(8):
        js_, is2 = np.nonzero(valides & (etiquettes == -1))
        bouge = False
        for j, i in zip(js_.tolist(), is2.tolist()):
            mieux, res_min = -1, 2 * PANS_ECART_M
            for jj, ii in _voisins4(j, i, ny, nx):
                k = etiquettes[jj, ii]
                if k >= 0:
                    c = plans[k][0]
                    r = abs(z[j, i] - (c[0] * X[j, i] + c[1] * Y[j, i] + c[2]))
                    if r < res_min:
                        mieux, res_min = int(k), r
            if mieux >= 0:
                etiquettes[j, i] = mieux
                bouge = True
        if not bouge:
            break
    return etiquettes, plans


def prolonger(etiquettes, plans, z, zone, X, Y):
    lab = etiquettes.copy()
    ny, nx = lab.shape
    while True:
        js, is_ = np.nonzero(zone & (lab == -1))
        bouge = False
        for j, i in zip(js.tolist(), is_.tolist()):
            mieux, res_min = -1, float("inf")
            for jj, ii in _voisins4(j, i, ny, nx):
                k = lab[jj, ii]
                if k >= 0:
                    c = plans[k][0]
                    r = abs(z[j, i] - (c[0] * X[j, i] + c[1] * Y[j, i] + c[2]))
                    if r < res_min:
                        mieux, res_min = int(k), r
            if mieux >= 0:
                lab[j, i] = mieux
                bouge = True
        if not bouge:
            return lab
