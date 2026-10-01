"""Toits en pans : plans ajustés au LiDAR et volume fermé, pour les toits que le
résumé manque.

La surface mesurée (toits.surface_toit) suit le MNH cellule par cellule : fidèle,
mais elle en garde le grain, et un faîtage y devient une crête bosselée. Ce
module remplace, là où c'est possible, cette grille par quelques plans :

1. segmentation de la surface du toit en pans plans, par croissance de régions
   déterministe (pas de RANSAC : la scène est mise en cache pour toujours, un
   tirage au sort y serait figé) ;
2. partition de l'emprise BD TOPO exacte par les frontières entre pans, posées
   sur la droite d'intersection P_i ∩ P_j quand elles en sont proches (faîtages,
   noues) ;
3. volume fermé : chaque face relevée sur son plan, murs verticaux aux
   décrochements entre pans et sur le pourtour, plancher. Le volume est VÉRIFIÉ
   — chaque arête portée par exactement deux triangles, en sens opposés — et
   refusé sinon.

Tout échec rend None : le toit garde alors la surface mesurée, puis le toit
résumé. Jamais un volume approximatif.

Mesuré le 2026-09-28 sur Gordes (79 toits) et Strasbourg (66) par les
prototypes d'outils/ (prototype_plans.py, prototype_brep.py) : couverture en
pans >= 80 % pour 91 % et 85 % des toits fiables, écart médian au LiDAR 0,10 et
0,17 m, moitié moins de triangles que la surface, 4 à 9 ms par bâtiment.
"""

import logging
import math
from array import array
from collections import Counter, defaultdict, deque

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize

journal = logging.getLogger(__name__)

# --- Segmentation ----------------------------------------------------------
# Tolérances de croissance d'une région : angle entre la normale d'une cellule
# et celle du plan courant, écart de la cellule à ce plan. Mesuré sur 165 toits
# fiables : à 12° et 0,15 m (valeurs courantes de la littérature), 30 % de
# couverture seulement sur les toits de plus de 35° de Strasbourg — à maille
# fixe, le bruit de la normale croît avec la pente. À 25° et 0,40 m, couverture
# >= 80 % pour 91 % des toits de Gordes et 85 % de ceux de Strasbourg, pour un
# écart médian au plan de 0,07 à 0,12 m.
PANS_ANGLE_DEG = 25.0
PANS_ECART_M = 0.40
# Plan réajusté (moindres carrés) toutes les N cellules ajoutées.
PANS_REAJUSTEMENT = 20
# Le produit scalaire de deux normales unitaires se calcule en float Python,
# dix fois moins cher que le `@` de numpy, qui passe par BLAS. Les deux ne
# sont pas toujours égaux au bit près : sur 1,6 million de produits des toits
# de Strasbourg en zone de 1 000 m, aucun écart sous macOS (Accelerate), mais
# 576 000 dans le conteneur (OpenBLAS), de 2,2e-16 au plus. Seule compte la
# comparaison au seuil d'angle : à moins de PANS_DOUTE_PRODUIT de lui, c'est
# le `@` d'origine qui tranche, et la région croît comme avant. Aucun des
# 1,6 million n'en était si près.
PANS_DOUTE_PRODUIT = 1e-12
# Un pan plus petit que 3 m² (12 cellules), ou que 4 % du toit, est du bruit :
# cheminée, lucarne, bord mêlé de sol.
PANS_MIN_CELLULES = 12
PANS_MIN_PART = 0.04
# Normale presque horizontale : une façade vue par le LiDAR, pas un pan.
PANS_NZ_MIN = 0.2
# Deux pans voisins presque coplanaires n'en font qu'un (sur-segmentation).
PANS_FUSION_DEG = 5.0
PANS_FUSION_M = 0.15
# Part des cellules du toit que les pans doivent couvrir : en deçà, le toit
# n'est pas fait de plans (dôme, végétation, toit trop découpé) et garde sa
# surface mesurée. Mesuré : 91 % (Gordes) et 85 % (Strasbourg) des toits
# fiables passent à 25° / 0,40 m.
PANS_COUVERTURE_MIN = 0.8

# --- Partition et volume ---------------------------------------------------
# Les étiquettes des pans sont prolongées jusqu'à cette distance hors de
# l'emprise : les frontières entre pans la traversent alors franchement, et
# l'emprise exacte les coupe — les murs suivent le contour BD TOPO, pas
# l'escalier des cellules.
PANS_DEBORD_M = 1.0
# Simplification des frontières (Douglas-Peucker) : moins d'une maille, pour
# ne pas couper à travers un petit pan.
PANS_SIMPLIFIE_M = 0.3
# Une frontière à moins de cette distance médiane de la droite P_i ∩ P_j est
# un faîtage ou une noue : elle est posée dessus, et les deux pans s'y
# rejoignent sans mur. Mesuré : médiane 0,58 à 0,80 m pour les vrais faîtages,
# p90 5,8 m pour les décrochements et les cours, qui gardent leur mur.
PANS_FAITAGE_M = 1.2
# Pente relative minimale entre deux plans pour que leur intersection soit
# une droite exploitable ; en deçà, c'est un décrochement.
PANS_FAITAGE_PENTE_MIN = 0.02
# Deux altitudes d'un même nœud à moins de 1 mm sont le même sommet.
PANS_SOUDURE_M = 1e-3
# Un sommet de toit plus haut que la plus haute cellule mesurée, à cette marge
# près, trahit un plan prolongé loin de ses cellules : le volume est refusé.
# Le faîtage analytique dépasse légitimement les cellules, lissées par la
# médiane 3×3 : sur un pan à 60°, 0,75 m d'écart horizontal font 1,3 m.
# Mesuré à Strasbourg, dépassements des toits refusés à 1 m de marge : sept
# entre 1,0 et 1,9 m, puis un saut à 2,4 m et une traîne jusqu'à 16,9 m.
# Pas de borne basse autre que le sol : les cellules à moins de 0,75 m du
# contour sont écartées (toits.SURFACE_RETRAIT_M), et un pan raide prolongé
# jusqu'à la gouttière descend légitimement plus bas qu'elles — une borne
# basse à 1 m refusait 29 toits sur 62 à Strasbourg.
PANS_MARGE_Z_M = 2.0
# Quantification du volume transmis : sommets au dixième de maille (~5 cm)
# dans le plan, au décimètre en hauteur, comme la surface.
PANS_QUANTUM_MAILLE = 10
PANS_QUANTUM_Z = 10


# --- Segmentation ----------------------------------------------------------

def _plan(xs, ys, zs):
    """Plan z = a·x + b·y + c (moindres carrés) et sa normale unitaire."""
    # column_stack plutôt que np.c_ : la même matrice (valeurs, ordre,
    # contiguïté), donc le même lstsq au bit près, et 4 µs au lieu de 17 par
    # appel (cProfile, 60 000 appels à Strasbourg en zone de 1 000 m).
    A = np.column_stack((xs, ys, np.ones(len(xs))))
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    n = np.array([-coef[0], -coef[1], 1.0])
    return coef, n / np.linalg.norm(n)


def proches(polygone, X, Y, rayon):
    """Masque des cellules dont le centre est à `rayon` au plus du polygone.

    Le même que `shapely.distance(points, polygone) <= rayon`, mais la
    distance n'est calculée que hors du polygone : dedans, GEOS la rend
    nulle. Mesuré sur 590 000 cellules des fenêtres de Strasbourg : 71 ns
    par cellule pour le test d'appartenance, 512 pour la distance, et 37 %
    des cellules dedans.
    """
    out = shapely.contains_xy(polygone, X, Y)
    js, is_ = np.nonzero(~out)
    out[js, is_] = shapely.distance(shapely.points(X[js, is_], Y[js, is_]), polygone) <= rayon
    return out


def _bordee(a, bord):
    """Grille bordée d'une cellule `bord`, à plat, en liste Python.

    La croissance des régions lit ses voisines aux décalages +1, -1, +W, -W
    (W = nx + 2) sans tester le bord de la fenêtre : une voisine hors de la
    grille tombe sur la bordure, non valide et sans étiquette, et compte
    comme si elle n'avait pas été lue.
    """
    return np.pad(a, 1, constant_values=bord).ravel().tolist()


def segmenter(z, valides, X, Y):
    """Pans plans d'une grille de hauteurs, par croissance de régions.

    Args:
        z: hauteurs sans NaN (cellules non valides comblées).
        valides: cellules lues comme toit.
        X, Y: coordonnées des centres, mètres, y vers le nord.

    Returns:
        (etiquettes, plans) : étiquette >= 0 par cellule assignée, -1 sinon ;
        plans[k] = (coef, normale), None pour une région rejetée.
    """
    ny, nx = z.shape
    dx, dy = X[0, 1] - X[0, 0], Y[0, 0] - Y[1, 0]
    dz_sud, dz_est = np.gradient(z, dy, dx)
    n = np.stack([-dz_est, dz_sud, np.ones_like(z)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    # Normales lissées 3×3 : sur un versant raide, le bruit du MNH fait battre
    # la normale d'une cellule seule et fragmente la région.
    p = np.pad(n, ((1, 1), (1, 1), (0, 0)), mode="edge")
    n = sum(p[a:a + ny, b:b + nx] for a in range(3) for b in range(3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    zp = np.pad(z, 1, mode="edge")
    courbure = np.abs(4 * z - zp[:-2, 1:-1] - zp[2:, 1:-1] - zp[1:-1, :-2] - zp[1:-1, 2:])

    js, is_ = np.nonzero(valides)
    ordre = np.lexsort((is_, js, courbure[js, is_]))
    plans = []
    cos_min = math.cos(math.radians(PANS_ANGLE_DEG))
    seuil = max(PANS_MIN_CELLULES, int(PANS_MIN_PART * valides.sum()))

    # La croissance passe cellule par cellule : elle lit des listes Python à
    # plat, bordées (`_bordee`), plutôt que des scalaires numpy, quatre à
    # cinq fois plus chers à indexer. Le résidu au plan y est le même calcul
    # en float Python qu'en float64 numpy, au bit près ; le produit scalaire
    # des normales aussi, sauf à PANS_DOUTE_PRODUIT du seuil, où le `@` numpy
    # d'avant tranche. Mesuré sur les 521 toits en pans de Strasbourg en zone
    # de 1 000 m : 4,8 s -> 1,95 s (médianes de six mesures alternées),
    # mêmes étiquettes et mêmes plans au bit près.
    W = nx + 2
    lab = [-1] * ((ny + 2) * W)
    val = _bordee(valides, False)
    Xp, Yp, zp_ = (np.pad(a, 1).ravel() for a in (X, Y, z))
    Xl, Yl, zl = Xp.tolist(), Yp.tolist(), zp_.tolist()
    N0, N1, N2 = (_bordee(n[..., a], 0.0) for a in range(3))
    # Est, ouest, sud, nord : l'ordre de parcours fixe celui des cellules de
    # la région, donc ses réajustements ; il ne doit pas changer.
    voisinage = (1, -1, W, -W)
    amorce = (-W - 1, -W, -W + 1, -1, 0, 1, W - 1, W, W + 1)

    # Graines par courbure croissante : on part du plat des pans.
    jsl, isl = js.tolist(), is_.tolist()
    for o in ordre.tolist():
        c0 = (jsl[o] + 1) * W + isl[o] + 1
        if lab[c0] != -1:
            continue
        k = len(plans)
        lab[c0] = k
        region = array("q", (c0,))
        coef = None
        file, depuis = deque(region), 0
        while file:
            c = file.popleft()
            for d in voisinage:
                v = c + d
                if lab[v] != -1 or not val[v]:
                    continue
                if coef is None:
                    # Amorce sur le voisinage 3×3 : la normale d'une cellule
                    # seule est trop bruitée pour porter un plan. Ajustée à
                    # la première voisine libre seulement : une graine sans
                    # voisine libre reste seule, et son plan ne servirait
                    # pas — 16 500 lstsq épargnés sur 45 500 graines à
                    # Strasbourg en zone de 1 000 m.
                    vois = np.array([c0 + e for e in amorce if val[c0 + e]])
                    coef, npl = _plan(Xp[vois], Yp[vois], zp_[vois])
                    ca, cb, cc = coef.tolist()
                    pa, pb, pc = npl.tolist()
                # Deux tests sans effet de bord, l'un ou l'autre rejette : le
                # résidu, le moins cher, d'abord.
                if abs(zl[v] - (ca * Xl[v] + cb * Yl[v] + cc)) > PANS_ECART_M:
                    continue
                produit = N0[v] * pa + N1[v] * pb + N2[v] * pc
                if abs(produit - cos_min) <= PANS_DOUTE_PRODUIT:
                    produit = float(n[v // W - 1, v % W - 1] @ npl)
                if produit < cos_min:
                    continue
                lab[v] = k
                region.append(v)
                file.append(v)
                depuis += 1
                if depuis >= PANS_REAJUSTEMENT:
                    depuis = 0
                    r = np.array(region)
                    coef, npl = _plan(Xp[r], Yp[r], zp_[r])
                    ca, cb, cc = coef.tolist()
                    pa, pb, pc = npl.tolist()
        if len(region) >= seuil:
            r = np.array(region)
            plans.append(_plan(Xp[r], Yp[r], zp_[r]))
        else:
            # Rejetée ci-dessous pour sa taille, quel que soit son plan : ne
            # pas l'ajuster épargne la plupart des lstsq (une région par
            # graine, et presque toutes n'ont que quelques cellules).
            plans.append(None)
    etiquettes = np.array(lab, dtype=np.int32).reshape(ny + 2, W)[1:-1, 1:-1].copy()

    # Régions trop petites ou presque verticales : rejetées.
    tailles = np.bincount(etiquettes[etiquettes >= 0], minlength=len(plans))
    rejet = np.array([p is None or tailles[k] < seuil or p[1][2] < PANS_NZ_MIN
                      for k, p in enumerate(plans)], dtype=bool)
    if rejet.any():
        for k in np.flatnonzero(rejet).tolist():
            plans[k] = None
        m = etiquettes >= 0
        e = etiquettes[m]
        etiquettes[m] = np.where(rejet[e], -1, e)

    # Fusion des pans voisins coplanaires, jusqu'à stabilité.
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

    # Cellules orphelines (bords de pans, cheminées arasées) : au pan voisin
    # le plus proche en hauteur, s'il l'est à moins de deux tolérances. En
    # listes à plat, comme la croissance ; une cellule rattachée l'est tout
    # de suite pour les suivantes de la même passe, comme avant.
    lab = _bordee(etiquettes, -1)
    coefs = [None if p is None else p[0].tolist() for p in plans]
    # Les orphelines dans l'ordre des lignes, comme np.nonzero : une passe
    # reprend celles que la précédente a laissées, dans le même ordre.
    restantes = np.flatnonzero(np.array(val) & (np.array(lab) == -1)).tolist()
    for _ in range(8):
        bouge, suivantes = False, []
        for c in restantes:
            mieux, res_min = -1, 2 * PANS_ECART_M
            for d in voisinage:
                k = lab[c + d]
                if k >= 0:
                    a, b, e = coefs[k]
                    r = abs(zl[c] - (a * Xl[c] + b * Yl[c] + e))
                    if r < res_min:
                        mieux, res_min = k, r
            if mieux >= 0:
                lab[c] = mieux
                bouge = True
            else:
                suivantes.append(c)
        if not bouge:
            break
        restantes = suivantes
    return np.array(lab, dtype=np.int32).reshape(ny + 2, W)[1:-1, 1:-1].copy(), plans


def _prolonger(etiquettes, plans, z, zone, X, Y):
    """Étend les étiquettes à toute la zone : pan voisin de moindre écart.

    Passe après passe, en listes à plat bordées comme `segmenter` : une
    cellule étiquetée l'est aussitôt pour les suivantes de la même passe.
    """
    ny, nx = etiquettes.shape
    W = nx + 2
    lab = _bordee(etiquettes, -1)
    zl, Xl, Yl = (_bordee(a, 0.0) for a in (z, X, Y))
    coefs = [None if p is None else p[0].tolist() for p in plans]
    dans_zone = np.pad(zone, 1, constant_values=False).ravel()
    # Comme dans segmenter : les cellules sans étiquette dans l'ordre des
    # lignes, et chaque passe reprend celles que la précédente a laissées.
    restantes = np.flatnonzero(dans_zone & (np.array(lab) == -1)).tolist()
    while True:
        bouge, suivantes = False, []
        for c in restantes:
            mieux, res_min = -1, float("inf")
            for d in (1, -1, W, -W):                 # est, ouest, sud, nord
                k = lab[c + d]
                if k >= 0:
                    a, b, e = coefs[k]
                    r = abs(zl[c] - (a * Xl[c] + b * Yl[c] + e))
                    if r < res_min:
                        mieux, res_min = k, r
            if mieux >= 0:
                lab[c] = mieux
                bouge = True
            else:
                suivantes.append(c)
        if not bouge:
            return np.array(lab, dtype=etiquettes.dtype).reshape(ny + 2, W)[1:-1, 1:-1].copy()
        restantes = suivantes


# --- Frontières ------------------------------------------------------------

def _chainer(aretes):
    """Arêtes (n1, n2) d'une frontière -> chaînes de nœuds, coupées aux nœuds
    de degré différent de 2."""
    incidentes = defaultdict(list)
    for k, (n1, n2) in enumerate(aretes):
        incidentes[n1].append(k)
        incidentes[n2].append(k)
    vues = [False] * len(aretes)

    def suivre(noeud, k):
        chaine = [noeud]
        while True:
            vues[k] = True
            n1, n2 = aretes[k]
            noeud = n2 if n1 == noeud else n1
            chaine.append(noeud)
            suite = [kk for kk in incidentes[noeud] if not vues[kk]]
            if len(incidentes[noeud]) != 2 or not suite:
                return chaine
            k = suite[0]

    chaines = []
    for noeud in sorted(n for n, ks in incidentes.items() if len(ks) != 2):
        for k in incidentes[noeud]:
            if not vues[k]:
                chaines.append(suivre(noeud, k))
    for k in range(len(aretes)):
        if not vues[k]:
            chaines.append(suivre(aretes[k][0], k))
    return chaines


def _frontieres(lab, dx, dy):
    """Frontières entre pans voisins : [(a, b, [(x, y), ...])], a < b.

    Les nœuds sont les coins des cellules, en demi-mailles entières (exactes)
    avant conversion en mètres ; la cellule (j, i) est centrée en
    (i·dx, −j·dy).
    """
    aretes = defaultdict(list)
    g, d = lab[:, :-1], lab[:, 1:]
    for j, i in zip(*np.nonzero((g != d) & (g >= 0) & (d >= 0))):
        a, b = sorted((int(g[j, i]), int(d[j, i])))
        aretes[(a, b)].append(((2 * i + 1, 2 * j - 1), (2 * i + 1, 2 * j + 1)))
    h, s = lab[:-1, :], lab[1:, :]
    for j, i in zip(*np.nonzero((h != s) & (h >= 0) & (s >= 0))):
        a, b = sorted((int(h[j, i]), int(s[j, i])))
        aretes[(a, b)].append(((2 * i - 1, 2 * j + 1), (2 * i + 1, 2 * j + 1)))
    lignes = []
    for (a, b), liste in sorted(aretes.items()):
        for chaine in _chainer(liste):
            pts = [(u * dx / 2, -v * dy / 2) for u, v in chaine]
            lignes.append((a, b, list(LineString(pts).simplify(PANS_SIMPLIFIE_M).coords)))
    return lignes


def _poser_faitages(lignes, plans):
    """Pose sur la droite P_i ∩ P_j les frontières qui en sont proches.

    Les extrémités sont partagées entre lignes : chaque nœud reçoit une seule
    nouvelle position — projection sur sa droite, ou intersection de ses
    droites (le sommet d'une croupe) — appliquée à toutes ses lignes. Une
    ligne qui en croiserait une autre reprend sa géométrie mesurée.
    """
    droites = {}
    for idx, (a, b, pts) in enumerate(lignes):
        ca, cb = plans[a][0], plans[b][0]
        da, db, dc = ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]
        n = math.hypot(da, db)
        if n < PANS_FAITAGE_PENTE_MIN:
            continue
        da, db, dc = da / n, db / n, dc / n
        if float(np.median([abs(da * x + db * y + dc) for x, y in pts])) <= PANS_FAITAGE_M:
            droites[idx] = (da, db, dc)

    incidences = defaultdict(list)
    for idx, d in droites.items():
        pts = lignes[idx][2]
        for noeud in (pts[0], pts[-1]):
            incidences[noeud].append(d)
    deplace = {}
    for noeud, ds in sorted(incidences.items()):
        if len(ds) >= 2:
            A = np.array([[d[0], d[1]] for d in ds])
            r = -np.array([d[2] for d in ds])
            sol, _, rang, _ = np.linalg.lstsq(A, r, rcond=None)
            if rang == 2 and math.dist(sol, noeud) <= 3 * PANS_FAITAGE_M:
                deplace[noeud] = (float(sol[0]), float(sol[1]))
                continue
        da, db, dc = ds[0]
        t = da * noeud[0] + db * noeud[1] + dc
        deplace[noeud] = (noeud[0] - t * da, noeud[1] - t * db)

    def bouts(pts):
        return [deplace.get(pts[0], pts[0])] + pts[1:-1] + [deplace.get(pts[-1], pts[-1])]

    posees = []
    for idx, (a, b, pts) in enumerate(lignes):
        pts = bouts(list(pts))
        if idx in droites:
            da, db, dc = droites[idx]
            milieu = [(x - (da * x + db * y + dc) * da, y - (da * x + db * y + dc) * db)
                      for x, y in pts[1:-1]]
            pts = list(LineString([pts[0]] + milieu + [pts[-1]]).simplify(PANS_SIMPLIFIE_M / 3).coords)
        posees.append((a, b, pts))
    geoms = [LineString(p) for _, _, p in posees]
    for idx in sorted(droites):
        if any(k != idx and geoms[idx].crosses(g) for k, g in enumerate(geoms)):
            a, b, pts = lignes[idx]
            posees[idx] = (a, b, bouts(list(pts)))
            geoms[idx] = LineString(posees[idx][2])
    return [(a, b, p) for a, b, p in posees if LineString(p).length > PANS_SOUDURE_M]


# --- Volume ------------------------------------------------------------------

def _ccw(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) > 0


def _trianguler(polygone, eclat_permis=True):
    """Triangles anti-horaires pavant le polygone, tous ses sommets gardés, ou
    None.

    Triangulation de Delaunay contrainte (trous compris). Un éclat d'aire
    nulle — polygonize en laisse entre deux lignes presque confondues (neuf
    sur 100 toits à Gordes et Strasbourg) — est découpé en éventail dans
    l'ordre de son anneau, supposé anti-horaire : des triangles invisibles,
    mais qui gardent le volume fermé.

    Un refus de GEOS n'est pas une erreur de source : c'est une géométrie que
    ce bâtiment ne permet pas, et il garde alors sa surface mesurée.
    """
    if not polygone.is_valid:
        return None
    if polygone.area <= 1e-9:
        if not eclat_permis or polygone.interiors:
            return None
        c = list(polygone.exterior.coords)[:-1]
        return [(c[0], c[k], c[k + 1]) for k in range(1, len(c) - 1)]
    try:
        triangles = shapely.constrained_delaunay_triangles(polygone)
    except shapely.errors.GEOSException:
        return None
    # Les sommets de tous les triangles d'un appel, plutôt qu'un objet shapely
    # par triangle et par anneau : les mêmes flottants, dans le même ordre,
    # pour 19 600 triangulations à Strasbourg en zone de 1 000 m.
    coords = shapely.get_coordinates(triangles)
    if len(coords) == 4 * shapely.get_num_geometries(triangles):
        tris = [[tuple(p) for p in t[:3]] for t in coords.reshape(-1, 4, 2).tolist()]
    else:
        tris = [list(t.exterior.coords)[:3] for t in triangles.geoms]
    aire = sum(abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2
               for a, b, c in tris)
    if abs(aire - polygone.area) > 1e-6 * max(polygone.area, 1.0):
        return None
    return [(a, b, c) if _ccw(a, b, c) else (a, c, b) for a, b, c in tris]


def ferme_et_oriente(triangles):
    """Chaque arête portée par exactement deux triangles, parcourue en sens
    opposés : volume fermé, normales cohérentes."""
    orientees = Counter()
    for a, b, c in triangles:
        orientees.update(((a, b), (b, c), (c, a)))
    return all(n == 1 and orientees.get((b, a)) == 1 for (a, b), n in orientees.items())


def volume(sommets, triangles):
    """Volume signé (positif si les normales pointent vers l'extérieur)."""
    s = np.asarray(sommets, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    return float(np.einsum("ij,ij->i", s[t[:, 0]], np.cross(s[t[:, 1]], s[t[:, 2]])).sum() / 6)


def construire_volume(emprise, lab, plans, X, Y):
    """Volume fermé des pans sur l'emprise exacte.

    Args:
        emprise: polygone BD TOPO, mètres, même repère que X, Y.
        lab: étiquettes prolongées au-delà de l'emprise.

    Returns:
        (sommets [(x, y, z)], toit, murs, plancher) — listes de triangles
        orientés vers l'extérieur — ou None si le volume n'est pas valide.
    """
    ny, nx = lab.shape
    dx, dy = X[0, 1] - X[0, 0], Y[0, 0] - Y[1, 0]
    ox, oy = X[0, 0], Y[0, 0]
    lignes = _poser_faitages(
        [(a, b, [(x + ox, y + oy) for x, y in p]) for a, b, p in _frontieres(lab, dx, dy)], plans)
    anneaux = [LineString(r.coords) for r in (emprise.exterior, *emprise.interiors)]
    reseau = shapely.unary_union([LineString(p) for _, _, p in lignes] + anneaux)
    faces = []
    for f in polygonize(getattr(reseau, "geoms", [reseau])):
        if not emprise.contains(f.representative_point()):
            continue
        # Pan de la face : majorité des cellules qu'elle contient, sinon la
        # cellule sous son point représentatif.
        minx, miny, maxx, maxy = f.bounds
        i0 = max(int(math.floor((minx - ox) / dx)), 0)
        i1 = min(int(math.ceil((maxx - ox) / dx)) + 1, nx)
        j0 = max(int(math.floor((oy - maxy) / dy)), 0)
        j1 = min(int(math.ceil((oy - miny) / dy)) + 1, ny)
        dedans = shapely.contains_xy(f, X[j0:j1, i0:i1], Y[j0:j1, i0:i1])
        labs = lab[j0:j1, i0:i1][dedans]
        labs = labs[labs >= 0]
        if labs.size:
            k = int(np.bincount(labs).argmax())
        else:
            p = f.representative_point()
            i = min(max(int(round((p.x - ox) / dx)), 0), nx - 1)
            j = min(max(int(round((oy - p.y) / dy)), 0), ny - 1)
            k = int(lab[j, i])
        if k < 0:
            return None                       # un trou dans le toit
        faces.append((orient(f, 1.0), k))

    def z(k, pt):
        c = plans[k][0]
        return float(c[0] * pt[0] + c[1] * pt[1] + c[2])

    # Segments : la face est à gauche de chaque arête de ses anneaux orientés.
    bords = defaultdict(list)
    for f, k in faces:
        for r in (f.exterior, *f.interiors):
            c = list(r.coords)
            for p, q in zip(c, c[1:]):
                if p != q:
                    bords[frozenset((p, q))].append((p, q, k))
    segments, croisements = [], {}
    for cle, liste in bords.items():
        if len(liste) > 2:
            return None
        p, q, kg = liste[0]
        kd = liste[1][2] if len(liste) == 2 else -1
        if kd == kg:
            continue
        if kd >= 0:
            gp, gq = z(kg, p) - z(kd, p), z(kg, q) - z(kd, q)
            if gp * gq < 0 and min(abs(gp), abs(gq)) > PANS_SOUDURE_M:
                # Les deux plans se croisent le long du segment : un sommet au
                # croisement exact, dans les deux faces et dans le mur.
                t = gp / (gp - gq)
                x = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
                croisements[cle] = x
                segments += [(p, x, kg, kd), (x, q, kg, kd)]
                continue
        segments.append((p, q, kg, kd))

    def conforme(anneau):
        c = list(anneau.coords)[:-1]
        out = []
        for m, p in enumerate(c):
            out.append(p)
            x = croisements.get(frozenset((p, c[(m + 1) % len(c)])))
            if x is not None:
                out.append(x)
        return out

    # Altitudes de chaque nœud, soudées à 1 mm près : un sommet unique par
    # altitude, partagé par toutes les faces et tous les murs qui s'y touchent.
    brutes = defaultdict(set)
    polys = []
    for f, k in faces:
        poly = Polygon(conforme(f.exterior), [conforme(r) for r in f.interiors])
        polys.append((poly, k))
        for r in (poly.exterior, *poly.interiors):
            for pt in r.coords:
                brutes[pt].add(z(k, pt))
    # Un toit qui passerait sous le sol au pourtour croiserait son propre mur.
    if min(min(v) for v in brutes.values()) <= PANS_SOUDURE_M:
        return None
    for p, q, _, kd in segments:
        if kd < 0:
            brutes[p].add(0.0)
            brutes[q].add(0.0)
    niveaux, canon = {}, {}
    for pt, vals in brutes.items():
        reps = []
        for v in sorted(vals):
            if not reps or v - reps[-1] > PANS_SOUDURE_M:
                reps.append(v)
            canon[(pt, v)] = reps[-1]
        niveaux[pt] = reps

    sommets, index = [], {}

    def sid(pt, zz):
        cle = (pt, canon.get((pt, zz), zz))
        if cle not in index:
            index[cle] = len(sommets)
            sommets.append((pt[0], pt[1], cle[1]))
        return index[cle]

    toit, murs, plancher = [], [], []
    for poly, k in polys:
        tris = _trianguler(poly)
        if tris is None:
            return None
        for a, b, c in tris:                  # anti-horaires : normale vers le haut
            toit.append((sid(a, z(k, a)), sid(b, z(k, b)), sid(c, z(k, c))))

    for p, q, kg, kd in segments:
        zg = (z(kg, p), z(kg, q))
        zd = (z(kd, p), z(kd, q)) if kd >= 0 else (0.0, 0.0)
        cg = [canon[(p, zg[0])], canon[(q, zg[1])]]
        cd = [canon[(p, zd[0])], canon[(q, zd[1])]]
        if cg == cd:
            continue                          # faîtage : les pans se soudent
        gauche_haute = (cg[0] - cd[0]) + (cg[1] - cd[1]) > 0
        haut, bas = (cg, cd) if gauche_haute else (cd, cg)
        L = math.dist(p, q)
        # Mur dans son plan (s, z), côtés verticaux coupés à tous les niveaux
        # du nœud : sans quoi un mur voisin arrêté à mi-hauteur laisse un
        # sommet en T.
        anneau = ([(0.0, bas[0]), (L, bas[1])]
                  + [(L, v) for v in niveaux[q] if bas[1] < v < haut[1]]
                  + [(L, haut[1]), (0.0, haut[0])]
                  + [(0.0, v) for v in reversed(niveaux[p]) if bas[0] < v < haut[0]])
        propre = [pt for m, pt in enumerate(anneau) if pt != anneau[m - 1]]
        if len(propre) < 3:
            continue
        tris = _trianguler(Polygon(propre))
        if tris is None:
            return None
        for a, b, c in tris:
            # (s, z) anti-horaire : normale à droite de p→q, côté bas si la
            # gauche est haute ; sinon, on retourne.
            if not gauche_haute:
                b, c = c, b
            murs.append(tuple(sid(p if s == 0.0 else q, zz) for s, zz in (a, b, c)))

    pourtour = [LineString([p, q]) for p, q, _, kd in segments if kd < 0]
    for f in polygonize(pourtour):
        if not emprise.contains(f.representative_point()):
            continue
        tris = _trianguler(f, eclat_permis=False)
        if tris is None:
            return None
        for a, b, c in tris:                  # retournés : normale vers le bas
            plancher.append((sid(a, 0.0), sid(c, 0.0), sid(b, 0.0)))

    if not ferme_et_oriente(toit + murs + plancher) or volume(sommets, toit + murs + plancher) <= 0:
        return None
    return sommets, toit, murs, plancher


def pans_du_toit(emprise, X, Y, z, valides):
    """Toit en pans d'un bâtiment, prêt pour la vue, ou None.

    Args:
        emprise: polygone BD TOPO, mètres locaux.
        X, Y: centres des cellules de la fenêtre MNH (y vers le nord).
        z: hauteurs du toit au-dessus du point bas du terrain, comblées.
        valides: cellules lues comme toit.

    Returns:
        dict(sommets, triangles, n_toit, n_pans, ecart_m) : sommets en
        triplets (u, v, w) — dixièmes de maille depuis le centre de la
        première cellule de la fenêtre, vers l'est et vers le sud, et
        décimètres de hauteur ; triangles orientés vers l'extérieur, les
        n_toit premiers pour le toit, les suivants pour les murs. Le plancher,
        vérifié, n'est pas transmis : il n'est jamais vu. None si les pans
        couvrent trop peu du toit ou si le volume n'est pas fermé.
    """
    etiquettes, plans = segmenter(z, valides, X, Y)
    if (etiquettes >= 0).sum() < PANS_COUVERTURE_MIN * valides.sum():
        return None
    zone = proches(emprise, X, Y, PANS_DEBORD_M)
    lab = _prolonger(etiquettes, plans, z, zone | (etiquettes >= 0), X, Y)
    vol = construire_volume(emprise, lab, plans, X, Y)
    if vol is None:
        return None
    sommets, toit, murs, plancher = vol

    if max(sommets[i][2] for t in toit for i in t) > z[valides].max() + PANS_MARGE_Z_M:
        return None

    # Quantification, puis nouvelle vérification : c'est le volume transmis
    # qui doit être fermé, pas celui d'avant l'arrondi.
    dx, dy = X[0, 1] - X[0, 0], Y[0, 0] - Y[1, 0]
    ox, oy = X[0, 0], Y[0, 0]
    quant, index, remap = [], {}, []
    for x, y, zz in sommets:
        cle = (round((x - ox) / dx * PANS_QUANTUM_MAILLE), round((oy - y) / dy * PANS_QUANTUM_MAILLE),
               round(zz * PANS_QUANTUM_Z))
        if cle not in index:
            index[cle] = len(quant)
            quant.append(cle)
        remap.append(index[cle])

    def requantifier(tris):
        out = []
        for t in tris:
            a, b, c = (remap[i] for i in t)
            if a != b and b != c and a != c:
                out.append((a, b, c))
        return out

    toit_q, murs_q, plancher_q = requantifier(toit), requantifier(murs), requantifier(plancher)
    reels = [(u * dx / PANS_QUANTUM_MAILLE, -v * dy / PANS_QUANTUM_MAILLE, w / PANS_QUANTUM_Z)
             for u, v, w in quant]
    tous = toit_q + murs_q + plancher_q
    if not ferme_et_oriente(tous) or volume(reels, tous) <= 0:
        return None

    js, is_ = np.nonzero(valides & (etiquettes >= 0))
    zp = np.array([plans[etiquettes[j, i]][0] @ (X[j, i], Y[j, i], 1.0) for j, i in zip(js, is_)])
    return {
        "sommets": [c for s in quant for c in s],
        "triangles": [i for t in toit_q + murs_q for i in t],
        "n_toit": len(toit_q),
        "n_pans": len({int(k) for k in np.unique(etiquettes) if k >= 0}),
        "ecart_m": round(float(np.median(np.abs(z[js, is_] - zp))), 2),
    }
