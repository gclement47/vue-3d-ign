"""Prototype C/D : du plan segmenté au B-Rep étanche, mesuré.

Étend prototype-plans.py (Niveau 0 validé : couverture >= 80 %) jusqu'au
volume fermé, sans arrangement analytique lourd :

- les cellules de l'emprise non assignées (bord érodé, cellules vertes,
  orphelines) rejoignent le plan voisin de moindre résidu, par dilatations
  déterministes, jusqu'à couvrir toute l'emprise ;
- les frontières entre labels (et avec l'extérieur) sont extraites du
  quadrillage, chaînées en polylignes PARTAGÉES, simplifiées une seule fois :
  les deux faces riveraines réutilisent la même polyligne, donc pas de
  fissure de simplification ;
- chaque face (polygonize des polylignes) est relevée sur SON plan ; le long
  d'une frontière interne, un quad vertical coud z_i à z_j — d'épaisseur
  quasi nulle sur un vrai faîtage (les deux plans s'y croisent), vrai mur de
  décrochement entre deux niveaux ; le pourtour descend à z = 0 et un
  plancher ferme le volume ;
- étanchéité VÉRIFIÉE : sommets soudés au mm, toute arête doit porter
  exactement deux triangles ; on compte les arêtes ouvertes au lieu de
  l'affirmer.

Métriques par site : part de B-Rep étanches, sommets/triangles (contre le
maillage dense), écart médian |B-Rep − MNH redressé|, minceur des coutures de
faîtage, distance des faîtages à la droite analytique P_i ∩ P_j, temps.
Usage : . .venv/bin/activate && python outils/prototype_brep.py
Sorties dans cache/mesures/ (ignoré par git) : resultats-brep.md, les OBJ des
pires bâtiments de Gordes, et les données de la visionneuse 3D, copiée à
côté — `cd cache/mesures && python3 -m http.server 8123` puis ouvrir
http://localhost:8123/inspecteur-lod2.html.

Résultats du 2026-09-28 (couverture >= 80 %, config 25°/0,40 m) : 79 B-Rep à
Gordes dont 92 % étanches (0 arête ouverte, COMPTÉE), 66 à Strasbourg dont
76 % ; écart médian 0,10 / 0,17 m ; moitié moins de triangles que la surface
dense ; 4-9 ms par bâtiment. Trois pièges résolus, chacun validé au compteur
d'arêtes ouvertes (32 % -> 92 % d'étanches à Gordes) : l'échelle de niveaux z
aux jonctions verticales (sommets en T sinon), le voisinage des faces par
table topologique des segments (les sondes géométriques ratent les faces
minces), la subdivision conforme au point exact où deux plans se croisent le
long d'une frontière (183 cas à Strasbourg). Restes connus : les faces à
trous, triangulées en Delaunay non conforme (5-8 par site), concentrent les
volumes non étanches restants.
"""

import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
DOSSIER = RACINE / "cache" / "mesures"

import numpy as np
import shapely
from shapely.geometry import shape, LineString
from shapely.ops import transform, polygonize

from vue3d.scene import emprise
from vue3d.couches import lire_couche, COUCHE_BATIMENTS
from vue3d.mnh import fetch_mnh_grid, fetch_sol_grid
from vue3d.ortho import fetch_exg_grid
from vue3d.toits import (TOITS_RESOLUTION_M, SURFACE_ECART_RESUME_M,
                         _cellules_locales, _verdure_locale, _combler,
                         _morceaux, _sol_bas, profil_toit, part_verte,
                         qualifier_couvert, cellules_du_toit, ecart_au_resume)

import prototype_plans as pp

PAS = TOITS_RESOLUTION_M
ANGLE, DIST = 25.0, 0.40          # config « large », validée par le prototype
COUVERTURE_MIN = 0.8
SIMPLIFIE_M = 0.3                 # tolérance de simplification des polylignes
SOUDURE_M = 1e-3                  # soudure des sommets au millimètre
# Deux plans dont les pentes projetées diffèrent de moins de ça sont un
# décrochement (pas de droite d'intersection exploitable), pas un faîtage.
FAITAGE_PENTE_MIN = 0.02
# Une frontière mesurée à moins de cette distance médiane de la droite
# P_i ∩ P_j est un faîtage : elle est posée dessus (mesuré au prototype
# précédent : p50 0,58-0,80 m, les vrais faîtages ; p90 5,8 m, les
# décrochements et les frontières de cour, qui gardent leur mur).
SNAP_DIST_M = 1.2

SITES = {"Gordes": (43.9116, 5.2003), "Strasbourg": (48.5819, 7.7510)}
OBJ_GORDES = {"BATIMENT0000000296689006", "BATIMENT0000000296689042",
              "BATIMENT0000000296689169"}


def remplir_emprise(labels, plans, plein, dans_emprise):
    """Étend les labels à toute l'emprise : plan voisin de moindre résidu."""
    ny, nx = labels.shape
    lab = np.where(dans_emprise, labels, -9)     # -9 : hors emprise
    X = np.arange(nx)[None, :] * PAS * np.ones((ny, 1))
    Y = np.arange(ny)[:, None] * PAS * np.ones((1, nx))
    for _ in range(nx + ny):
        js, is_ = np.nonzero(dans_emprise & (lab == -1))
        if not len(js):
            break
        bouge = False
        for j, i in zip(js, is_):
            mieux, res_min = -1, float("inf")
            for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                jj, ii = j + dj, i + di
                if 0 <= jj < ny and 0 <= ii < nx and lab[jj, ii] >= 0:
                    c = plans[lab[jj, ii]]["coef"]
                    r = abs(plein[j, i] - (c[0] * X[j, i] + c[1] * Y[j, i] + c[2]))
                    if r < res_min or (r == res_min and lab[jj, ii] < mieux):
                        mieux, res_min = int(lab[jj, ii]), r
            if mieux >= 0:
                lab[j, i] = mieux
                bouge = True
        if not bouge:
            break
    return np.where(lab == -9, -1, lab), X, Y


def polylignes_frontieres(lab):
    """Frontières entre labels (et avec l'extérieur, -1), chaînées par paire.

    Returns:
        liste de (label_a, label_b, [points 2D]) ; a < b, b = -1 pour le bord.
    """
    ny, nx = lab.shape
    segs = defaultdict(list)          # (a, b) -> [(p, q)]

    def bord(a, b, p, q):
        if a == b:
            return
        cle = (min(a, b), max(a, b)) if a >= 0 and b >= 0 else (max(a, b), -1)
        segs[cle].append((p, q))

    for j in range(ny):
        for i in range(nx):
            a = int(lab[j, i])
            if a < 0:
                continue
            x0, x1 = (i - 0.5) * PAS, (i + 0.5) * PAS
            y0, y1 = (j - 0.5) * PAS, (j + 0.5) * PAS
            bord(a, int(lab[j, i + 1]) if i + 1 < nx else -1, (x1, y0), (x1, y1))
            if i == 0 or lab[j, i - 1] < 0:
                bord(a, -1, (x0, y0), (x0, y1))
            bord(a, int(lab[j + 1, i]) if j + 1 < ny else -1, (x0, y1), (x1, y1))
            if j == 0 or lab[j - 1, i] < 0:
                bord(a, -1, (x0, y0), (x1, y0))

    def quant(p):
        return (round(p[0], 4), round(p[1], 4))

    polylignes = []
    for (a, b), liste in sorted(segs.items()):
        adj = defaultdict(list)
        for k, (p, q) in enumerate(liste):
            adj[quant(p)].append(k)
            adj[quant(q)].append(k)
        vus = set()
        departs = sorted(n for n, ks in adj.items() if len(ks) != 2)
        restants = sorted(set(range(len(liste))))
        while restants:
            dep = None
            for n in departs:
                ks = [k for k in adj[n] if k not in vus]
                if ks:
                    dep, k = n, ks[0]
                    break
            if dep is None:                       # boucle fermée
                k = restants[0]
                dep = quant(liste[k][0])
            chaine = [dep]
            noeud = dep
            while True:
                p, q = liste[k]
                suivant = quant(q) if quant(p) == noeud else quant(p)
                vus.add(k)
                restants.remove(k)
                chaine.append(suivant)
                noeud = suivant
                ks = [kk for kk in adj[noeud] if kk not in vus]
                if len(adj[noeud]) != 2 or not ks:
                    break
                k = ks[0]
            polylignes.append((a, b, chaine))
    return polylignes


def construire_brep(lab, plans, plein, dans_emprise):
    """B-Rep : toits par plan, coutures, murs, plancher. Métriques d'étanchéité."""
    lab, X, Y = remplir_emprise(lab, plans, plein, dans_emprise)
    if not (lab >= 0).any():
        return None
    ny, nx = lab.shape

    brutes = polylignes_frontieres(lab)
    lignes = []
    for a, b, pts in brutes:
        simple = list(LineString(pts).simplify(SIMPLIFIE_M).coords)
        lignes.append((a, b, simple))

    z_plan = lambda k, x, y: float(plans[k]["coef"] @ (x, y, 1.0))

    def cle2d(p):
        return (round(p[0] / SOUDURE_M), round(p[1] / SOUDURE_M))

    # --- Étape D : faîtage posé sur la droite analytique P_i ∩ P_j quand la
    # frontière mesurée en est proche. Sur la droite, z_i = z_j exactement :
    # les deux pans partagent leurs sommets, plus de couture à cet endroit.
    droites = {}
    for idx, (a, b, pts) in enumerate(lignes):
        if b < 0:
            continue
        ca, cb = plans[a]["coef"], plans[b]["coef"]
        da, db, dc = ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]
        n = math.hypot(da, db)
        if n < FAITAGE_PENTE_MIN:
            continue                      # décrochement : pas de droite utile
        da, db, dc = da / n, db / n, dc / n
        if float(np.median([abs(da * x + db * y + dc) for x, y in pts])) <= SNAP_DIST_M:
            droites[idx] = (da, db, dc)

    # Les extrémités sont partagées entre polylignes : chaque noeud reçoit UNE
    # nouvelle position (projection sur sa droite, ou intersection de ses
    # droites — le sommet d'une croupe), appliquée à toutes ses polylignes.
    incidences = defaultdict(list)
    for idx in droites:
        _, _, pts = lignes[idx]
        for noeud in (cle2d(pts[0]), cle2d(pts[-1])):
            incidences[noeud].append(droites[idx])
    remap = {}
    for noeud, ds in incidences.items():
        x0 = (noeud[0] * SOUDURE_M, noeud[1] * SOUDURE_M)
        if len(ds) >= 2:
            A = np.array([[d[0], d[1]] for d in ds])
            r = -np.array([d[2] for d in ds])
            sol, _, rang, _ = np.linalg.lstsq(A, r, rcond=None)
            if rang == 2 and math.dist(sol, x0) <= 3 * SNAP_DIST_M:
                remap[noeud] = (float(sol[0]), float(sol[1]))
                continue
        da, db, dc = ds[0]
        t = da * x0[0] + db * x0[1] + dc
        remap[noeud] = (x0[0] - t * da, x0[1] - t * db)

    ajustees, poses = [], 0
    for idx, (a, b, pts) in enumerate(lignes):
        pts = list(pts)
        for pos in (0, -1):
            pts[pos] = remap.get(cle2d(pts[pos]), pts[pos])
        if idx in droites:
            da, db, dc = droites[idx]
            pts = [pts[0]] + [(x - (da * x + db * y + dc) * da,
                               y - (da * x + db * y + dc) * db)
                              for x, y in pts[1:-1]] + [pts[-1]]
            pts = list(LineString(pts).simplify(SIMPLIFIE_M / 3).coords)
            poses += 1
        ajustees.append((a, b, pts))
    # Une ligne déplacée qui en croise une autre casserait polygonize (lignes
    # non nodées) : elle reprend sa géométrie mesurée, extrémités remappées.
    revertis = 0
    geoms = [LineString(p) for _, _, p in ajustees]
    for idx in sorted(droites):
        if any(k != idx and geoms[idx].crosses(g) for k, g in enumerate(geoms)):
            a, b, pts = lignes[idx]
            pts = list(pts)
            for pos in (0, -1):
                pts[pos] = remap.get(cle2d(pts[pos]), pts[pos])
            ajustees[idx] = (a, b, pts)
            geoms[idx] = LineString(pts)
            revertis += 1
            poses -= 1
    lignes = [(a, b, pts) for a, b, pts in ajustees
              if LineString(pts).length > SOUDURE_M]

    # Topologie unique : les lignes ajustées sont nodées ensemble (union) et
    # tout le reste — murs, plancher, niveaux — se dérive des faces produites,
    # pour que chaque arête soit partagée à l'identique par ses deux porteurs.
    faces = list(polygonize(shapely.unary_union(
        [LineString(p) for _, _, p in lignes])))

    sommets, index, triangles, etiquettes = [], {}, [], []

    def sid(x, y, z):
        cle = (round(x / SOUDURE_M), round(y / SOUDURE_M), round(z / SOUDURE_M))
        if cle not in index:
            index[cle] = len(sommets)
            sommets.append((x, y, z))
        return index[cle]

    def tri(etiquette, p1, p2, p3):
        """etiquette : index du plan pour un pan, -1 mur, -2 plancher."""
        a, b, c = sid(*p1), sid(*p2), sid(*p3)
        if a != b and b != c and a != c:
            triangles.append((a, b, c))
            etiquettes.append(etiquette)

    fans = [0]

    def oreilles(anneau):
        """Ear clipping d'un anneau simple 2D ; tous les sommets sont gardés.

        Sans oreille trouvable (colinéarités des contours en escalier), repli
        en éventail depuis le centroïde : les triangles peuvent se chevaucher
        sur une face non étoilée, mais ils sont coplanaires une fois relevés
        et chaque arête de bord reste portée une seule fois — l'étanchéité
        tient. Le compteur `fans` dit combien de faces y recourent.
        """
        pts = list(anneau)
        if sum((q[0] - p[0]) * (q[1] + p[1])
               for p, q in zip(pts, pts[1:] + pts[:1])) > 0:
            pts.reverse()
        tris, garde_fou = [], 0
        while len(pts) > 3 and garde_fou < 20000:
            garde_fou += 1
            n = len(pts)
            for k in range(n):
                p0, p1, p2 = pts[k - 1], pts[k], pts[(k + 1) % n]
                aire = ((p1[0] - p0[0]) * (p2[1] - p0[1])
                        - (p2[0] - p0[0]) * (p1[1] - p0[1]))
                if aire <= 1e-12:
                    continue
                if any(_dans_tri(p, p0, p1, p2) for m, p in enumerate(pts)
                       if m not in (k - 1 if k else n - 1, k, (k + 1) % n)
                       and p not in (p0, p1, p2)):
                    continue
                tris.append((p0, p1, p2))
                del pts[k]
                break
            else:
                break
        if len(pts) == 3:
            tris.append(tuple(pts))
        elif len(pts) > 3:
            fans[0] += 1
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            tris.extend(((cx, cy), pts[k], pts[(k + 1) % len(pts)])
                        for k in range(len(pts)))
        return tris

    delaunay_faces = 0
    faces_k = []
    for face in faces:
        # Label à la majorité des cellules couvertes : le point représentatif
        # d'une face mince retombe parfois sur la cellule du voisin.
        minx, miny, maxx, maxy = face.bounds
        i0f = max(int(math.floor(minx / PAS)), 0)
        i1f = min(int(math.ceil(maxx / PAS)) + 1, nx)
        j0f = max(int(math.floor(miny / PAS)), 0)
        j1f = min(int(math.ceil(maxy / PAS)) + 1, ny)
        k = -1
        if i0f < i1f and j0f < j1f:
            dedans = shapely.contains_xy(face, X[j0f:j1f, i0f:i1f], Y[j0f:j1f, i0f:i1f])
            labs = lab[j0f:j1f, i0f:i1f][dedans]
            labs = labs[labs >= 0]
            if labs.size:
                k = int(np.bincount(labs).argmax())
        if k < 0:
            p = face.representative_point()
            i = min(max(int(round(p.x / PAS)), 0), nx - 1)
            j = min(max(int(round(p.y / PAS)), 0), ny - 1)
            k = int(lab[j, i])
        if k < 0:
            continue
        faces_k.append((face, k))

    # Murs et coutures dérivés des faces, topologiquement : un segment soudé
    # appartient à exactement deux faces (couture) ou à une seule (pourtour).
    bowties, sondes_perdues = 0, 0
    bords = defaultdict(list)
    for fi, (face, k) in enumerate(faces_k):
        for ring in [face.exterior, *face.interiors]:
            pts_r = list(ring.coords)
            for p, q in zip(pts_r, pts_r[1:]):
                if math.dist(p, q) <= SOUDURE_M:
                    continue
                bords[frozenset((cle2d(p), cle2d(q)))].append((p, q, k))
    # Deux plans qui se croisent le long d'un segment (le gap change de signe)
    # imposent un sommet au croisement exact (z_i = z_j), inséré dans les DEUX
    # faces et dans le mur — sans lui, la couture laisse un sommet en T.
    croisements = {}
    segments = []
    for cle, liste in bords.items():
        p, q, ka = liste[0]
        if len(liste) == 1:
            segments.append((p, q, ka, -1))
            continue
        if len(liste) != 2:
            sondes_perdues += 1           # segment porté 3 fois : anomalie comptée
            continue
        kb = liste[1][2]
        if kb == ka:                      # même plan des deux côtés : rien à coudre
            continue
        g_p = z_plan(ka, *p) - z_plan(kb, *p)
        g_q = z_plan(ka, *q) - z_plan(kb, *q)
        if g_p * g_q < 0 and min(abs(g_p), abs(g_q)) > SOUDURE_M:
            t = g_p / (g_p - g_q)
            x = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
            croisements[cle] = x
            bowties += 1
            segments.append((p, x, ka, kb))
            segments.append((x, q, ka, kb))
        else:
            segments.append((p, q, ka, kb))

    # Triangulation des faces, points de croisement insérés dans les anneaux.
    def anneau_conforme(ring):
        pts_r = list(ring.coords)[:-1]
        out = []
        for m, p in enumerate(pts_r):
            q = pts_r[(m + 1) % len(pts_r)]
            out.append(p)
            if math.dist(p, q) > SOUDURE_M:
                x = croisements.get(frozenset((cle2d(p), cle2d(q))))
                if x is not None:
                    out.append(x)
        return out

    for face, k in faces_k:
        if face.interiors:
            delaunay_faces += 1
            pts = [c for r in [face.exterior, *face.interiors] for c in anneau_conforme(r)]
            dts = shapely.delaunay_triangles(shapely.MultiPoint(pts))
            tris2d = [list(t.exterior.coords)[:-1] for t in dts.geoms
                      if face.contains(t.representative_point())]
        else:
            tris2d = oreilles(anneau_conforme(face.exterior))
        for t in tris2d:
            tri(k, *[(x, y, z_plan(k, x, y)) for x, y in t])

    # Niveaux z par noeud : les côtés verticaux des murs s'y subdivisent,
    # sinon un mur voisin arrêté à mi-hauteur laisse un sommet en T.
    niveaux = defaultdict(set)
    for p, q, ka, kb in segments:
        for pt in (p, q):
            niveaux[cle2d(pt)].add(z_plan(ka, *pt))
            niveaux[cle2d(pt)].add(z_plan(kb, *pt) if kb >= 0 else 0.0)

    def echelle(pt, z_haut, z_bas):
        lo, hi = min(z_haut, z_bas), max(z_haut, z_bas)
        entre = [v for v in niveaux[cle2d(pt)] if lo + SOUDURE_M < v < hi - SOUDURE_M]
        return sorted(entre, reverse=(z_bas < z_haut))

    minceurs, d_faitage = [], []
    for p, q, ka, kb in segments:
        za_p, za_q = z_plan(ka, *p), z_plan(ka, *q)
        zb_p, zb_q = (z_plan(kb, *p), z_plan(kb, *q)) if kb >= 0 else (0.0, 0.0)
        if kb >= 0:
            ca, cb = plans[ka]["coef"], plans[kb]["coef"]
            da, db, dc = ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]
            pente = math.hypot(da, db)
            if pente >= FAITAGE_PENTE_MIN:        # vrai faîtage : droite définie
                minceurs.append((abs(za_p - zb_p) + abs(za_q - zb_q)) / 2)
                d_faitage.append((abs(da * p[0] + db * p[1] + dc)
                                  + abs(da * q[0] + db * q[1] + dc)) / (2 * pente))
        g_p, g_q = za_p - zb_p, za_q - zb_q
        if abs(g_p) <= SOUDURE_M and abs(g_q) <= SOUDURE_M:
            continue                      # faîtage posé : les pans se soudent
        if g_p * g_q < 0 and min(abs(g_p), abs(g_q)) > SOUDURE_M:
            bowties += 1                  # plans qui se croisent : compté, pas cousu
            continue
        anneau = ([(0.0, za_p), (1.0, za_q)]
                  + [(1.0, v) for v in echelle(q, za_q, zb_q)]
                  + [(1.0, zb_q), (0.0, zb_p)]
                  + [(0.0, v) for v in reversed(echelle(p, za_p, zb_p))])
        propre = [anneau[0]]
        for pt in anneau[1:]:
            if abs(pt[0] - propre[-1][0]) > 1e-9 or abs(pt[1] - propre[-1][1]) > SOUDURE_M:
                propre.append(pt)
        while len(propre) > 1 and (abs(propre[0][0] - propre[-1][0]) < 1e-9
                                   and abs(propre[0][1] - propre[-1][1]) <= SOUDURE_M):
            propre.pop()
        if len(propre) >= 3:
            for t3 in oreilles(propre):
                tri(-1, *[(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), z)
                          for t, z in t3])

    # Plancher : les segments de pourtour ferment le contour à z = 0.
    contour = list(polygonize(shapely.unary_union(
        [LineString([p, q]) for p, q, _, kb in segments if kb == -1])))
    for sol in contour:
        if sol.interiors:                 # cour intérieure : trous du plancher
            delaunay_faces += 1
            ptsf = [c for r in [sol.exterior, *sol.interiors] for c in r.coords[:-1]]
            dts = shapely.delaunay_triangles(shapely.MultiPoint(ptsf))
            tris2d = [list(t.exterior.coords)[:-1] for t in dts.geoms
                      if sol.contains(t.representative_point())]
        else:
            tris2d = oreilles(list(sol.exterior.coords)[:-1])
        for t in tris2d:
            tri(-2, *[(x, y, 0.0) for x, y in reversed(t)])

    # Étanchéité : chaque arête non dégénérée doit porter deux triangles.
    aretes = defaultdict(int)
    for a, b, c in triangles:
        for e in ((a, b), (b, c), (a, c)):
            aretes[tuple(sorted(e))] += 1
    ouvertes = sum(1 for n in aretes.values() if n != 2)

    # Écart au MNH redressé, sur les cellules de l'emprise.
    js, is_ = np.nonzero(dans_emprise & (lab >= 0))
    zp = np.array([z_plan(int(lab[j, i]), X[j, i], Y[j, i]) for j, i in zip(js, is_)])
    ecart = float(np.median(np.abs(plein[js, is_] - zp)))

    return {"sommets": len(sommets), "triangles": len(triangles),
            "ouvertes": ouvertes, "delaunay": delaunay_faces,
            "ecart": ecart, "faces": len(faces_k),
            "poses": poses, "revertis": revertis, "bowties": bowties,
            "fans": fans[0], "sondes": sondes_perdues,
            "minceur": float(np.median(minceurs)) if minceurs else None,
            "d_faitage": float(np.median(d_faitage)) if d_faitage else None,
            "mesh": (sommets, triangles, etiquettes)}


def _dans_tri(p, a, b, c):
    """Strictement intérieur : un point posé sur une arête ne bloque pas
    l'oreille — les contours en escalier alignent beaucoup de sommets."""
    e = 1e-9
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    return (d1 > e and d2 > e and d3 > e) or (d1 < -e and d2 < -e and d3 < -e)


def ecrire_obj(chemin, mesh):
    sommets, triangles, _ = mesh
    with open(chemin, "w") as f:
        for x, y, z in sommets:
            f.write(f"v {x:.3f} {z:.3f} {y:.3f}\n")   # y vers le haut (usage OBJ)
        for a, b, c in triangles:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def mesurer_site(nom, lat, lon, exporter=()):
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

    lignes_resume, ecartes = [], 0
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
        i0, i1, j0, j1 = fenetre["i0"], fenetre["i1"], fenetre["j0"], fenetre["j1"]
        bas = _sol_bas(sol_arr, geom, xs, ys, lon0, lat0, m_lon)
        if bas is None:
            continue
        masque = fenetre["valides"] & np.isfinite(fenetre["lisse"])
        z = np.where(masque,
                     fenetre["lisse"] + np.nan_to_num(sol_arr[j0:j1, i0:i1] - bas),
                     np.nan)
        debut = time.perf_counter()
        labels, plans, _, _ = pp.segmenter(z, masque, ANGLE, DIST)
        assigne = labels >= 0
        if not masque.any() or assigne.sum() / masque.sum() < COUVERTURE_MIN:
            ecartes += 1                       # repli : surface ou toit résumé
            continue
        plein = _combler(np.where(masque, z, np.nan))
        dans_emprise = shapely.contains_xy(poly_seul, fenetre["X"], fenetre["Y"])
        r = construire_brep(labels, plans, plein, dans_emprise)
        if r is None:
            ecartes += 1
            continue
        r["ms"] = (time.perf_counter() - debut) * 1000
        r["cleabs"] = cleabs
        r["dense"] = 2 * int(masque.sum())     # triangles du maillage actuel
        # Position dans la scène : origine de la fenêtre (mètres locaux, y vers
        # le nord dans xs/ys) et altitude de la base, pour poser les volumes.
        r["x0"] = float(xs[i0])
        r["y0"] = float(ys[j0])
        r["base"] = float(bas)
        lignes_resume.append(r)
        if cleabs in exporter:
            ecrire_obj(f"{DOSSIER}/brep-{cleabs}.obj", r["mesh"])
    return lignes_resume, ecartes


def pct(vals, t):
    return float(np.percentile(np.asarray(vals, dtype=np.float64), t))


def main():
    import json
    DOSSIER.mkdir(parents=True, exist_ok=True)
    sortie = ["# Prototype C/D : B-Rep étanche depuis les plans segmentés",
              f"Config « large » ({ANGLE:.0f}°, {DIST} m), bâtiments à couverture >= {COUVERTURE_MIN:.0%}."]
    visionneuse = {}
    for nom, (lat, lon) in SITES.items():
        rows, ecartes = mesurer_site(nom, lat, lon,
                                     exporter=OBJ_GORDES if nom == "Gordes" else ())
        visionneuse[nom] = [{
            "cleabs": r["cleabs"], "x0": round(r["x0"], 2), "y0": round(r["y0"], 2),
            "base": round(r["base"], 2), "faces": r["faces"],
            "ouvertes": r["ouvertes"], "ecart": round(r["ecart"], 2),
            "s": [round(v, 2) for xyz in r["mesh"][0] for v in xyz],
            "t": [i for abc in r["mesh"][1] for i in abc],
            "e": r["mesh"][2],
        } for r in rows]

        def p(s=""):
            print(s)
            sortie.append(s)

        etanches = sum(1 for r in rows if r["ouvertes"] == 0)
        p(f"\n## {nom} — {len(rows)} B-Rep construits, {ecartes} bâtiments en repli")
        p(f"- étanches (0 arête ouverte) : {etanches}/{len(rows)} "
          f"({100*etanches/max(len(rows),1):.0f} %) ; "
          f"arêtes ouvertes p90 {pct([r['ouvertes'] for r in rows], 90):.0f}, "
          f"max {max(r['ouvertes'] for r in rows)}")
        p(f"- triangles : p50 {pct([r['triangles'] for r in rows], 50):.0f} "
          f"(maillage dense actuel p50 {pct([r['dense'] for r in rows], 50):.0f}), "
          f"max {max(r['triangles'] for r in rows)}")
        p(f"- écart médian au MNH redressé : p50 {pct([r['ecart'] for r in rows], 50):.2f} m, "
          f"p90 {pct([r['ecart'] for r in rows], 90):.2f} m")
        minceurs = [r["minceur"] for r in rows if r["minceur"] is not None]
        dfait = [r["d_faitage"] for r in rows if r["d_faitage"] is not None]
        if minceurs:
            p(f"- couture de faîtage (|z_i − z_j| médian) : p50 {pct(minceurs,50):.2f} m, "
              f"p90 {pct(minceurs,90):.2f} m")
        if dfait:
            p(f"- distance du faîtage à la droite P_i ∩ P_j : p50 {pct(dfait,50):.2f} m, "
              f"p90 {pct(dfait,90):.2f} m")
        p(f"- faîtages posés sur leur droite analytique : {sum(r['poses'] for r in rows)} "
          f"(revenus pour croisement : {sum(r['revertis'] for r in rows)}) ; "
          f"coutures coupées à leur croisement de plans : {sum(r['bowties'] for r in rows)} ; "
          f"segments anormaux : {sum(r['sondes'] for r in rows)}")
        p(f"- faces en repli Delaunay (trous) : {sum(r['delaunay'] for r in rows)} ; "
          f"en éventail (oreilles bloquées) : {sum(r['fans'] for r in rows)}")
        p(f"- temps par bâtiment : p50 {pct([r['ms'] for r in rows],50):.0f} ms, "
          f"max {max(r['ms'] for r in rows):.0f} ms")
        if nom == "Gordes":
            p("\n### Pires cascades de la mesure initiale")
            for r in rows:
                if r["cleabs"] in OBJ_GORDES:
                    p(f"- {r['cleabs']} : {r['faces']} faces, {r['triangles']} tris, "
                      f"{r['ouvertes']} arêtes ouvertes, écart {r['ecart']:.2f} m "
                      f"→ brep-{r['cleabs']}.obj")
    with open(DOSSIER / "resultats-brep.md", "w") as f:
        f.write("\n".join(sortie) + "\n")
    with open(DOSSIER / "donnees-brep.json", "w") as f:
        json.dump(visionneuse, f, separators=(",", ":"))
    shutil.copy(Path(__file__).with_name("inspecteur-lod2.html"), DOSSIER)
    print(f"\nRésultats écrits dans {DOSSIER}/resultats-brep.md ; visionneuse :"
          f"\n  cd {DOSSIER} && python3 -m http.server 8123"
          "\n  http://localhost:8123/inspecteur-lod2.html")


if __name__ == "__main__":
    main()
