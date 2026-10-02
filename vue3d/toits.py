"""Toitures mesurées au LiDAR HD : gouttière et faîtage par bâtiment.

La BD TOPO fournit `altitude_minimale_toit` (gouttière) et `altitude_maximale_toit`
(faîtage), mais ces champs manquent sur une part des bâtiments — 31 % sur un
site méditerranéen — et sont bruités là où ils existent, dans les
deux sens (3,5 m de dénivelé annoncés pour 0,5 m mesurés, 0,9 pour 2,6).

Le MNH LiDAR HD donne la hauteur du sursol à 0,5 m de maille, toitures
comprises : à l'intérieur d'une emprise, ses cellules dessinent le vrai profil
du toit. Ce module en tire, par bâtiment, une gouttière (15e percentile), un
faîtage (85e percentile) et l'orientation du faîtage (axe principal des
cellules les plus hautes).

Trois précautions, toutes mesurées :

- L'emprise est érodée de 1,5 m : les cellules qui chevauchent le contour
  tirent le bas du profil vers le sol (0,2 m lu sur une maison de 5 m).
- Les percentiles, et non min/max : un arbre qui surplombe gonfle le maximum
  (16 m lus sur une maison de 4 m).
- La maille : à 2 m — le réglage de la végétation — 9 toits sur 16 seulement
  sont lisibles une fois l'emprise érodée ; à 0,5 m, 14.

Hors couverture LiDAR HD, le repli MNS − MNT sert de la même façon, avec
`source` pour le dire : il lisse les cimes et donc les faîtages.

Les arbres, ensuite. Mesuré sur un terrain boisé du sud de la France :

- Une annexe de 12 m² entièrement sous la canopée lit un plateau de 12 m,
  régulier, qui passe tous les tests de forme. Seule l'orthophoto le
  contredit : deux tiers de l'emprise sont verts, contre 0 à 2 % sur les
  toits voisins. D'où `sous_couvert`, posé d'après la part verte de
  l'emprise, qui l'emporte sur la forme du profil.
- Une remise de 32 m² à moitié sous un arbre donne un profil bimodal : un
  mode bas vers 2,7 m (le toit), un mode haut vers 11 m (le feuillage). Le
  test de pente rejette la mesure, à raison, mais le repli BD TOPO annonce
  8,2 m de murs : ses altitudes de toit viennent d'un MNS photogrammétrique,
  contaminé par les mêmes arbres. D'où le mode bas, retenu comme toiture
  quand le profil est bimodal.
"""

import concurrent.futures
import logging
import math
import multiprocessing
import os
import sys
import threading
import time

import numpy as np
import shapely
from shapely.geometry import shape
from shapely.ops import transform

from .ortho import EXG_SEUIL
from .pans import pans_du_toit, proches

journal = logging.getLogger(__name__)

TOITS_RESOLUTION_M = 0.5
TOITS_EROSION_M = 1.5
# En deçà, le profil n'est pas lisible : 20 cellules à 0,5 m font 5 m².
TOITS_MIN_CELLULES = 20
# Rapport des variances de l'axe principal sur l'axe secondaire : en dessous,
# les cellules hautes ne dessinent pas de ligne et l'orientation est du bruit.
TOITS_NETTETE_MIN = 3.0
# Au-delà de cet écart entre le 85e et le 95e percentile, le haut du profil
# n'est plus un toit : un arbre qui surplombe, une antenne, un bâtiment voisin
# plus haut dont les cellules débordent. Mesuré : 16 m lus sur une maison de
# 4 m (p85 à 9,7, p95 bien au-delà). Un toit, même à croupe, a un p95 proche
# de son p85.
TOITS_ECART_SUSPECT_M = 2.0
# Pente maximale plausible d'un toit à deux pans : ~50°, soit une montée de
# 0,6 fois la largeur du bâtiment (tan 50° x demi-largeur). Au-delà, le haut
# du profil n'est pas le toit. Ce garde-fou attrape ce que l'écart p85–p95
# manque : un arbre qui couvre plus de 15 % de l'emprise met le p85 lui-même
# dans le feuillage (mesuré : 9,9 m de « dénivelé » sur une annexe de 7 m).
TOITS_PENTE_MAX = 0.6
# Part de l'emprise que l'orthophoto voit verte (ExG >= seuil) au-delà de
# laquelle le LiDAR mesure une canopée et non un toit. Mesuré : 0 à 2 % sur
# trois toits réels d'un terrain boisé, 66 % sur l'annexe sous les arbres.
TOITS_COUVERT_PART_VERTE = 0.33
# En dessous de cette part verte, un profil rejeté ne doit rien aux arbres
# (voisin plus haut, antenne, ortho d'une autre date) : le repli BD TOPO garde
# son sens. Au-dessus, la BD TOPO est contaminée comme le LiDAR.
TOITS_ARBRES_PART_VERTE = 0.10
# Profil bimodal : le mode bas doit couvrir au moins cette part de l'emprise
# pour être un toit et non le sol vu entre les branches.
TOITS_MODE_BAS_PART_MIN = 0.25
# Creux entre les deux modes : la densité de cellules au fond du creux doit
# tomber sous cette fraction du pic BAS, celui du toit. Un versant continu
# (densité à peu près uniforme) n'a pas de creux et n'est donc pas coupé en
# deux. Se comparer au plus petit des deux pics laissait passer la comparaison
# à côté quand le mode haut est une traîne clairsemée : mesuré sur un hangar
# de 104 m² dont 120 cellules sont à 3 m et la traîne d'arbres compte 2 à 8
# cellules par mètre, le creux à 3 cellules dépassait 0,35 x 8.
TOITS_MODE_BAS_CREUX = 0.35


def _largeur_min(polygone_m):
    """Petit côté du rectangle englobant orienté, en mètres.

    shapely émet des avertissements numériques (division par zéro, valeur
    invalide) sur des emprises alignées aux axes et peut renvoyer un rectangle
    dégénéré ; un NaN ici rendrait la comparaison de pente fausse et la mesure
    « non fiable » à tort. On retombe alors sur la boîte alignée.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            c = list(polygone_m.minimum_rotated_rectangle.exterior.coords)
            largeur = min(math.dist(c[i], c[i + 1]) for i in range(4))
        except Exception:
            largeur = float("nan")
    if not math.isfinite(largeur) or largeur <= 0:
        minx, miny, maxx, maxy = polygone_m.bounds
        largeur = min(maxx - minx, maxy - miny)
    return largeur


def _percentile(valeurs_triees, t):
    return valeurs_triees[min(len(valeurs_triees) - 1, int(t * len(valeurs_triees)))]


def _coupure_bimodale(hauteurs):
    """Seuil qui sépare un toit du feuillage qui le recouvre en partie.

    Seuil d'Otsu (variance inter-classes maximale) sur les hauteurs triées,
    puis trois garde-fous : le mode bas couvre au moins un quart des cellules,
    les deux modes sont à plus de TOITS_ECART_SUSPECT_M l'un de l'autre, et il
    y a un vrai creux entre eux — la densité autour du seuil tombe sous
    TOITS_MODE_BAS_CREUX fois le plus petit pic. Sans creux, c'est un versant
    continu, pas deux niveaux.

    Returns:
        le seuil (float) ou None si le profil n'est pas bimodal.
    """
    n = len(hauteurs)
    if n < TOITS_MIN_CELLULES:
        return None
    # Sommes cumulées pour la variance inter-classes en une passe par seuil.
    meilleur, seuil = -1.0, None
    somme = sum(hauteurs)
    cumul, cumul_n = 0.0, 0
    pas = 0.5
    prochain = hauteurs[0] + pas
    for i, h in enumerate(hauteurs):
        # Coupure juste sous h : le mode bas est ce qui précède, h exclu.
        if h >= prochain and 0 < cumul_n < n:
            prochain = h + pas
            nb, nh = cumul_n, n - cumul_n
            mb, mh = cumul / nb, (somme - cumul) / nh
            inter = nb * nh * (mh - mb) ** 2
            if inter > meilleur:
                meilleur, seuil = inter, hauteurs[i - 1]
        cumul += h
        cumul_n += 1
    if seuil is None:
        return None
    # Histogramme au mètre : un pic de chaque côté du seuil d'Otsu, et entre
    # les deux pics un creux. La coupure retenue est le haut du bac le plus
    # creux, pas le seuil d'Otsu, qui tombe au bord du mode bas plutôt qu'au
    # milieu du vide.
    comptes = {}
    for h in hauteurs:
        comptes[math.floor(h)] = comptes.get(math.floor(h), 0) + 1
    bac_seuil = math.floor(seuil)
    pic_bas = max((b for b in comptes if b <= bac_seuil), key=comptes.get)
    hauts = [b for b in comptes if b > bac_seuil]
    if not hauts:
        return None
    pic_haut = max(hauts, key=comptes.get)
    entre = list(range(pic_bas + 1, pic_haut))
    if not entre:
        return None
    bac_creux = min(entre, key=lambda b: comptes.get(b, 0))
    if comptes.get(bac_creux, 0) > TOITS_MODE_BAS_CREUX * comptes[pic_bas]:
        return None
    coupure = float(bac_creux + 1)
    # Les garde-fous portent sur la coupure RETENUE, pas sur celle d'Otsu :
    # les deux diffèrent, et valider l'une pour renvoyer l'autre laissait
    # passer un mode bas de 8 cellules (2 m²) sur une cabane de 11 m²
    # entièrement sous les arbres.
    bas = [h for h in hauteurs if h <= coupure]
    haut = [h for h in hauteurs if h > coupure]
    if not haut or len(bas) < max(TOITS_MODE_BAS_PART_MIN * n, TOITS_MIN_CELLULES):
        return None
    if _percentile(haut, 0.5) - _percentile(bas, 0.85) < TOITS_ECART_SUSPECT_M:
        return None
    return coupure


def profil_toit(cellules, polygone_m, erosion_m=TOITS_EROSION_M, largeur_min=None):
    """Gouttière, faîtage et orientation du faîtage d'un bâtiment.

    Args:
        cellules: (x, y, h) en mètres locaux, h = hauteur de sursol : liste
            de triplets, ou tableau n × 3 (`_cellules_de_la_boite`).
        polygone_m: emprise shapely en mètres locaux, même repère.

    Returns:
        dict(gouttiere, faitage, denivele, axe_deg, nettete, n) ou None si
        l'emprise érodée ne contient pas assez de cellules. `axe_deg` est
        l'angle du faîtage depuis l'est, dans [0, 180[, ou None si la mesure
        n'est pas nette.
    """
    tableau = np.asarray(cellules, dtype=np.float64).reshape(-1, 3)
    cx, cy = tableau[:, 0], tableau[:, 1]

    def eroder(marge):
        inner = polygone_m.buffer(-marge)
        # Petite annexe : on érode moins plutôt que de renoncer.
        return inner if not inner.is_empty else polygone_m.buffer(-marge / 3)

    def contenu(inner):
        """Les cellules dont le centre est dans `inner`, dans leur ordre.

        Même prédicat que `inner.contains(Point(x, y))`, d'un seul appel
        vectorisé : un Point shapely par cellule, c'était 4 millions de
        Points et 29 s sur 50 de toitures (sous cProfile) à Strasbourg en
        zone de 1 000 m, 1 376 bâtiments. Mêmes cellules dans le même ordre,
        vérifié appel par appel sur les entrées rejouées de quatre lieux.
        """
        if inner.is_empty:
            return []
        minx, miny, maxx, maxy = inner.bounds
        boite = np.flatnonzero((minx <= cx) & (cx <= maxx) & (miny <= cy) & (cy <= maxy))
        garde = boite[shapely.contains_xy(inner, cx[boite], cy[boite])]
        # tolist : des float Python, comme avant — les sommes de l'ACP plus
        # bas (sum() compensée de Python 3.12) en dépendent au bit près.
        return tableau[garde].tolist()

    # Deux érosions. La gouttière est au bord : érodée de 1,5 m, elle est
    # mesurée 1,5 m à l'intérieur du versant et remonte (mesuré : +0,45 m sur
    # une pente de 3 m pour 6 m de demi-largeur). À 1 m on garde la protection
    # contre les cellules de sol qui chevauchent le contour, en perdant moins.
    dedans = contenu(eroder(erosion_m))
    if len(dedans) < TOITS_MIN_CELLULES:
        # Petite emprise dont l'érosion laisse un ruban de quelques cellules
        # (une remise de 32 m² : 5 × 6 m, 2 × 3 m une fois érodée) : on érode
        # moins plutôt que de renoncer. Mesuré : sans ce repli, la remise
        # n'avait pas de profil et le rendu tombait sur la BD TOPO.
        dedans = contenu(eroder(erosion_m / 3))
        if len(dedans) < TOITS_MIN_CELLULES:
            return None
    bord = contenu(eroder(min(erosion_m, 1.0))) or dedans

    hauteurs = sorted(h for _, _, h in dedans)
    gouttiere = _percentile(sorted(h for _, _, h in bord), 0.15)
    faitage = _percentile(hauteurs, 0.85)
    p95 = _percentile(hauteurs, 0.95)
    # Fiabilité : le rendu préfère le LiDAR à la BD TOPO, mais seulement quand
    # la mesure ressemble à un toit. Sinon il retombe sur la BD TOPO.
    if largeur_min is None:
        largeur_min = _largeur_min(polygone_m)
    fiable = ((p95 - faitage) <= TOITS_ECART_SUSPECT_M
              and (faitage - gouttiere) <= TOITS_PENTE_MAX * max(largeur_min, 1.0))

    # Profil bimodal : un toit sous un arbre qui en couvre une partie. Le mode
    # bas (cellules à moins de 2 m de la gouttière) est le toit, le mode haut
    # le feuillage. On ne l'évalue que sur un profil rejeté : sur un profil
    # plausible, deux niveaux sont deux corps de bâtiment, pas un arbre.
    mode_bas = False
    part_haute = 0.0
    if not fiable:
        coupure = _coupure_bimodale(hauteurs)
        if coupure is not None:
            bas = [h for h in hauteurs if h <= coupure]
            part_haute = 1 - len(bas) / len(hauteurs)
            hauteurs = bas
            gouttiere = _percentile(bas, 0.15)
            faitage = _percentile(bas, 0.85)
            p95 = _percentile(bas, 0.95)
            mode_bas = True
            fiable = (faitage - gouttiere) <= TOITS_PENTE_MAX * max(largeur_min, 1.0)
            dedans = [(x, y, h) for x, y, h in dedans if h <= coupure]

    # Orientation : axe principal (ACP 2D) du quart supérieur des cellules.
    seuil = _percentile(hauteurs, 0.75)
    hautes = [(x, y) for x, y, h in dedans if h >= seuil]
    axe_deg, nettete = None, None
    # Un faîtage est une ligne, donc une minorité de cellules. Sur un toit plat
    # toutes sont « hautes » : l'axe principal serait celui de l'emprise, et un
    # bâtiment allongé passerait le test de netteté sans avoir de faîtage.
    if len(hautes) >= 3 and len(hautes) <= 0.5 * len(dedans):
        mx = sum(x for x, _ in hautes) / len(hautes)
        my = sum(y for _, y in hautes) / len(hautes)
        sxx = sum((x - mx) ** 2 for x, _ in hautes)
        syy = sum((y - my) ** 2 for _, y in hautes)
        sxy = sum((x - mx) * (y - my) for x, y in hautes)
        # Valeurs propres de la matrice de covariance, pour le rapport
        # grand axe / petit axe indépendamment de l'orientation.
        tr, det = sxx + syy, sxx * syy - sxy * sxy
        disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
        l1, l2 = tr / 2 + disc, tr / 2 - disc
        nettete = round(l1 / l2, 1) if l2 > 1e-9 else 1e9
        if nettete >= TOITS_NETTETE_MIN:
            axe_deg = round(math.degrees(0.5 * math.atan2(2 * sxy, sxx - syy)) % 180, 1)

    profil = {
        "gouttiere": round(gouttiere, 1),
        "faitage": round(faitage, 1),
        "denivele": round(faitage - gouttiere, 1),
        "p95": round(p95, 1),
        "fiable": fiable,
        # Toit lu dans le mode bas d'un profil bimodal ; part_haute dit quelle
        # part de l'emprise le feuillage recouvre.
        "mode_bas": mode_bas,
        "part_haute": round(part_haute, 2),
        "axe_deg": axe_deg,
        "nettete": None if nettete is None else (nettete if nettete < 1e6 else None),
        "n": len(dedans),
    }
    corps = corps_de_toit(dedans, seuil)
    if len(corps) >= 2:
        profil["corps"] = corps
    return profil


# Une composante de faîtage plus petite est du bruit (un chien-assis, une
# cheminée, un arbre qui déborde) : 20 cellules à 0,5 m = 5 m².
CORPS_MIN_CELLULES = 20
# Part minimale de l'emprise pour qu'un corps ait son propre toit.
CORPS_PART_MIN = 0.08


def _composantes(points_grille):
    """Composantes connexes (8-voisinage) d'un ensemble de cellules entières."""
    restants = set(points_grille)
    comps = []
    while restants:
        depart = restants.pop()
        pile, comp = [depart], [depart]
        while pile:
            a = pile.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    b = (a[0] + dx, a[1] + dy)
                    if b in restants:
                        restants.discard(b)
                        pile.append(b)
                        comp.append(b)
        comps.append(comp)
    return comps


def _axe_et_nettete(points):
    """Axe principal (deg depuis l'est, [0, 180[) et rapport des variances."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    syy = sum((y - my) ** 2 for _, y in points)
    sxy = sum((x - mx) * (y - my) for x, y in points)
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    nettete = l1 / l2 if l2 > 1e-9 else 1e9
    axe = math.degrees(0.5 * math.atan2(2 * sxy, sxx - syy)) % 180
    return axe, nettete, (mx, my)


# Deux faîtages à moins de cet écart relatif du plus proche sont à égalité
# pour numpy : la cellule est alors départagée par le calcul d'origine. Le
# carré de numpy (x * x) et celui de Python (x ** 2, le pow() de la libm)
# diffèrent d'une unité du dernier bit sur 2 635 nombres tirés sur 2 millions
# (macOS) ; une distance au carré en porte donc une erreur relative de
# l'ordre de 1e-15, mille fois sous ce seuil.
CORPS_EGALITE_REL = 1e-12
# Cellules traitées d'un bloc : la matrice des distances (cellules × points
# d'un faîtage) et chacun de ses temporaires restent sous 8 Mo. Un faîtage
# sous-échantillonné par f[::max(1, len(f) // 60)] garde jusqu'à 119 points
# (119 cellules, pas de 1) : 8 192 × 119 × 8 octets = 7,8 Mo.
CORPS_BLOC = 8192


def _faitage_le_plus_proche(dedans, echant):
    """Index du faîtage le plus proche de chaque cellule (x, y, h).

    Le même que `min(range(K), key=lambda k: min((x - a) ** 2 + (y - b) ** 2
    for a, b in echant[k]))`, le premier en cas d'égalité, mais en numpy :
    cellule par cellule, c'était 7 s sur 50 de toitures (sous cProfile) à
    Strasbourg en zone de 1 000 m. Un écart d'arrondi entre les deux calculs
    ne peut inverser que deux faîtages presque à égalité : ces cellules-là
    sont recalculées comme avant (CORPS_EGALITE_REL).
    """
    n = len(dedans)
    if n == 0:
        return []
    xy = np.array([(c[0], c[1]) for c in dedans], dtype=np.float64)
    points = [np.asarray(e, dtype=np.float64).reshape(-1, 2) for e in echant]
    d2 = np.empty((len(echant), n))
    for debut in range(0, n, CORPS_BLOC):
        x = xy[debut:debut + CORPS_BLOC, 0:1]
        y = xy[debut:debut + CORPS_BLOC, 1:2]
        for k, p in enumerate(points):
            dx, dy = x - p[:, 0], y - p[:, 1]
            d2[k, debut:debut + CORPS_BLOC] = (dx * dx + dy * dy).min(axis=1)
    choix = d2.argmin(axis=0)
    if len(echant) >= 2:
        deux = np.partition(d2, 1, axis=0)
        for i in np.flatnonzero(deux[1] - deux[0] <= CORPS_EGALITE_REL * deux[0]).tolist():
            x, y = dedans[i][0], dedans[i][1]
            choix[i] = min(range(len(echant)),
                           key=lambda k: min((x - a) ** 2 + (y - b) ** 2 for a, b in echant[k]))
    return choix.tolist()


def corps_de_toit(dedans, seuil_haut, pas=TOITS_RESOLUTION_M):
    """Découpe un bâtiment en corps de toit d'après ses faîtages.

    Une maison en ailes a plusieurs faîtages, que la BD TOPO ignore : elle ne
    donne qu'une gouttière et un faîtage pour toute l'emprise, et le rendu
    posait un seul toit à deux pans sur le rectangle englobant — faux sur
    63 % des bâtiments de plus de 120 m² mesurés (rectangularité < 0,85).

    Méthode : les cellules du quart supérieur forment des composantes connexes,
    une par faîtage ; chaque cellule de l'emprise est rattachée au faîtage le
    plus proche (partition de Voronoï) ; chaque corps reçoit son profil et sa
    boîte le long de son propre axe. Un corps dont les cellules hautes ne
    dessinent pas de ligne (netteté < 3) est une croupe ou une pyramide :
    `axe_deg` vaut None et le rendu en fait une pyramide.

    Returns:
        liste de corps (vide si un seul faîtage) : cx, cy en mètres locaux,
        axe_deg, nettete, gouttiere, faitage, denivele, longueur, largeur
        (boîte le long de l'axe), part (fraction des cellules).
    """
    hautes = {(round(x / pas), round(y / pas)) for x, y, h in dedans if h >= seuil_haut}
    comps = [c for c in _composantes(hautes) if len(c) >= CORPS_MIN_CELLULES]
    if len(comps) < 2:
        return []
    faitages = [[(a * pas, b * pas) for a, b in c] for c in comps]
    # Sous-échantillon des faîtages pour la distance : environ 60 points
    # suffisent (119 au plus selon la longueur du faîtage, CORPS_BLOC).
    echant = [f[::max(1, len(f) // 60)] for f in faitages]
    choix = _faitage_le_plus_proche(dedans, echant)
    groupes = [[] for _ in comps]
    for c, k in zip(dedans, choix):
        groupes[k].append(c)

    corps = []
    for f, cellules in zip(faitages, groupes):
        if len(cellules) < CORPS_PART_MIN * len(dedans):
            continue
        axe, nettete, _ = _axe_et_nettete(f)
        hs = sorted(h for _, _, h in cellules)
        gout, fait = _percentile(hs, 0.15), _percentile(hs, 0.85)
        # Boîte du corps dans le repère de son axe (ou de l'emprise si flou).
        a = math.radians(axe if nettete >= TOITS_NETTETE_MIN else 0.0)
        cos, sin = math.cos(-a), math.sin(-a)
        xs = [x * cos - y * sin for x, y, _ in cellules]
        ys = [x * sin + y * cos for x, y, _ in cellules]
        cx = sum(x for x, _, _ in cellules) / len(cellules)
        cy = sum(y for _, y, _ in cellules) / len(cellules)
        corps.append({
            "cx": round(cx, 2), "cy": round(cy, 2),
            "axe_deg": round(axe, 1) if nettete >= TOITS_NETTETE_MIN else None,
            "nettete": round(min(nettete, 999.0), 1),
            "gouttiere": round(gout, 1), "faitage": round(fait, 1),
            "denivele": round(fait - gout, 1),
            # + une cellule de chaque côté : la boîte des centres de cellules
            # s'arrête une demi-maille avant le bord réel.
            "longueur": round(max(xs) - min(xs) + pas, 1),
            "largeur": round(max(ys) - min(ys) + pas, 1),
            "part": round(len(cellules) / len(dedans), 2),
        })
    return corps if len(corps) >= 2 else []


# --- Surface du toit ------------------------------------------------------
# Le toit à deux pans par corps résume mal une maison dont une moitié est
# surélevée, ou dont le faîtage n'est pas au milieu : il en garde une gouttière
# et un faîtage. Le MNH, lui, voit les pans et les marches. Pour chaque toit
# fiable, la scène embarque donc la grille des hauteurs sous l'emprise, que la
# vue découpe sur le contour.
#
# Cellules retenues. Mesuré sur 78 toits fiables à Gordes et 106 à Strasbourg,
# part des cellules qui lisent le sol (moins de la moitié de la gouttière)
# selon la distance au contour :
#
#                  0 m     0,25    0,5     0,75    1 m et au-delà
#   Gordes        9,4 %    5,1    4,0     2,2     ~1,5 %
#   Strasbourg    4,9 %    4,1    2,9     2,4     ~1,5 %
#
# Au bord, une cellule mêle toit et sol. À 0,75 m on rejoint le palier, fait
# de cours et de terrasses que le seuil de hauteur écarte ensuite. Les
# cellules écartées, et celles du bord, reprennent la moyenne de leurs voisines
# retenues, de proche en proche.
SURFACE_RETRAIT_M = 0.75
# Sous cette fraction de la gouttière, une cellule lit le sol (cour, terrasse).
SURFACE_SOL_PART = 0.5
#
# Encodage. Mesuré sur Strasbourg (111 surfaces, 215 000 cellules), poids
# ajouté à la scène compressée :
#
#   int16 en base64                              140 Ko
#   entiers en liste JSON                        112 Ko
#   écarts entre voisines, liste JSON             97 Ko
#   … et cellules inutiles répétées               71 Ko   (Gordes : 35 Ko)
#
# Un toit est lisse : l'écart d'une cellule à sa voisine tient presque
# toujours dans quelques décimètres, que gzip compresse bien mieux que des
# hauteurs. La fenêtre est la boîte de l'emprise : un bâtiment en biais en
# laisse plus de la moitié hors du contour (55 % des cellules à Strasbourg),
# et la vue n'en lit que celles dont le carré touche l'emprise. Les autres
# répètent leur voisine : un écart nul ne coûte presque rien.


def _encoder_ecarts(q, utile):
    """Écarts entre cellules voisines, en liste, les cellules inutiles répétées.

    Première valeur absolue ; ensuite, dans une ligne, l'écart à la cellule de
    gauche, et en tête de ligne l'écart à la tête de la ligne précédente.
    """
    q = q.astype(np.int64)
    ny, nx = q.shape
    # Chaque cellule inutile prend la dernière utile à sa gauche ; en tête de
    # ligne, faute de gauche, la tête de la ligne précédente.
    for j in range(ny):
        if not utile[j, 0]:
            q[j, 0] = q[j - 1, 0] if j else q[j, 0]
    idx = np.where(utile, np.arange(nx)[None, :], 0)
    idx[:, 0] = 0
    np.maximum.accumulate(idx, axis=1, out=idx)
    q = np.take_along_axis(q, idx, axis=1)
    ecarts = np.diff(q, axis=1, prepend=0)
    ecarts[1:, 0] = np.diff(q[:, 0])
    return ecarts.ravel().tolist()


def _mediane_3x3(grille, valides):
    """Médiane 3 × 3 des seules cellules retenues ; NaN ailleurs.

    Gomme une cheminée ou une antenne d'une cellule sans arrondir une marche :
    de part et d'autre d'une arête, la majorité du voisinage est du même côté.

    La médiane de np.nanmedian, calculée comme lui — le tri, puis la moyenne
    des deux cellules du milieu, (bas + haut) / 2 — sans ses tableaux
    masqués ni son avertissement par cellule toute en NaN : 1,36 s -> 0,17 s
    sur les 732 fenêtres de Strasbourg en zone de 1 000 m (médianes de six
    mesures alternées, machine chargée), mêmes valeurs au bit près.

    Précondition de cette égalité : aucune cellule valide ne vaut −0,0 ni
    NaN. cellules_du_toit la tient : une cellule valide a h ≥
    SURFACE_SOL_PART × gouttière, comparaison que NaN ne passe pas, et la
    gouttière, percentile de hauteurs du MNH positives et arrondies au
    décimètre (vue3d/mnh.py), vaut au moins 0,1 m. Hors de là, le résultat
    s'écarte de nanmedian : −0,0 au lieu de 0,0 quand les deux cellules du
    milieu sont des −0,0, NaN placés autrement quand une cellule valide est
    NaN (sur 3 000 grilles tirées au sort, 1 467 et 429 écarts ; aucun avec
    des +0,0, des négatifs ou des infinis).
    """
    g = np.where(valides, grille, np.nan)
    p = np.pad(g, 1, constant_values=np.nan)
    ny, nx = g.shape
    pile = np.stack([p[dj:dj + ny, di:di + nx] for dj in range(3) for di in range(3)])
    pile.sort(axis=0)                                 # les NaN en dernier
    n = 9 - np.isnan(pile).sum(axis=0)
    bas = np.take_along_axis(pile, (np.maximum(n - 1, 0) // 2)[None], axis=0)[0]
    haut = np.take_along_axis(pile, (n // 2)[None], axis=0)[0]
    return np.where(valides, (bas + haut) / 2.0, np.nan)


# Les huit voisines d'une cellule, dans l'ordre où _combler les somme : cet
# ordre fixe l'arrondi de la somme.
_HUIT = [(dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]


def _combler(grille):
    """Remplit les NaN par la moyenne des voisins connus, de proche en proche.

    À chaque passe, une cellule inconnue qui touche une connue prend la
    moyenne de ses voisines connues, toutes calculées sur l'état d'avant la
    passe. Seules les cellules qui touchent celles remplies à la passe
    précédente peuvent l'être à la suivante : ce sont elles seules qu'on
    évalue, et non toute la fenêtre à chaque passe. La somme reste celle de
    numpy sur la pile des huit voisines — 0 + v0 + v1 + … + v7, de gauche à
    droite, une voisine inconnue comptant 0 — vérifiée égale au bit près
    sur 3 000 fenêtres tirées au sort. Mesuré sur les 772 fenêtres de
    Strasbourg en zone de 1 000 m : 1,9 s -> 0,8 s.
    """
    ny, nx = grille.shape
    W = nx + 2
    g = np.pad(grille, 1, constant_values=np.nan)
    plat = g.ravel()
    decalages = np.array([dj * W + di for dj, di in _HUIT])[:, None]
    interieur = np.zeros(g.shape, dtype=bool)
    interieur[1:-1, 1:-1] = True
    interieur = interieur.ravel()
    js, is_ = np.nonzero(np.isnan(grille))
    candidates = (js + 1) * W + is_ + 1
    while candidates.size:
        voisins = plat[candidates + decalages]
        connus = ~np.isnan(voisins)
        termes = np.where(connus, voisins, 0.0)
        somme = 0.0 + termes[0]
        for t in termes[1:]:
            somme = somme + t
        n = connus.sum(axis=0)
        ok = n > 0
        if not ok.any():
            break
        remplies = candidates[ok]
        plat[remplies] = somme[ok] / n[ok]
        autour = np.unique((remplies + decalages).ravel())
        candidates = autour[interieur[autour] & np.isnan(plat[autour])]
    return g[1:-1, 1:-1].copy()


def cellules_du_toit(polygone_m, hauteurs, xs, ys, gouttiere, verdure=None):
    """Fenêtre de la grille MNH sous une emprise, et ses cellules lisibles.

    Args:
        polygone_m: emprise en mètres locaux.
        hauteurs: grille MNH (numpy, lignes depuis le nord) ; xs, ys : mètres
            locaux de ses colonnes (croissants) et de ses lignes (décroissants).
        gouttiere: gouttière du profil, pour écarter les cellules de sol.
        verdure: grille ExG alignée sur `hauteurs`, ou None.

    Returns:
        dict(i0, i1, j0, j1, X, Y, valides, lisse) — `lisse` est la médiane
        3 × 3 des cellules retenues, NaN ailleurs — ou None si l'emprise
        déborde de la grille ou si trop peu de cellules sont lisibles.
    """
    minx, miny, maxx, maxy = polygone_m.bounds
    # Une colonne de centres de chaque côté au-delà de l'emprise : la vue
    # interpole entre centres, il lui faut encadrer le contour.
    i0 = max(int(np.searchsorted(xs, minx)) - 1, 0)
    i1 = min(int(np.searchsorted(xs, maxx, side="right")) + 1, len(xs))
    j0 = max(int(np.searchsorted(-ys, -maxy)) - 1, 0)
    j1 = min(int(np.searchsorted(-ys, -miny, side="right")) + 1, len(ys))
    if i1 - i0 < 2 or j1 - j0 < 2:
        return None
    # Une emprise qui déborde de la grille n'aurait qu'une partie de son toit.
    # La scène découpe ses bâtiments en retrait du bord (vue3d/batiments.py)
    # pour que la grille les encadre : ce refus n'attrape plus que l'appelant
    # qui ne l'aurait pas fait.
    if xs[i0] > minx or xs[i1 - 1] < maxx or ys[j0] < maxy or ys[j1 - 1] > miny:
        return None
    X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1])
    h = np.asarray(hauteurs[j0:j1, i0:i1], dtype=np.float64)
    # Quatre conditions par cellule : les deux tests de valeur d'abord, puis
    # le contour sur ce qui reste, et la distance au bord, la plus chère,
    # sur les seules cellules dedans. Chaque test est celui d'avant, cellule
    # par cellule : 1,05 s -> 0,43 s sur les 738 fenêtres de Strasbourg en
    # zone de 1 000 m (médiane 3×3 non comprise), mêmes cellules retenues.
    retenues = h >= SURFACE_SOL_PART * gouttiere
    if verdure is not None:
        retenues &= verdure[j0:j1, i0:i1] < EXG_SEUIL
    js, is_ = np.nonzero(retenues)
    dedans = shapely.contains_xy(polygone_m, X[js, is_], Y[js, is_])
    js, is_ = js[dedans], is_[dedans]
    loin_du_bord = shapely.distance(shapely.points(X[js, is_], Y[js, is_]),
                                    polygone_m.boundary) >= SURFACE_RETRAIT_M
    valides = np.zeros(X.shape, dtype=bool)
    valides[js[loin_du_bord], is_[loin_du_bord]] = True
    if valides.sum() < TOITS_MIN_CELLULES:
        return None
    return {"i0": i0, "i1": i1, "j0": j0, "j1": j1, "X": X, "Y": Y,
            "valides": valides, "lisse": _mediane_3x3(h, valides)}


def pans_toit(polygone_m, cellules, sol=None, sol_bas=None):
    """Toit en pans (vue3d/pans.py) sur la fenêtre d'un toit, ou None.

    Même surface que surface_toit — le MNH lissé, redressé sur le terrain —
    mais réduite à quelques plans et fermée par ses murs.
    """
    i0, i1, j0, j1 = cellules["i0"], cellules["i1"], cellules["j0"], cellules["j1"]
    valides = cellules["valides"]
    z = cellules["lisse"]
    if sol is not None and sol_bas is not None:
        z = z + np.nan_to_num(np.asarray(sol[j0:j1, i0:i1], dtype=np.float64) - sol_bas)
    pans = pans_du_toit(polygone_m, cellules["X"], cellules["Y"],
                        _combler(np.where(valides, z, np.nan)), valides)
    if pans:
        pans.update(i0=i0, j0=j0)
    return pans


def surface_toit(polygone_m, hauteurs, xs, ys, gouttiere, verdure=None, sol=None, sol_bas=None,
                 cellules=None):
    """Grille des hauteurs du toit sous une emprise, prête pour la vue.

    Args:
        polygone_m, hauteurs, xs, ys, gouttiere, verdure: voir cellules_du_toit.
        sol, sol_bas: terrain aligné sur `hauteurs` (celui dont le MNH est
            tiré) et son altitude la plus basse sous le contour ; None sur
            terrain inconnu.
        cellules: résultat de cellules_du_toit, s'il est déjà calculé.

    Returns:
        dict(i0, j0, l, h, zero_m, pas_m, z) : fenêtre de la grille MNH (index
        de sa première colonne et de sa première ligne, dimensions), hauteurs
        en décimètres au-dessus de zero_m, encodées par `_encoder_ecarts` —
        rapportées au point le plus bas du terrain sous le contour, comme la
        base que la vue donne au bâtiment. None si trop peu de cellules sont
        lisibles.
    """
    c = cellules or cellules_du_toit(polygone_m, hauteurs, xs, ys, gouttiere, verdure)
    if c is None:
        return None
    i0, i1, j0, j1, X, Y = c["i0"], c["i1"], c["j0"], c["j1"], c["X"], c["Y"]
    z = _combler(c["lisse"])
    if sol is not None and sol_bas is not None:
        z = z + np.nan_to_num(np.asarray(sol[j0:j1, i0:i1], dtype=np.float64) - sol_bas)
    zero = float(np.floor(np.nanmin(z)))
    q = np.rint((z - zero) * 10)
    # Utile : le carré d'interpolation de la cellule touche l'emprise, soit un
    # centre à moins d'une demi-diagonale de maille du contour.
    demi_diagonale = 0.5 * math.hypot(xs[1] - xs[0], ys[0] - ys[1]) + 1e-6
    utile = proches(polygone_m, X, Y, demi_diagonale)
    return {"i0": i0, "j0": j0, "l": i1 - i0, "h": j1 - j0,
            "zero_m": zero, "pas_m": 0.1, "z": _encoder_ecarts(q, utile)}


# --- Toit résumé ou surface ? ----------------------------------------------
# La surface suit le LiDAR, mais elle en garde le grain : sur un toit simple,
# le toit résumé (deux pans, corps, pyramide) est aussi juste et plus net. On
# ne l'abandonne que là où il s'écarte du LiDAR.
#
# Écart médian entre le toit résumé et le LiDAR, par toit fiable :
#
#                   p25    p50    p75    p90
#   Gordes (89)    0,37   0,60   1,03   1,49 m
#   Strasbourg     0,65   0,98   1,40   2,08 m
#
# Aucune rupture dans la distribution : le seuil est un arbitrage, fixé sur des
# toits examinés un à un. Bien résumés : une maison simple (0,10 m), un toit à
# deux pans de Gordes (0,60), un de Strasbourg (0,64). Manqués : une maison
# dont une moitié est surélevée (0,81), deux niveaux sous un seul faîtage
# (1,03), un îlot dont les faîtages font le tour de la cour (0,98), une aile
# en L (1,40). Un appentis dessiné en deux pans ne s'écarte que de 0,37 m :
# faux, mais sur 0,8 m de dénivelé, il reste résumé.
#
# À 0,7 m, la surface va à 40 % des toits fiables de Gordes, 70 % de ceux de
# Strasbourg. L'écart relatif au dénivelé ne sépare pas mieux : il gonfle sur
# les toits peu pentus (0,37 pour le bon résumé de Gordes, 0,30 pour les deux
# niveaux).
SURFACE_ECART_RESUME_M = 0.7


def _tourner(x, y, ang):
    c, s = math.cos(-ang), math.sin(-ang)
    return x * c - y * s, x * s + y * c


def _rectangle_selon_axe(pts, ang):
    u, v = _tourner(pts[:, 0], pts[:, 1], ang)
    return {"ang": ang, "x0": u.min(), "x1": u.max(), "y0": v.min(), "y1": v.max()}


def _rectangle_min(polygone_m):
    """Rectangle d'aire minimale sur les côtés de l'enveloppe convexe."""
    enveloppe = np.asarray(polygone_m.convex_hull.exterior.coords)[:-1, :2]
    meilleur = None
    for a, b in zip(enveloppe, np.roll(enveloppe, -1, axis=0)):
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        r = _rectangle_selon_axe(enveloppe, ang)
        aire = (r["x1"] - r["x0"]) * (r["y1"] - r["y0"])
        if meilleur is None or aire < meilleur[0]:
            meilleur = (aire, r)
    return meilleur[1] if meilleur else None


def _deux_pans(rect, g, f, X, Y, suivant_x=None):
    """Toit à deux pans sur le rectangle (NaN au-dehors) ; faîtage au milieu."""
    u, v = _tourner(X, Y, rect["ang"])
    lx, ly = rect["x1"] - rect["x0"], rect["y1"] - rect["y0"]
    if suivant_x is None:
        suivant_x = lx >= ly
    d = (np.abs(v - (rect["y0"] + rect["y1"]) / 2) / max(ly / 2, 1e-9) if suivant_x
         else np.abs(u - (rect["x0"] + rect["x1"]) / 2) / max(lx / 2, 1e-9))
    e = 1e-6
    dedans = ((u >= rect["x0"] - e) & (u <= rect["x1"] + e)
              & (v >= rect["y0"] - e) & (v <= rect["y1"] + e))
    return np.where(dedans, f - (f - g) * np.clip(d, 0, 1), np.nan)


def _pyramide(rect, g, f, X, Y):
    u, v = _tourner(X, Y, rect["ang"])
    hx = max((rect["x1"] - rect["x0"]) / 2, 1e-9)
    hy = max((rect["y1"] - rect["y0"]) / 2, 1e-9)
    d = np.maximum(np.abs(u - (rect["x0"] + rect["x1"]) / 2) / hx,
                   np.abs(v - (rect["y0"] + rect["y1"]) / 2) / hy)
    return np.where(d <= 1 + 1e-6, f - (f - g) * np.clip(d, 0, 1), np.nan)


def hauteur_resumee(profil, polygone_m, X, Y):
    """Hauteur du toit résumé en (X, Y), tel que la vue le dessine.

    Même règle que construire() dans index.html pour un profil LiDAR fiable :
    un toit par corps (deux pans sur son axe, pyramide sans axe) au-dessus de
    murs à la plus basse de leurs gouttières ; sinon deux pans sur le
    rectangle orienté selon l'axe mesuré, ou d'aire minimale ; toit plat sous
    0,5 m de dénivelé. À tenir en accord avec la vue.
    """
    corps = profil.get("corps") or []
    if len(corps) >= 2:
        z = np.full(X.shape, max(min(c["gouttiere"] for c in corps), 0.5))
        for c in corps:
            ang = math.radians(c["axe_deg"] if c["axe_deg"] is not None else 0.0)
            cx, cy = _tourner(np.float64(c["cx"]), np.float64(c["cy"]), ang)
            rect = {"ang": ang, "x0": cx - c["longueur"] / 2, "x1": cx + c["longueur"] / 2,
                    "y0": cy - c["largeur"] / 2, "y1": cy + c["largeur"] / 2}
            g = max(c["gouttiere"], 0.5)
            f = max(c["faitage"], g + 0.3)
            toit = (_deux_pans(rect, g, f, X, Y, True) if c["axe_deg"] is not None
                    else _pyramide(rect, g, f, X, Y))
            z = np.fmax(z, toit)
        return z
    mur = max(profil["gouttiere"], 0.5)
    denivele = max(profil["faitage"] - profil["gouttiere"], 0.0)
    if denivele <= 0.5:
        return np.full(X.shape, mur)
    axe = profil.get("axe_deg")
    if axe is not None:
        pts = np.asarray(polygone_m.exterior.coords)[:-1, :2]
        rect = _rectangle_selon_axe(pts, math.radians(axe))
    else:
        rect = _rectangle_min(polygone_m)
    z = _deux_pans(rect, mur, mur + denivele, X, Y, True if axe is not None else None)
    return np.where(np.isnan(z), mur, z)


def ecart_au_resume(profil, polygone_m, cellules):
    """Écart médian, en mètres, entre le toit résumé et le LiDAR lissé.

    Sur la forme seule : le toit résumé ignore la pente du terrain, on le
    compare donc au MNH, hauteur au-dessus du sol local, et non à la surface
    redressée.
    """
    v = cellules["valides"]
    resume = hauteur_resumee(profil, polygone_m, cellules["X"], cellules["Y"])
    return float(np.median(np.abs(resume - cellules["lisse"])[v]))


def part_verte(polygone_m, verdure):
    """Part des cellules de l'emprise que l'orthophoto voit vertes.

    Args:
        verdure: (exg, xs, ys) — la grille ExG et les coordonnées locales de
            ses colonnes et de ses lignes, en mètres. Toutes les cellules,
            sol compris : un toit n'a pas de sursol mais a une couleur.

    Returns:
        float dans [0, 1], ou None si l'emprise ne contient aucune cellule.

    Le test point-dans-polygone est vectorisé sur la seule fenêtre de la
    boîte englobante : cellule par cellule, quinze bâtiments sur une grille
    de 370 000 cellules demandaient 26 s, bien au-delà des 8 s que le rendu
    accorde aux toitures.
    """
    exg, xs, ys = verdure
    minx, miny, maxx, maxy = polygone_m.bounds
    i0, i1 = np.searchsorted(xs, [minx, maxx])
    # ys descend du nord : -ys croît, et la boîte s'y lit du haut vers le bas.
    j0, j1 = np.searchsorted(-ys, [-maxy, -miny])
    i0, j0 = max(int(i0) - 1, 0), max(int(j0) - 1, 0)
    i1, j1 = min(int(i1) + 1, len(xs)), min(int(j1) + 1, len(ys))
    if i0 >= i1 or j0 >= j1:
        return None
    X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1])
    dedans = shapely.contains_xy(polygone_m, X, Y)
    n = int(dedans.sum())
    if not n:
        return None
    return int((exg[j0:j1, i0:i1][dedans] >= EXG_SEUIL).sum()) / n


def qualifier_couvert(profil, part):
    """Pose sous_couvert et hauteur_inconnue sur un profil d'après la part verte.

    - sous_couvert : l'emprise est verte au-delà de TOITS_COUVERT_PART_VERTE.
      Le LiDAR y mesure une canopée ; la mesure est rejetée même si sa forme
      ressemble à un toit plat.
    - hauteur_inconnue : ni le LiDAR ni la BD TOPO ne sont crédibles. C'est le
      cas sous couvert, et pour un profil rejeté sur une emprise en partie
      verte : la BD TOPO, tirée d'un MNS photogrammétrique, y est contaminée
      par les mêmes arbres. Le rendu doit le dire, pas inventer une hauteur.
    """
    profil["part_verte"] = None if part is None else round(part, 2)
    couvert = part is not None and part >= TOITS_COUVERT_PART_VERTE
    profil["sous_couvert"] = couvert
    if couvert:
        profil["fiable"] = False
    profil["hauteur_inconnue"] = couvert or (
        not profil["fiable"] and part is not None and part >= TOITS_ARBRES_PART_VERTE)
    return profil


def _cellules_locales(grille, lon0, lat0):
    """Grille MNH -> liste (x, y, h) en mètres autour de (lon0, lat0)."""
    west, south, east, north = grille["bbox"]
    nx, ny = grille["width"], grille["height"]
    m_lon = 111320 * math.cos(math.radians(lat0))
    m_lat = 111320
    valeurs = grille["values"]
    out = []
    for j in range(ny):
        y = (north - (north - south) * (j + 0.5) / ny - lat0) * m_lat
        base = j * nx
        for i in range(nx):
            h = valeurs[base + i]
            if h <= 0:
                continue   # sol : inutile pour un toit
            x = (west + (east - west) * (i + 0.5) / nx - lon0) * m_lon
            out.append((x, y, h))
    return out


def _cellules_de_la_boite(valeurs, xs, ys, bornes):
    """Les (x, y, h) de `_cellules_locales` dont le centre tombe dans la
    boîte `bornes` (minx, miny, maxx, maxy), dans le même ordre, en tableau
    n × 3.

    `profil_toit` ne garde d'une liste que ce qui tombe dans l'emprise érodée
    du bâtiment : lui donner la grille entière, c'était la relire en Python
    pour chaque bâtiment. Mesuré sur Gordes en zone de 1 000 m (676
    bâtiments, grille de 1 440 × 1 985) : 41 s de toitures, 6 s ainsi, pour
    la même scène à l'octet.

    Args:
        valeurs: hauteurs de la grille en float64 (ny × nx), sans arrondi :
            les float de la liste d'origine.
        xs, ys: coordonnées locales des colonnes (croissantes) et des lignes
            (décroissantes), de `_verdure_locale`.
    """
    minx, miny, maxx, maxy = bornes
    i0, i1 = np.searchsorted(xs, minx, "left"), np.searchsorted(xs, maxx, "right")
    j0, j1 = np.searchsorted(-ys, -maxy, "left"), np.searchsorted(-ys, -miny, "right")
    fenetre = valeurs[j0:j1, i0:i1]
    js, is_ = np.nonzero(fenetre > 0)       # sol : inutile pour un toit
    return np.column_stack((xs[i0:i1][is_], ys[j0:j1][js], fenetre[js, is_]))


def _verdure_locale(exg, bbox, lon0, lat0):
    """Grille ExG (numpy) + coordonnées locales de ses colonnes et lignes.

    Le repère est celui des cellules du MNH : mètres autour de (lon0, lat0),
    y vers le nord. `xs` croît, `ys` décroît (la grille descend du nord-ouest).
    """
    west, south, east, north = bbox
    ny, nx = exg.shape
    m_lon = 111320 * math.cos(math.radians(lat0))
    xs = (west + (east - west) * (np.arange(nx) + 0.5) / nx - lon0) * m_lon
    ys = (north - (north - south) * (np.arange(ny) + 0.5) / ny - lat0) * 111320
    return exg, xs, ys


def _morceaux(geom):
    return [geom] if geom.geom_type == "Polygon" else list(geom.geoms)


def _sol_bas(sol, geom, xs, ys, lon0, lat0, m_lon):
    """Altitude la plus basse du terrain aux sommets du contour extérieur.

    Comme la base que la vue donne au bâtiment (baseBatiment) : toit et murs
    partent ainsi du même zéro. Cellule la plus proche : sous un bâtiment, le
    terrain LiDAR est interpolé depuis le sol alentour, donc lisse.
    """
    alts = []
    for a in _morceaux(geom):
        for lon, lat, *_ in a.exterior.coords:
            i = int(np.clip(np.searchsorted(xs, (lon - lon0) * m_lon), 0, len(xs) - 1))
            j = int(np.clip(np.searchsorted(-ys, -(lat - lat0) * 111320), 0, len(ys) - 1))
            alts.append(sol[j, i])
    bas = np.nanmin(alts) if alts and not np.all(np.isnan(alts)) else np.nan
    return float(bas) if np.isfinite(bas) else None


def _toit_du_batiment(f, valeurs, hauteurs, exg, sol, verdure, xs, ys, lon0, lat0, m_lon):
    """Profil de toit d'un bâtiment de `toits_pour_emprise`, ou None pour un
    bâtiment sans identifiant ni géométrie lisible, qui n'en reçoit pas.

    Les grilles sont celles de la scène entière, lues avec ses index — ou,
    dans un processus du bassin, leurs fenêtres autour du bâtiment
    (`_Fenetre`), lues avec les mêmes index : le calcul est le même.
    """
    cleabs = (f.get("properties") or {}).get("cleabs")
    if not cleabs or not f.get("geometry"):
        return None
    try:
        geom = shape(f["geometry"])
    except Exception:
        return None
    # Projection locale identique à celle des cellules.
    poly = transform(lambda x, y, z=None: ((x - lon0) * m_lon, (y - lat0) * 111320), geom)
    # Un bâtiment coupé par le bord de la scène (vue3d/batiments.py) juge
    # la pente de son toit à la largeur du bâtiment entier : le morceau
    # gardé d'un toit à deux pans est plus étroit que lui sans être plus
    # raide. Mesuré à Strasbourg : à la largeur du morceau, 3 toits
    # fiables entiers devenaient « non fiables » une fois coupés.
    coupe = (f.get("properties") or {}).get("coupe") or {}
    cellules = _cellules_de_la_boite(valeurs, xs, ys, poly.bounds)
    profil = profil_toit(cellules, poly, largeur_min=coupe.get("largeur_m"))
    part = part_verte(poly, verdure) if verdure is not None else None
    if profil is None:
        # Emprise illisible : moins de TOITS_MIN_CELLULES cellules une fois
        # érodée. Sans profil publié, la vue retombait sur les altitudes
        # BD TOPO même sous le couvert, où elles sortent d'un MNS
        # photogrammétrique contaminé par la même canopée : 8,1 m de murs
        # annoncés pour une cabane d'environ 3 m sous les arbres, et le
        # verdict basculait au gré du calage de la grille (20, 21 ou 19
        # cellules pour la même cabane selon le point arrondi de la scène).
        # Un profil minimal laisse qualifier_couvert trancher : hauteur
        # inconnue si la part verte le justifie, sinon (fiable à faux) la
        # vue garde son repli BD TOPO — hors canopée, il reste bon.
        return qualifier_couvert({"fiable": False, "n": 0}, part)
    profil = qualifier_couvert(profil, part)
    if (profil["mode_bas"] and part is not None and part < TOITS_ARBRES_PART_VERTE):
        # Deux niveaux, pas un arbre : le « mode haut » d'une emprise que
        # l'orthophoto ne voit pas verte est le bâtiment lui-même — tours
        # et lanternes d'un château, corps principal au-dessus d'une cour.
        # Le château de Chambord (33 % de l'emprise au-dessus des
        # terrasses, 7 % de vert) était ainsi dessiné comme ses seules
        # terrasses, sous cinq corps de toit ; un immeuble d'Annecy de
        # 22,5 m, à 3,7 m, la hauteur de sa cour. Seule la forme mesurée
        # peut dire deux niveaux ; le résumé du niveau bas reste le repli.
        #
        # Mesuré sur 1 516 bâtiments de dix lieux : 123 profils relus en
        # mode bas, dont 50 à moins de TOITS_ARBRES_PART_VERTE de vert. De
        # ceux-là, 26 y gagnent leur forme — 16 en pans, 10 en surface, à
        # 3 à 12 m du résumé du niveau bas pour les plus grands —, 16
        # gardent ce résumé (niveau haut trop petit pour s'en écarter de
        # 0,7 m, emprise en morceaux) et 8 restent rejetés. Les 73 autres,
        # sous les arbres, ne changent pas.
        profil["mode_bas"] = False
        profil["deux_niveaux"] = True
    # Seulement là où le profil est un toit : sous un arbre (mode bas),
    # la grille décrirait le feuillage autant que la toiture.
    # D'un seul tenant aussi : la vue pose chaque morceau d'une emprise
    # sur sa propre base, une grille commune n'aurait pas de zéro.
    if (profil["fiable"] and not profil["mode_bas"] and not profil["hauteur_inconnue"]
            and len(_morceaux(geom)) == 1):
        poly_seul = _morceaux(poly)[0]
        fenetre = cellules_du_toit(poly_seul, hauteurs, xs, ys, profil["gouttiere"], exg)
        if fenetre is not None:
            ecart = ecart_au_resume(profil, poly_seul, fenetre)
            profil["ecart_resume"] = round(ecart, 2)
            if ecart > SURFACE_ECART_RESUME_M:
                bas = _sol_bas(sol, geom, xs, ys, lon0, lat0, m_lon) if sol is not None else None
                # Les pans d'abord : la forme de la surface, sans son
                # grain. La surface reste le repli d'un toit qui n'est pas
                # fait de plans.
                pans = pans_toit(poly_seul, fenetre, sol, bas)
                if pans:
                    profil["pans"] = pans
                else:
                    surface = surface_toit(poly_seul, hauteurs, xs, ys, profil["gouttiere"],
                                           exg, sol, bas, fenetre)
                    if surface:
                        profil["surface"] = surface
    return profil


def toits_pour_emprise(west, south, east, north, batiments_geojson, grille, exg,
                       sol=None, processus=None):
    """Profils de toit LiDAR pour les bâtiments d'une emprise.

    Args:
        grille, exg: grille MNH à 0,5 m et grille de verdure ExG, téléchargées
            une fois et partagées avec la segmentation des houppiers.
        sol: terrain sous la grille MNH (`mnh.fetch_sol_grid`), ou None. Il
            redresse la surface des toits sur un terrain en pente : le MNH est
            une hauteur au-dessus du sol local, et un faîtage horizontal y
            paraît incliné comme le sol. Sous une emprise, ce terrain varie de
            3,1 m en médiane à Gordes (p90 6,7 m, jusqu'à 11 m), de 0,6 m à
            Strasbourg. Pas le RGE ALTI de la scène : sur un coteau en
            restanques, il variait de 2,1 m sous une maison dont le terrain
            LiDAR ne varie que de 0,6 m, et ondulait le toit d'autant.
        processus: nombre de processus du calcul (`_toits_en_parallele`) ;
            None pour celui du service, 1 pour tout calculer ici.

    Returns:
        dict: source ('lidar_hd' | 'mns_mnt' | None), resolution_m, ortho, et
        `toits` : {cleabs: profil}. Vide si l'emprise n'est couverte par
        aucune source d'altitude. Un bâtiment dont l'emprise ne livre pas
        assez de cellules reçoit un profil minimal (fiable à faux, part verte
        qualifiée), jamais rien : la vue doit savoir s'il est sous les arbres.
    """
    west, south, east, north = map(float, (west, south, east, north))
    resultat = {"source": grille.get("source"), "resolution_m": TOITS_RESOLUTION_M,
                "toits": {}, "ortho": exg is not None,
                # Les fenêtres des surfaces de toit sont des index de cette grille.
                "grille": {"bbox": grille.get("bbox"), "width": grille.get("width"),
                           "height": grille.get("height")}}
    if not grille.get("couvert"):
        return resultat
    lon0, lat0 = (west + east) / 2, (south + north) / 2
    m_lon = 111320 * math.cos(math.radians(lat0))
    # Second avis de l'orthophoto : sans elle, la canopée passe pour un toit.
    verdure = _verdure_locale(exg, grille["bbox"], lon0, lat0) if exg is not None else None
    # Les profils lisent les hauteurs d'origine, les surfaces leur arrondi en
    # float32 — tiré du tableau plutôt que de la liste, le même arrondi de
    # chaque float, dix fois moins cher.
    valeurs = np.asarray(grille["values"], dtype=np.float64).reshape(
        grille["height"], grille["width"])
    hauteurs = valeurs.astype(np.float32)
    xs, ys = _verdure_locale(hauteurs, grille["bbox"], lon0, lat0)[1:]
    if sol is not None:
        sol = np.asarray(sol, dtype=np.float64).reshape(hauteurs.shape)
    features = batiments_geojson.get("features", [])
    profils = _toits_en_parallele(features, valeurs, hauteurs, exg, sol, verdure, xs, ys,
                                  lon0, lat0, m_lon, processus)
    if profils is None:
        profils = [_toit_du_batiment(f, valeurs, hauteurs, exg, sol, verdure, xs, ys,
                                     lon0, lat0, m_lon) for f in features]
    # Dans l'ordre des bâtiments, quel que soit celui du calcul : c'est
    # l'ordre du dictionnaire, donc celui de la scène écrite.
    for f, profil in zip(features, profils):
        if profil is not None:
            resultat["toits"][f["properties"]["cleabs"]] = profil
    mesures = sum(1 for p in resultat["toits"].values() if "gouttiere" in p)
    journal.info("Toits LiDAR : %d bâtiment(s) mesuré(s), %d profil(s) minimal(aux), source=%s",
                 mesures, len(resultat["toits"]) - mesures, resultat["source"])
    return resultat


# --- Calcul en parallèle ------------------------------------------------------
# Chaque bâtiment se calcule seul, sur sa fenêtre des grilles : les toitures
# se répartissent sur un bassin de processus. Des processus et non des fils :
# la segmentation des pans et le profil sont des boucles Python, que le GIL
# sérialise.
#
# Le bassin est créé une fois, au premier calcul, et sert ensuite à toutes
# les scènes — et aux huit fils de gunicorn à la fois : un envoi au bassin est
# sûr entre fils. Ses processus naissent par « spawn » (macOS) ou d'un
# serveur « forkserver » (Linux, le conteneur), jamais d'un fork du service,
# qui a des fils et dont un verrou pris par l'un d'eux resterait pris dans
# l'enfant. S'il ne peut pas démarrer, ou s'il casse en route (un processus
# tué), les toitures se calculent dans le processus du service, comme avant :
# même résultat, plus lentement.
#
# Rien de ce que rend un processus ne dépend de lui : chaque bâtiment lit les
# mêmes nombres avec les mêmes index (`_Fenetre`) et fait les mêmes
# opérations. Les bâtiments partent du centre de la scène vers le bord, mais
# les profils sont rangés dans l'ordre des bâtiments : le dictionnaire des
# toits est le même, à l'octet. Cet ordre d'envoi est sans effet visible : la
# scène n'est écrite, et la page ne l'affiche, qu'une fois tous les toits
# rendus. Livrer les toits du centre d'abord demanderait une couche à part,
# écartée : les toits sont tirés du MNH, et ce qui touche au MNH va dans la
# scène (CLAUDE.md).
#
# Des fils plutôt que des processus ? Mesuré à Strasbourg en zone de
# 1 000 m : 6,1 s dans un seul fil, 7,0 s sur dix fils (le GIL), 1,3 s sur
# dix processus.

# Sous ce nombre de bâtiments, le bassin coûte plus qu'il ne rapporte.
# Mesuré sur les premiers bâtiments de Gordes, bassin de 10 déjà démarré :
# 32 bâtiments, 0,056 s ici contre 0,064 s au bassin ; 64, 0,138 contre
# 0,112 ; 128, 0,33 contre 0,18.
TOITS_PARALLELE_MIN = 48
# Bâtiments par envoi au bassin, au plus : assez pour amortir l'envoi, assez
# peu pour que les processus finissent ensemble. Mesuré deux fois sur
# Strasbourg en zone de 1 000 m (bassin de 10, machine chargée par d'autres
# calculs) : 1 → 3,7 et 1,7 s ; 4 → 2,5 et 1,6 ; 8 → 1,50 et 1,43 ;
# 16 → 1,59 et 1,59 ; 32 → 1,48 et 1,92 ; 64 → 1,69 et 1,89.
TOITS_LOT_MAX = 8
# Cellules de marge autour de la boîte d'un bâtiment, dans sa fenêtre : les
# lectures de ce module débordent d'une cellule au plus (une colonne de part
# et d'autre de l'emprise, pour encadrer le contour).
TOITS_FENETRE_MARGE = 4
# Cassé plus de fois que cela, le bassin n'est plus recréé.
TOITS_BASSIN_ESSAIS = 3
# Intervalle, en secondes, auquel un processus du bassin vérifie que le
# service vit encore (`_veiller_parent`) : un appel système par seconde, et
# un service tué ne laisse rien derrière lui plus d'une seconde.
TOITS_VEILLE_S = 1.0
# Processus du bassin au plus, sans VUE3D_TOITS_PROCESSUS. Le gain croît
# encore à dix sur les dix cœurs du M4 — toitures de Strasbourg en zone de
# 1 000 m, bassin chaud, médianes de trois tours, machine chargée par
# d'autres calculs (charge 70 à 100) : 1 processus 8,3 s ; 2, 6,7 s ; 4,
# 4,2 s ; 6, 3,7 s ; 8, 2,2 s ; 10, 1,5 s. Mais chaque processus garde sa
# mémoire, après cette scène : 95 à 275 Mo (RSS) sous macOS (spawn), 27 à
# 41 Mo (PSS) dans le conteneur (forkserver) ; et le service garde sa part,
# environ 0,3 s à Strasbourg (grilles en tableaux, envoi des fenêtres) :
# de 16 à 64 processus, la loi d'Amdahl n'en promet qu'environ 0,5 s de
# moins, pour 48 processus permanents de plus. Au-delà de 16 :
# VUE3D_TOITS_PROCESSUS.
TOITS_PROCESSUS_MAX = 16

_bassin = None
# Réentrant : un bassin qui ne se crée pas est abandonné sous le verrou de
# sa création.
_bassin_verrou = threading.RLock()
_bassin_casse = 0
# Le bassin ne sert que si le service l'a autorisé (vue3d/app.py), ou si
# l'appelant le demande (`processus`). Ses processus réexécutent le script
# principal de qui les lance : gunicorn, flask et pytest le gardent derrière
# `if __name__ == "__main__"`, un script de mesure ou de rejeu ne le fait
# pas toujours, et chaque processus y referait tout le rejeu.
_bassin_autorise = False


def autoriser_bassin(oui=True):
    """Autorise le bassin pour les appels sans `processus` : le service le
    fait au démarrage, et le bassin démarre aussitôt, dans un fil à part
    (`_chauffer_bassin`), plutôt qu'à la première scène."""
    global _bassin_autorise
    _bassin_autorise = oui
    if oui and processus_toits() >= 2:
        threading.Thread(target=_chauffer_bassin, name="bassin-des-toitures",
                         daemon=True).start()


def _chauffer_bassin():
    """Fait naître le bassin et ses processus, sans attendre de scène.

    Un processus du bassin met du temps à naître — un Python neuf qui
    importe numpy, shapely et ce module (spawn), ou un fork du serveur
    forkserver — et la première scène après le démarrage du service le
    payait. Toitures de la première scène, Gordes, emprise par défaut
    (médianes de cinq, en alternant, M4 chargé par d'autres calculs), le
    bassin né à la première scène, puis au démarrage :

        rejouée 1 s après le démarrage, macOS (spawn)       0,77 -> 0,28 s
        rejouée 1 s après le démarrage, conteneur           0,97 -> 0,50 s
        rejouée dès le démarrage, macOS                     0,83 -> 0,72 s
        rejouée dès le démarrage, conteneur                 0,90 -> 0,64 s
        servie par flask, demandée dès qu'il répond, macOS  0,53 -> 0,20 s

    Servies par flask, les toitures commencent 0,9 à 1 s après le
    démarrage, à la fin des lectures ; la scène arrive en 1,07 s au lieu de
    1,57 (houppiers pendant les toitures compris, scene.assembler). Sur un
    cœur, ces toitures prennent 0,55 à 0,67 s : le bassin froid faisait
    pire que pas de bassin.

    Un envoi au bassin fait naître un processus tant qu'aucun n'est libre :
    une tâche vide par processus les lance tous.
    """
    bassin = _bassin_de_calcul()
    if bassin is None:
        return
    try:
        for tache in [bassin.submit(os.getpid) for _ in range(max(2, processus_toits()))]:
            tache.result()
    except concurrent.futures.BrokenExecutor as exc:
        _abandonner_bassin(f"bassin cassé au démarrage ({exc})", bassin)
    except Exception as exc:
        # Rien de perdu : le bassin naîtra à la première scène, ou celle-ci
        # se calculera dans le service.
        journal.warning("Toitures : le bassin n'a pas démarré d'avance (%r)", exc)


def processus_toits():
    """Processus du calcul des toitures : VUE3D_TOITS_PROCESSUS s'il est
    donné (1 : pas de bassin), sinon les cœurs dont dispose le service,
    TOITS_PROCESSUS_MAX au plus."""
    valeur = os.environ.get("VUE3D_TOITS_PROCESSUS", "").strip()
    if valeur:
        try:
            return max(1, int(valeur))
        except ValueError:
            journal.warning("VUE3D_TOITS_PROCESSUS=%r n'est pas un nombre : ignoré", valeur)
    try:
        coeurs = len(os.sched_getaffinity(0))      # Linux : les cœurs du conteneur
    except AttributeError:
        coeurs = os.cpu_count() or 1
    return min(coeurs, TOITS_PROCESSUS_MAX)


def _bassin_de_calcul():
    """Le bassin de processus du service, créé au premier appel ; None s'il
    est hors service."""
    global _bassin
    if multiprocessing.parent_process() is not None:
        return None             # dans un processus du bassin : pas de bassin
    with _bassin_verrou:
        if _bassin is None and _bassin_casse < TOITS_BASSIN_ESSAIS:
            n = max(2, processus_toits())
            methode = "forkserver" if sys.platform.startswith("linux") else "spawn"
            try:
                contexte = multiprocessing.get_context(methode)
                if methode == "forkserver":
                    # Le serveur importe numpy et shapely une fois ; chaque
                    # processus en naît prêt.
                    contexte.set_forkserver_preload([__name__])
                _bassin = concurrent.futures.ProcessPoolExecutor(
                    max_workers=n, mp_context=contexte, initializer=_veiller_parent,
                    initargs=(os.getpid(),))
            except Exception as exc:
                _abandonner_bassin(f"bassin de {n} processus impossible à créer : {exc}")
        return _bassin


def _veiller_parent(service):
    """Initialiseur des processus du bassin : ils se terminent avec le service.

    Un service arrêté net (SIGTERM sur `flask run`, SIGKILL de gunicorn sur
    un délai dépassé) n'arrête pas son bassin : ses dix processus, attendant
    leur prochain bâtiment sur un tube dont ils gardent eux-mêmes l'autre
    bout, restaient en vie, rattachés à init (constaté sur macOS). Un
    processus du bassin s'arrête donc quand son parent change — le service
    (spawn) ou le serveur forkserver, qui s'arrête avec lui — ou quand le
    service n'existe plus : un processus encore en train de démarrer quand
    le service meurt a déjà init pour parent en arrivant ici.

    Args:
        service: numéro du processus du service.
    """
    parent = os.getppid()

    def vivant():
        if os.getppid() != parent:
            return False
        try:
            os.kill(service, 0)
        except (ProcessLookupError, PermissionError):   # fini, ou numéro repris
            return False
        return True

    def veiller():
        while vivant():
            time.sleep(TOITS_VEILLE_S)
        os._exit(0)

    threading.Thread(target=veiller, name="veille-du-service", daemon=True).start()


def _abandonner_bassin(raison, bassin=None):
    """Arrête `bassin` s'il est encore celui du service (None : celui qui
    n'a pas pu naître) ; le calcul suivant en recrée un, jusqu'à
    TOITS_BASSIN_ESSAIS fois."""
    global _bassin, _bassin_casse
    journal.warning("Toitures calculées dans le processus du service : %s", raison)
    with _bassin_verrou:
        if bassin is _bassin:
            _bassin_casse += 1
            if _bassin is not None:
                _bassin.shutdown(wait=False, cancel_futures=True)
            _bassin = None


class _Fenetre:
    """Fenêtre d'une grille, lue avec les index de la grille entière.

    Un processus du bassin ne reçoit des grilles que la fenêtre autour de son
    bâtiment ; les fonctions de ce module la lisent avec les index de la
    grille entière, calculés sur xs et ys entiers, et obtiennent les mêmes
    nombres. Une lecture qui sortirait de la fenêtre lève IndexError plutôt
    que de rendre autre chose : le calcul reprend alors dans le processus du
    service.
    """

    def __init__(self, a, j0, i0, forme):
        self.a, self.j0, self.i0, self.shape = a, j0, i0, forme

    def _bornes(self, t, axe):
        """(début, fin) d'une tranche dans la grille entière, bornées comme
        numpy borne une tranche ; fin <= début si elle est vide."""
        debut, fin, pas = t.indices(self.shape[axe])
        if pas != 1:
            raise IndexError("tranche à pas non unitaire")
        return debut, fin

    def __getitem__(self, cle):
        j, i = cle
        if isinstance(j, slice) and isinstance(i, slice):
            bornes = (self._bornes(j, 0), self._bornes(i, 1))
            if any(fin <= debut for debut, fin in bornes):
                # Vide dans la grille entière : vide ici aussi, de même forme.
                return np.empty([max(fin - debut, 0) for debut, fin in bornes],
                                dtype=self.a.dtype)
            (j0, j1), (i0, i1) = bornes
            if (j0 < self.j0 or i0 < self.i0 or j1 > self.j0 + self.a.shape[0]
                    or i1 > self.i0 + self.a.shape[1]):
                raise IndexError(f"lecture hors de la fenêtre : {j0}:{j1}, {i0}:{i1}")
            return self.a[j0 - self.j0:j1 - self.j0, i0 - self.i0:i1 - self.i0]
        if isinstance(j, slice) or isinstance(i, slice):
            raise IndexError("lecture mixte d'une fenêtre non prévue")
        jj, ii = int(j) - self.j0, int(i) - self.i0
        if not (0 <= jj < self.a.shape[0] and 0 <= ii < self.a.shape[1]):
            raise IndexError(f"lecture hors de la fenêtre : {j}, {i}")
        return self.a[jj, ii]


def _bornes_geojson(geometrie):
    """(ouest, sud, est, nord) des sommets d'une géométrie GeoJSON ; None si
    elle n'en a pas que les fenêtres sachent borner — pas un dict, aucune
    coordonnée, un sommet sans deux nombres finis. Ce bâtiment-là est
    calculé dans le service, comme sur un cœur : shape() y échoue ou non,
    de même."""
    if not isinstance(geometrie, dict):
        return None
    lons, lats = [], []
    pile = [geometrie.get("coordinates")]
    while pile:
        c = pile.pop()
        if isinstance(c, (list, tuple)) and c:
            if isinstance(c[0], (int, float)):
                if (len(c) < 2 or not isinstance(c[1], (int, float))
                        or not (math.isfinite(c[0]) and math.isfinite(c[1]))):
                    return None
                lons.append(c[0])
                lats.append(c[1])
            else:
                pile.extend(c)
    return (min(lons), min(lats), max(lons), max(lats)) if lons else None


def _toits_du_lot(lot, xs, ys, verdure_xy, lon0, lat0, m_lon, forme):
    """Profils d'un lot de bâtiments, dans un processus du bassin.

    `lot` : (bâtiment, j0, i0, valeurs, exg, sol) — les trois grilles sur la
    fenêtre du bâtiment, qui commence à la ligne j0 et à la colonne i0.
    """
    out = []
    for f, j0, i0, v, e, s in lot:
        valeurs = _Fenetre(v, j0, i0, forme)
        # Comme toits_pour_emprise : le float32 des mêmes float.
        hauteurs = _Fenetre(v.astype(np.float32), j0, i0, forme)
        exg = None if e is None else _Fenetre(e, j0, i0, forme)
        sol = None if s is None else _Fenetre(s, j0, i0, forme)
        verdure = None if exg is None else (exg, *verdure_xy)
        out.append(_toit_du_batiment(f, valeurs, hauteurs, exg, sol, verdure, xs, ys,
                                     lon0, lat0, m_lon))
    return out


def _toits_en_parallele(features, valeurs, hauteurs, exg, sol, verdure, xs, ys,
                        lon0, lat0, m_lon, processus=None):
    """Profils des bâtiments, dans leur ordre, calculés par le bassin ; None
    s'il n'y a pas lieu (peu de bâtiments, processus=1, bassin hors service)
    ou si le bassin a échoué : l'appelant calcule alors tout lui-même.

    Les bâtiments partent du centre de la scène vers le bord, sans effet sur
    le résultat ni sur l'affichage : la scène attend tous les profils.

    Args:
        processus: None, le bassin du service s'il est autorisé
            (`autoriser_bassin`) ; 1, aucun ; 2 ou plus, le bassin, dont la
            taille reste celle de `processus_toits`.
    """
    if processus is None:
        processus = processus_toits() if _bassin_autorise else 1
    if processus < 2 or len(features) < TOITS_PARALLELE_MIN:
        return None
    if exg is not None and exg.shape != valeurs.shape:
        return None
    bassin = _bassin_de_calcul()
    if bassin is None:
        return None
    ny, nx = valeurs.shape
    profils = [None] * len(features)
    envois = []                 # (distance au centre, index, tâche)
    for k, f in enumerate(features):
        if not (f.get("properties") or {}).get("cleabs") or not f.get("geometry"):
            continue            # pas de profil, comme dans _toit_du_batiment
        bornes = _bornes_geojson(f["geometry"])
        if bornes is None:
            # Géométrie que les fenêtres ne savent pas borner : calculée ici.
            profils[k] = _toit_du_batiment(f, valeurs, hauteurs, exg, sol,
                                           verdure, xs, ys, lon0, lat0, m_lon)
            continue
        o, s_, e, n = bornes
        x0, x1 = (o - lon0) * m_lon, (e - lon0) * m_lon
        y0, y1 = (s_ - lat0) * 111320, (n - lat0) * 111320
        i0 = max(int(np.searchsorted(xs, x0)) - TOITS_FENETRE_MARGE, 0)
        i1 = min(int(np.searchsorted(xs, x1, "right")) + TOITS_FENETRE_MARGE, nx)
        j0 = max(int(np.searchsorted(-ys, -y1)) - TOITS_FENETRE_MARGE, 0)
        j1 = min(int(np.searchsorted(-ys, -y0, "right")) + TOITS_FENETRE_MARGE, ny)
        fenetre = (slice(j0, j1), slice(i0, i1))
        tache = (f, j0, i0, valeurs[fenetre],
                 None if exg is None else exg[fenetre],
                 None if sol is None else sol[fenetre])
        envois.append((((x0 + x1) / 2) ** 2 + ((y0 + y1) / 2) ** 2, k, tache))
    envois.sort(key=lambda t: (t[0], t[1]))
    taille = max(1, min(TOITS_LOT_MAX, -(-len(envois) // (4 * processus))))
    lots = [envois[a:a + taille] for a in range(0, len(envois), taille)]
    verdure_xy = None if verdure is None else verdure[1:]
    futures = []
    try:
        for lot in lots:
            futures.append(bassin.submit(_toits_du_lot, [t for _, _, t in lot], xs, ys,
                                         verdure_xy, lon0, lat0, m_lon, (ny, nx)))
        for lot, fut in zip(lots, futures):
            for (_, k, _), profil in zip(lot, fut.result()):
                profils[k] = profil
    except Exception as exc:
        # Un processus tué, un bassin qui ne démarre pas — ou une erreur du
        # calcul lui-même, que le calcul ici reproduira et laissera passer :
        # rien n'est avalé, la scène reste complète ou n'est pas écrite.
        for fut in futures:
            fut.cancel()
        if isinstance(exc, concurrent.futures.BrokenExecutor):
            _abandonner_bassin(f"bassin cassé ({exc})", bassin)
        else:
            journal.warning("Toitures : le bassin a échoué (%r), calcul repris ici", exc)
        return None
    return profils
