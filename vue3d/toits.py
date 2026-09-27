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

import logging
import math

import numpy as np
import shapely
from shapely.geometry import shape, Point

from .ortho import EXG_SEUIL
from .pans import pans_du_toit

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
        cellules: liste de (x, y, h) en mètres locaux, h = hauteur de sursol.
        polygone_m: emprise shapely en mètres locaux, même repère.

    Returns:
        dict(gouttiere, faitage, denivele, axe_deg, nettete, n) ou None si
        l'emprise érodée ne contient pas assez de cellules. `axe_deg` est
        l'angle du faîtage depuis l'est, dans [0, 180[, ou None si la mesure
        n'est pas nette.
    """
    def eroder(marge):
        inner = polygone_m.buffer(-marge)
        # Petite annexe : on érode moins plutôt que de renoncer.
        return inner if not inner.is_empty else polygone_m.buffer(-marge / 3)

    def contenu(inner):
        if inner.is_empty:
            return []
        minx, miny, maxx, maxy = inner.bounds
        return [(x, y, h) for x, y, h in cellules
                if minx <= x <= maxx and miny <= y <= maxy and inner.contains(Point(x, y))]

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
    # Sous-échantillon des faîtages pour la distance : 60 points suffisent.
    echant = [f[::max(1, len(f) // 60)] for f in faitages]

    def plus_proche(x, y):
        return min(range(len(echant)),
                   key=lambda k: min((x - a) ** 2 + (y - b) ** 2 for a, b in echant[k]))

    groupes = [[] for _ in comps]
    for x, y, h in dedans:
        groupes[plus_proche(x, y)].append((x, y, h))

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
    """
    g = np.where(valides, grille, np.nan)
    p = np.pad(g, 1, constant_values=np.nan)
    ny, nx = g.shape
    pile = np.stack([p[dj:dj + ny, di:di + nx] for dj in range(3) for di in range(3)])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # tranche toute en NaN
        m = np.nanmedian(pile, axis=0)
    return np.where(valides, m, np.nan)


def _combler(grille):
    """Remplit les NaN par la moyenne des voisins connus, de proche en proche."""
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
    # Une emprise qui déborde de la scène n'aurait qu'une partie de son toit :
    # la vue garde alors le toit résumé, qui la couvre en entier.
    if xs[i0] > minx or xs[i1 - 1] < maxx or ys[j0] < maxy or ys[j1 - 1] > miny:
        return None
    X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1])
    h = np.asarray(hauteurs[j0:j1, i0:i1], dtype=np.float64)
    dedans = shapely.contains_xy(polygone_m, X, Y)
    loin_du_bord = dedans & (shapely.distance(
        shapely.points(X.ravel(), Y.ravel()), polygone_m.boundary).reshape(X.shape)
        >= SURFACE_RETRAIT_M)
    valides = loin_du_bord & (h >= SURFACE_SOL_PART * gouttiere)
    if verdure is not None:
        valides &= verdure[j0:j1, i0:i1] < EXG_SEUIL
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
    utile = shapely.distance(shapely.points(X.ravel(), Y.ravel()),
                             polygone_m).reshape(X.shape) <= demi_diagonale
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


def toits_pour_emprise(west, south, east, north, batiments_geojson, grille, exg,
                       sol=None):
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

    Returns:
        dict: source ('lidar_hd' | 'mns_mnt' | None), resolution_m, ortho, et
        `toits` : {cleabs: profil}. Vide si l'emprise n'est couverte par
        aucune source d'altitude. Un bâtiment dont l'emprise ne livre pas
        assez de cellules reçoit un profil minimal (fiable à faux, part verte
        qualifiée), jamais rien : la vue doit savoir s'il est sous les arbres.
    """
    from shapely.ops import transform

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
    cellules = _cellules_locales(grille, lon0, lat0)
    # Second avis de l'orthophoto : sans elle, la canopée passe pour un toit.
    verdure = _verdure_locale(exg, grille["bbox"], lon0, lat0) if exg is not None else None
    hauteurs = np.asarray(grille["values"], dtype=np.float32).reshape(
        grille["height"], grille["width"])
    xs, ys = _verdure_locale(hauteurs, grille["bbox"], lon0, lat0)[1:]
    if sol is not None:
        sol = np.asarray(sol, dtype=np.float64).reshape(hauteurs.shape)
    for f in batiments_geojson.get("features", []):
        cleabs = (f.get("properties") or {}).get("cleabs")
        if not cleabs or not f.get("geometry"):
            continue
        try:
            geom = shape(f["geometry"])
        except Exception:
            continue
        # Projection locale identique à celle des cellules.
        poly = transform(lambda x, y, z=None: ((x - lon0) * m_lon, (y - lat0) * 111320), geom)
        profil = profil_toit(cellules, poly)
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
            resultat["toits"][cleabs] = qualifier_couvert({"fiable": False, "n": 0}, part)
            continue
        profil = qualifier_couvert(profil, part)
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
        resultat["toits"][cleabs] = profil
    mesures = sum(1 for p in resultat["toits"].values() if "gouttiere" in p)
    journal.info("Toits LiDAR : %d bâtiment(s) mesuré(s), %d profil(s) minimal(aux), source=%s",
                 mesures, len(resultat["toits"]) - mesures, resultat["source"])
    return resultat
