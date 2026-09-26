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


def toits_pour_emprise(west, south, east, north, batiments_geojson, grille, exg):
    """Profils de toit LiDAR pour les bâtiments d'une emprise.

    Args:
        grille, exg: grille MNH à 0,5 m et grille de verdure ExG, téléchargées
            une fois et partagées avec la segmentation des houppiers.

    Returns:
        dict: source ('lidar_hd' | 'mns_mnt' | None), resolution_m, ortho, et
        `toits` : {cleabs: profil}. Vide si l'emprise n'est couverte par
        aucune source d'altitude.
    """
    from shapely.ops import transform

    west, south, east, north = map(float, (west, south, east, north))
    resultat = {"source": grille.get("source"), "resolution_m": TOITS_RESOLUTION_M,
                "toits": {}, "ortho": exg is not None}
    if not grille.get("couvert"):
        return resultat
    lon0, lat0 = (west + east) / 2, (south + north) / 2
    m_lon = 111320 * math.cos(math.radians(lat0))
    cellules = _cellules_locales(grille, lon0, lat0)
    # Second avis de l'orthophoto : sans elle, la canopée passe pour un toit.
    verdure = _verdure_locale(exg, grille["bbox"], lon0, lat0) if exg is not None else None
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
        if profil:
            part = part_verte(poly, verdure) if verdure is not None else None
            resultat["toits"][cleabs] = qualifier_couvert(profil, part)
    journal.info("Toits LiDAR : %d bâtiment(s) mesuré(s), source=%s",
                 len(resultat["toits"]), resultat["source"])
    return resultat
