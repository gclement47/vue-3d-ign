"""Houppiers mesurés sur le MNH LiDAR HD à 0,5 m.

La version précédente regroupait les cellules du MNH à 2 m côté navigateur, par
prise de sommets gloutonne, et dessinait chaque houppier avec la même
allométrie (rayon = 26 % de la hauteur, houppier sur 80 % de la hauteur).
Un ratio unique produit forcément le ballon : un pin d'Alep est un parasol
aplati sur un fût nu, un chêne vert une boule presque aussi large que haute,
un cyprès une colonne dix fois plus haute que large.

Ici la forme n'est plus supposée mais lue dans la grille : à 0,5 m, un
houppier de 8 m compte deux cents cellules, assez pour un profil. Chaque
houppier est un bassin de la grille lissée (segmentation par descente depuis
les sommets, la méthode courante en foresterie), et porte :

- son sommet (position, hauteur brute), sa surface au sol, son allongement
  et l'axe de celui-ci (une haie est longue, un arbre isolé rond) ;
- son profil radial : hauteur moyenne par couronne de distance au sommet,
  rapportée à la hauteur du sommet. Un pin donne un plateau, un chêne un
  dôme, un cyprès une pointe ;
- sa nature BD TOPO et son essence BD Forêt v2 quand elles existent, pour la
  palette et le port du tronc — jamais pour la forme.

Le classement d'une cellule reste celui de la vue : sous 2 m ignorée, dans
une emprise bâtie ignorée, dans une zone de végétation BD TOPO végétation,
sinon végétation si l'orthophoto est verte (ExG >= 4), sinon masse
indéterminée (mur, véhicule, sol nu), segmentée de la même façon mais bridée
à 3 m de rayon. Hors des forêts BD TOPO, une cellule de plus de 40 m n'est ni
l'un ni l'autre, et n'est pas dessinée (SURSOL_HAUTEUR_MAX_M).
"""

import logging
import math

import numpy as np
import shapely
from shapely.geometry import shape

from .mnh import MNH_SEUIL_M
from .ortho import EXG_SEUIL

journal = logging.getLogger(__name__)

HOUPPIERS_RESOLUTION_M = 0.5
# Rayon de recherche d'un sommet, d'après la hauteur : h / 4, entre 2 et
# 6 m. Mesuré sur 8 ha de forêt mixte méditerranéenne, 4,6 ha de canopée :
# - h / 2,5 borné à 9 m (l'ancienne règle du regroupement à 2 m) : 1 470
#   houppiers, mais les plus hauts pins avalent leurs voisins sur un disque
#   parfait de 9 m (1 010 cellules, allongement 1,0) ;
# - la demi-largeur de Popescu & Wynne (3,0 + 0,006 h²) / 2 : 4 260
#   houppiers de 1,5 m de rayon médian, trop fins pour des chênes verts ;
# - h / 4 : entre les deux, 2,5 m à 10 m de haut, 6 m à 25 m.
RAYON_MIN_M = 2.0
RAYON_FENETRE_MAX_M = 6.0
RAYON_MAX_VEGETATION_M = 9.0
# Une masse indéterminée n'est pas un arbre : au-delà, un mur ou un talus
# devient un rocher qui écrase la scène.
RAYON_MAX_MASSE_M = 3.0
# Un houppier s'étend au-delà de la fenêtre où l'on cherche son sommet : la
# fenêtre sert à ne pas prendre deux sommets sur le même arbre, l'étendue à
# ne pas laisser sa lisière orpheline. Mesuré sur 8 ha de forêt mixte avec
# la même valeur pour les deux : 709 sommets et 5 400 orphelins, chacun un
# anneau de quelques cellules autour d'un houppier bridé.
ETENDUE_FACTEUR = 2.0
# En deçà, ce n'est pas un houppier : 6 cellules à 0,5 m font 1,5 m².
HOUPPIERS_MIN_CELLULES = 6
# Couronnes du profil radial.
PROFIL_BINS = 6
# Allongement maximal transmis : au-delà, une haie de 40 m est une ligne.
ALLONGEMENT_MAX = 4.0
FORET_LAYER = "LANDCOVER.FORESTINVENTORY.V2:formation_vegetale"
# Hors forêt, plafond de ce qui peut être un arbre ou une masse. Le MNH garde
# ce qui passe au-dessus du sol, et l'orthophoto ne dit que la couleur de ce
# sol : à Notre-Dame de Paris, les flèches des deux grues du chantier, au-dessus
# des arbres des quais, sortaient en houppiers de 52 à 88 m — plus hauts que
# les tours (69 m) — et le débord des tours sur leur emprise en houppiers et en
# masses de 60 à 66 m. Cellules de plus de 40 m hors de tout bâtiment, sur 22
# lieux dont six forêts, par nature de zone BD TOPO :
#
#   forêt fermée     261   de vrais arbres : sapins des Vosges, jusqu'à 43,3 m
#   bois              63   toutes sous une flèche de grue, à Notre-Dame
#   haie, vigne        0
#   hors zone     50 251   grues et débords de tours à Notre-Dame (542) et à la
#                          Part-Dieu (75), toit du Stade de France au-delà de
#                          son emprise (49 634)
#
# Hors forêt, le plus haut houppier plausible est à 35,4 m ; en forêt, 99
# dépassent 35 m dans le Jura, 10 dépassent 40 m dans les Vosges. D'où 40 m,
# hors des seules forêts BD TOPO : les arbres records du pays (douglas de plus
# de 60 m) y sont, et rien n'y est plafonné. Ce qui dépasse n'est pas dessiné —
# ni la grue, ni le toit d'un stade hors de son emprise, qui ne faisait qu'une
# couronne de masses de 46 m. Reste, en forêt, l'artefact des falaises : à
# Rocamadour, 7 houppiers de 42 à 60 m sur des chênes du causse, dont le MNH
# compte la hauteur depuis le pied de la paroi.
SURSOL_HAUTEUR_MAX_M = 40.0
NATURES_FORET = "Forêt"


# Fenêtre de lissage, en cellules de part et d'autre : 2 -> 5 × 5, soit
# 2,5 m à 0,5 m de maille. Mesuré sur 8 ha de forêt mixte méditerranéenne : en 3 × 3,
# 3 582 houppiers de 1,3 m de rayon médian (un par branche) ; en 5 × 5, un
# nombre compatible avec une futaie (une centaine à l'hectare).
LISSAGE_RAYON = 2


def lisser(grille, rayon=LISSAGE_RAYON):
    """Moyenne glissante carrée, bords répliqués : gomme les branches pour ne
    garder que les sommets d'arbres. Un bruit déterministe infime lève les
    égalités : sur une cime plate, deux cellules de même hauteur lissée
    feraient deux sommets."""
    p = np.pad(grille, rayon, mode="edge")
    s = np.zeros_like(grille, dtype=np.float32)
    n = 2 * rayon + 1
    for dj in range(n):
        for di in range(n):
            s += p[dj:dj + grille.shape[0], di:di + grille.shape[1]]
    s /= n * n
    J, I = np.indices(grille.shape)
    bruit = np.sin(J * 12.9898 + I * 78.233) * 43758.5453
    return s + (bruit - np.floor(bruit)).astype(np.float32) * 1e-3


# Les huit voisins d'une cellule, dans l'ordre où la descente départage deux
# voisins de même score.
VOISINS = [(dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]
PAS_J = np.array([dj for dj, _ in VOISINS], dtype=np.intp)
PAS_I = np.array([di for _, di in VOISINS], dtype=np.intp)


def _dilater(grille):
    """Maximum sur le voisinage 3 × 3 (dilatation carrée de rayon 1), bords
    répliqués.

    Séparée : le maximum des trois colonnes, puis celui des trois lignes —
    quatre comparaisons par cellule au lieu de neuf, et ni bordure ni copie
    de la grille. Le maximum ne dépend pas de l'ordre où l'on compare, aux
    zéros signés près, que les comparaisons qui suivent ne distinguent pas.
    """
    lignes = grille.copy()
    np.maximum(lignes[:, 1:], grille[:, :-1], out=lignes[:, 1:])
    np.maximum(lignes[:, :-1], grille[:, 1:], out=lignes[:, :-1])
    out = lignes.copy()
    np.maximum(out[1:], lignes[:-1], out=out[1:])
    np.maximum(out[:-1], lignes[1:], out=out[:-1])
    return out


def _indices(sel):
    """Lignes et colonnes des cellules vraies, comme np.nonzero : à plat
    puis divisées, dix fois plus vite sur une grille (mesuré sur 2,9 M
    cellules : 0,6 ms contre 5,4 ms)."""
    return np.divmod(np.flatnonzero(sel), sel.shape[1])


def rayon_cellules(hauteur, pas, rayon_max):
    """Rayon de recherche d'un sommet, en cellules, d'après sa hauteur."""
    r = np.clip(hauteur / 4.0, RAYON_MIN_M, min(rayon_max, RAYON_FENETRE_MAX_M))
    return np.maximum(np.round(r / pas), 1).astype(np.int32)


def sommets(lisse, masque, rayons):
    """Maxima locaux de la grille lissée, chacun dans sa propre fenêtre.

    Une cellule est un sommet si elle égale le maximum du carré de rayon
    r(h) autour d'elle, limité à la grille.

    Seul un maximum du carré 3 × 3 peut l'être : la recherche ne va plus
    loin que pour ceux-là, couronne après couronne, et s'arrête pour chacun
    à la première qui le dépasse — le maximum d'un carré est celui de ses
    couronnes. Dilater toute la grille douze fois, une par rayon, prenait
    72 ms sur Gordes en zone de 1 000 m (houppiers), 212 ms à Strasbourg ;
    10 et 26 ms ainsi, pour les mêmes sommets (machine chargée, médiane de
    trois).
    """
    base = np.where(masque, lisse, -1.0).astype(np.float32)
    marqueurs = np.zeros(lisse.shape, dtype=bool)
    r_max = int(rayons.max()) if rayons.size else 0
    if r_max < 1:
        return marqueurs
    ny, nx = base.shape
    candidats = np.flatnonzero(masque & (base >= _dilater(base)))
    r = rayons.reshape(-1)[candidats]
    h = base.reshape(-1)[candidats]
    # Bordée de -inf sur r_max cellules : hors de la grille, rien ne dépasse.
    largeur = nx + 2 * r_max
    bordee = np.pad(base, r_max, mode="constant", constant_values=-np.inf).reshape(-1)
    cj, ci = np.divmod(candidats, nx)
    position = (cj + r_max) * largeur + (ci + r_max)
    vivants = r >= 1
    for k in range(2, r_max + 1):
        a_voir = np.flatnonzero(vivants & (r >= k))
        if not a_voir.size:
            break
        couronne = np.array([dj * largeur + di for dj in range(-k, k + 1)
                             for di in range(-k, k + 1) if max(abs(dj), abs(di)) == k],
                            dtype=np.intp)
        plus_haut = bordee[position[a_voir][:, None] + couronne].max(axis=1)
        vivants[a_voir[~(h[a_voir] >= plus_haut)]] = False
    marqueurs.reshape(-1)[candidats[vivants]] = True
    return marqueurs


def segmenter(lisse, masque, rayons, etendue_max, stats=None):
    """Rattache chaque cellule du masque au sommet dont elle descend : voir
    _segmenter, dont les sommets sont ici rendus en liste de (j, i)."""
    labels, apex_j, apex_i = _segmenter(lisse, masque, rayons, etendue_max, stats)
    return labels, [(0, 0)] + list(zip(apex_j[1:].tolist(), apex_i[1:].tolist()))


def _segmenter(lisse, masque, rayons, etendue_max, stats=None):
    """Rattache chaque cellule du masque au sommet dont elle descend.

    Descente vectorisée : à chaque passe, une cellule non étiquetée prend
    l'étiquette de son plus haut voisin étiqueté, si le sommet de cette
    étiquette est à moins de r(sommet) ; les voisins qui ne sont pas plus
    bas qu'elle passent avant les autres, pour qu'une cellule descende vers
    le bassin d'où elle vient. On itère jusqu'à convergence. Les cellules
    qu'aucun sommet n'atteint (une cime qu'aucun rayon ne couvre) deviennent
    à leur tour des sommets, et la descente reprend.

    Exiger un voisin strictement plus haut, comme le fait un parcours en
    ordre décroissant, ne convient pas à une descente par passes : une bosse
    sur le flanc d'un houppier voit tous ses voisins étiquetés plus bas
    qu'elle, reste orpheline et devient un sommet. Mesuré sur 8 ha de forêt
    mixte : 3 500 houppiers de 1,3 m de rayon médian avec l'exigence, contre
    un nombre compatible avec une futaie sans elle.

    Les segments plus petits que HOUPPIERS_MIN_CELLULES sont fondus dans le
    voisin le plus haut, ou abandonnés s'ils n'en ont pas.

    Returns:
        labels (int32, 0 = hors masque), lignes et colonnes du sommet de
        chaque étiquette (index 1..n ; l'index 0 ne sert pas). `etendue_max`
        borne l'étendue d'un segment, en cellules.
    """
    ny, nx = lisse.shape
    largeur = nx + 2
    # Bordées d'une cellule vide : un voisin hors grille n'a ni étiquette, ni
    # hauteur, ni place à prendre. Les cellules y sont repérées par leur
    # indice à plat : leurs huit voisins sont à des décalages fixes.
    # `labels` et `restants` sont les vues sans la bordure, écrites en place.
    labels_bordes = np.zeros((ny + 2, nx + 2), dtype=np.int32)
    labels = labels_bordes[1:-1, 1:-1]
    labels_plat = labels_bordes.reshape(-1)
    decalages = np.array([dj * largeur + di for dj, di in VOISINS], dtype=np.intp)
    restants_bordes = np.zeros((ny + 2, nx + 2), dtype=bool)
    restants_bordes[1:-1, 1:-1] = masque
    restants_plat = restants_bordes.reshape(-1)
    bordure_plat = np.ones((ny + 2, nx + 2), dtype=bool)
    bordure_plat[1:-1, 1:-1] = False
    bordure_plat = bordure_plat.reshape(-1)
    hauteur_ref = np.where(masque, lisse, -1.0).astype(np.float32)
    hauteur_plat = np.pad(hauteur_ref, 1, mode="constant", constant_values=-1.0).reshape(-1)
    # Ligne et colonne du sommet de chaque étiquette ; l'étiquette 0 n'en a pas.
    apex_j = np.zeros(1, dtype=np.intp)
    apex_i = np.zeros(1, dtype=np.intp)

    def poser_marqueurs(cellules):
        """Nouveaux sommets, en indices à plat croissants (l'ordre des
        lignes, comme np.nonzero) : une étiquette chacun."""
        nonlocal apex_j, apex_i
        labels_plat[cellules] = np.arange(len(apex_j), len(apex_j) + len(cellules))
        restants_plat[cellules] = False
        js, is_ = np.divmod(cellules, largeur)
        apex_j = np.concatenate([apex_j, js - 1])
        apex_i = np.concatenate([apex_i, is_ - 1])

    js, is_ = _indices(sommets(lisse, masque, rayons))
    poser_marqueurs((js + 1) * largeur + (is_ + 1))
    if stats is not None:
        stats["marqueurs"] = len(apex_j) - 1
    # Dédoublonnage d'un front sans tri : chaque cellule y garde une seule
    # de ses occurrences.
    vu = np.empty(labels_plat.size, dtype=np.intp)
    for _ in range(64):
        restants = np.flatnonzero(restants_plat)
        if not restants.size:
            break
        progres = False
        ray = np.minimum(rayons[apex_j, apex_i] * ETENDUE_FACTEUR, etendue_max).astype(np.float32)
        ray[0] = 0
        # Seules les cellules restantes au contact d'une étiquette peuvent en
        # recevoir une. Le front complet n'est cherché sur toute la grille
        # qu'une fois par reprise ; ensuite, seules les voisines des cellules
        # que la passe vient d'étiqueter sont évaluées. Une cellule du front
        # qui n'a rien reçu n'a pas à l'être de nouveau tant qu'aucune
        # voisine ne change : son évaluation ne dépend que des étiquettes de
        # ses voisines, des hauteurs et des rayons, fixes pendant la reprise.
        # Mêmes cellules étiquetées à chaque passe, donc même nombre de
        # passes et même scène à l'octet. Mesuré sur Gordes en zone de
        # 1 000 m (19 312 houppiers, 67 passes) : la descente prenait 1,4 s
        # à réévaluer toute la grille et tout le front à chaque passe ; la
        # segmentation entière, sommets et fusion compris, passe de 2,27 s à
        # 0,22 s (machine chargée, médiane de trois).
        cellules = None
        for _ in range(64):
            if cellules is None:
                front = restants[(labels_plat[restants[:, None] + decalages] > 0).any(axis=1)]
                pris, etiquettes = _descendre(front, labels_plat, hauteur_plat, decalages,
                                              largeur, apex_j, apex_i, ray)
            else:
                front, pris, etiquettes = _propager(cellules, etiquettes, labels_plat,
                                                    restants_plat, hauteur_plat, decalages,
                                                    largeur, apex_j, apex_i, ray, vu)
            if not pris.any():
                break
            cellules, etiquettes = front[pris], etiquettes[pris]
            labels_plat[cellules] = etiquettes
            restants_plat[cellules] = False
            progres = True
        restants = np.flatnonzero(restants_plat)
        if not restants.size:
            break
        # Orphelins : leurs maxima locaux (3 × 3) deviennent des sommets. Une
        # voisine qui n'est pas orpheline compte pour -1, une hors de la
        # grille pour rien.
        v = restants[:, None] + decalages
        h_vois = np.where(restants_plat[v], hauteur_plat[v],
                          np.where(bordure_plat[v], np.float32(-np.inf), np.float32(-1.0)))
        nouveaux = restants[hauteur_plat[restants] >= h_vois.max(axis=1)]
        if not nouveaux.size and not progres:
            break
        poser_marqueurs(nouveaux)
    if stats is not None:
        stats["orphelins"] = len(apex_j) - 1 - stats["marqueurs"]
    _fondre_petits(labels_plat, hauteur_plat, decalages, len(apex_j))
    return labels, apex_j, apex_i


def _descendre(front, labels_plat, hauteur_plat, decalages, largeur, apex_j, apex_i, ray):
    """Une passe de la descente sur les cellules du front (indices à plat
    dans la grille bordée) : (prise, étiquette) de chacune.

    Une cellule prend l'étiquette de son voisin étiqueté le plus haut dont le
    sommet est à portée, les voisins qui ne sont pas plus bas qu'elle passant
    avant les autres ; à score égal, le premier dans l'ordre de VOISINS.
    """
    if not front.size:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int32)
    vl = labels_plat[front[:, None] + decalages]
    # Seuls les voisins étiquetés sont départagés : le tiers des huit.
    paires = np.flatnonzero(vl > 0)
    cellule = front[paires >> 3]
    voisin = cellule + decalages[paires & 7]
    l = vl.reshape(-1)[paires]
    vh = hauteur_plat[voisin]
    dist = np.hypot(cellule // largeur - 1 - apex_j[l], cellule % largeur - 1 - apex_i[l])
    score = vh + np.where(vh >= hauteur_plat[cellule], 1000.0, 0.0)
    ok = (dist <= ray[l]) & (score > -1.0)
    # Le premier maximum, comme la comparaison stricte d'un voisin au suivant.
    scores = np.full(vl.shape, -np.inf)
    scores.reshape(-1)[paires[ok]] = score[ok]
    k = scores.argmax(axis=1)
    rangs = np.arange(len(front))
    pris = scores[rangs, k] > -np.inf
    return pris, vl[rangs, k]


def _propager(cellules, etiquettes, labels_plat, restants_plat, hauteur_plat, decalages,
              largeur, apex_j, apex_i, ray, vu):
    """La passe qui suit l'étiquetage de `cellules` (et de leurs
    `etiquettes`) : (front, prise, étiquette), comme _descendre sur le front
    de leurs voisines restantes.

    Une voisine qui n'a rien reçu à la passe précédente n'avait aucun voisin
    étiqueté recevable, et ne peut en avoir de nouveau que parmi `cellules` :
    seules ces paires-là sont départagées, au lieu des huit voisins de
    chaque cellule du front. Le départage reste celui de _descendre : le
    plus haut score, à égalité le premier dans l'ordre de VOISINS. Mesuré
    sur Gordes en zone de 1 000 m (houppiers) : 2,85 millions de paires
    départagées au lieu de 9 millions de voisins lus, et la segmentation
    de 609 à 514 ms, puis 344 ms une fois l'écart au sommet calculé par
    source (machine chargée, médianes de cinq).
    """
    # Ce qui ne dépend que de la source : son écart à son sommet, sa
    # hauteur, la portée de son étiquette. L'écart de la cible au sommet est
    # celui de la source plus le pas qui les sépare.
    sj, si = np.divmod(cellules, largeur)
    ecart_j = sj - 1 - apex_j[etiquettes]
    ecart_i = si - 1 - apex_i[etiquettes]
    h_source = hauteur_plat[cellules]
    portee = ray[etiquettes]
    cibles = (cellules[:, None] + decalages).reshape(-1)
    garde = np.flatnonzero(restants_plat[cibles])
    cible = cibles[garde]
    source = garde >> 3
    pas = garde & 7
    dist = np.hypot(ecart_j[source] + PAS_J[pas], ecart_i[source] + PAS_I[pas])
    vh = h_source[source]
    score = vh + np.where(vh >= hauteur_plat[cible], 1000.0, 0.0)
    ok = (dist <= portee[source]) & (score > -1.0)
    # Une ligne par cible : le front, sans doublon.
    rang = np.arange(len(cible))
    vu[cible] = rang
    front = cible[vu[cible] == rang]
    vu[front] = np.arange(len(front))
    # La source vue de la cible : la direction opposée, VOISINS étant
    # symétrique (le voisin d'indice 7 - k est l'opposé de celui d'indice k).
    scores = np.full((len(front), len(decalages)), -np.inf)
    scores[vu[cible[ok]], 7 - pas[ok]] = score[ok]
    k = scores.argmax(axis=1)
    pris = scores[np.arange(len(front)), k] > -np.inf
    return front, pris, labels_plat[front + decalages[k]]


def _fondre_petits(labels_plat, hauteur_plat, decalages, n_labels):
    """Fond les segments trop petits dans leur plus haut voisin étiqueté.

    Les petits segments naissent aux lisières : une bosse qu'aucun sommet
    n'atteint. Les rattacher au voisin garde la surface du houppier entière ;
    ceux qui n'ont aucun voisin assez grand sont abandonnés (étiquette 0).

    Travaille, comme la descente, sur la grille bordée d'une cellule vide
    (étiquette 0, hauteur -1) et à plat : `decalages` mène aux huit voisins.
    Modifie `labels_plat` en place.
    """
    for _ in range(3):
        n = np.bincount(labels_plat, minlength=n_labels)
        petits = np.zeros(n_labels, dtype=bool)
        petits[1:] = n[1:] < HOUPPIERS_MIN_CELLULES
        # Seules les cellules des petits segments sont évaluées, et non toute
        # la grille huit fois : 116 ms avant, 29 ms ainsi sur Gordes en zone
        # de 1 000 m (houppiers), pour les mêmes étiquettes.
        cibles = np.flatnonzero(petits[labels_plat])
        if not cibles.size:
            return
        # Lues avant toute écriture : les cibles changent toutes ensemble.
        v = cibles[:, None] + decalages
        vl = labels_plat[v]
        vh = hauteur_plat[v]
        meilleur_h = np.full(len(cibles), -1.0, dtype=np.float32)
        meilleur_l = np.zeros(len(cibles), dtype=np.int32)
        for k in range(len(decalages)):
            ok = (vl[:, k] > 0) & ~petits[vl[:, k]] & (vh[:, k] > meilleur_h)
            meilleur_h = np.where(ok, vh[:, k], meilleur_h)
            meilleur_l = np.where(ok, vl[:, k], meilleur_l)
        labels_plat[cibles] = meilleur_l


def _fenetre(geom, lons, lats):
    """Tranches (lignes, colonnes) de la grille que couvre la boîte de `geom`.

    Tester un polygone contre la grille entière coûte autant pour une remise
    que pour une forêt. Mesuré sur Gordes en zone de 1 000 m (677 emprises,
    grille de 1 440 × 1 985) : 33,4 s pour le masque du bâti, 0,03 s ainsi,
    pour le même masque.

    Args:
        lons, lats: coordonnées des colonnes (croissantes) et des lignes
            (décroissantes) de la grille.
    """
    minx, miny, maxx, maxy = geom.bounds
    return (slice(np.searchsorted(-lats, -maxy, "left"), np.searchsorted(-lats, -miny, "right")),
            slice(np.searchsorted(lons, minx, "left"), np.searchsorted(lons, maxx, "right")))


# Classement par blocs (_dedans) : côté d'un bloc en cellules, et nombre de
# centres à tester au-delà duquel on y recourt. Temps processeur des masques
# de la zone de 1 000 m, machine chargée, médiane de trois, pour un test
# centre par centre puis par blocs de 4, 8, 16 et 32 cellules (seuil 4 096) :
#
#                          centre    4      8      16     32
#   Gordes, zones           149     81     77     88    102   ms
#   Strasbourg, bâti        306    331*   258    268    307   ms  (* seuil 1 024)
#
# Sous 4 096 centres, un bâtiment ordinaire : ses côtés touchent la plupart
# des blocs, et les témoins coûtent plus qu'ils n'épargnent (seuil 512 :
# 366 ms à Strasbourg).
BLOC = 8
BLOCS_SEUIL = 4096


def _dedans(geom, fenetre, choisies, lons, lats):
    """(lignes, colonnes), dans la fenêtre, des cellules choisies dont `geom`
    contient le centre.

    Seules les cellules dont la réponse compte sont testées : c'est le test
    qui coûte, pas le parcours de la fenêtre. Les rivières de Strasbourg en
    zone de 1 000 m ont pour boîte presque toute la grille : 2,2 millions de
    centres testés chacune, 0,1 à 0,16 s ; 0,35 million une fois retirés le
    sol et les toits déjà masqués.
    """
    jj, ii = _indices(choisies)
    if not jj.size:
        return jj, ii
    x, y = lons[fenetre[1]], lats[fenetre[0]]
    if jj.size < BLOCS_SEUIL or geom.geom_type not in ("Polygon", "MultiPolygon"):
        d = shapely.contains_xy(geom, x[ii], y[jj])
        return jj[d], ii[d]
    # Par blocs de BLOC × BLOC cellules. Un centre est dedans si une demi-
    # droite qui en part coupe les anneaux un nombre impair de fois (c'est
    # ainsi que GEOS le détermine, en arithmétique exacte) : la réponse ne
    # change donc pas d'un centre à l'autre d'un bloc qu'aucun côté du
    # polygone ne touche, et un seul centre y est testé. Un côté touche
    # peut-être le bloc si sa boîte touche celle des centres du bloc
    # (comparaisons exactes, bornes comprises) ; ces blocs-là sont testés
    # centre par centre.
    ny, nx = len(y), len(x)
    nby, nbx = -(-ny // BLOC), -(-nx // BLOC)
    debut_i = np.arange(nbx) * BLOC
    debut_j = np.arange(nby) * BLOC
    # Boîtes des blocs : colonnes d'ouest en est, lignes du nord au sud.
    x0, x1 = x[debut_i], x[np.minimum(debut_i + BLOC, nx) - 1]
    y1, y0 = y[debut_j], y[np.minimum(debut_j + BLOC, ny) - 1]
    xy, anneau = shapely.get_coordinates(shapely.get_rings(shapely.get_parts(geom)),
                                         return_index=True)
    cote = np.flatnonzero(anneau[1:] == anneau[:-1])
    sx = np.sort(np.stack([xy[cote, 0], xy[cote + 1, 0]]), axis=0)
    sy = np.sort(np.stack([xy[cote, 1], xy[cote + 1, 1]]), axis=0)
    c_lo = np.searchsorted(x1, sx[0], "left")
    c_hi = np.searchsorted(x0, sx[1], "right") - 1
    # Les lignes vont du nord au sud : latitudes décroissantes, comptées en
    # opposé pour la recherche.
    r_lo = np.searchsorted(-y0, -sy[1], "left")
    r_hi = np.searchsorted(-y1, -sy[0], "right") - 1
    k = (c_lo <= c_hi) & (r_lo <= r_hi)
    # Blocs touchés : somme des rectangles de blocs [r_lo, r_hi] × [c_lo, c_hi]
    # par différences cumulées.
    touches = np.zeros((nby + 1, nbx + 1), dtype=np.int64)
    np.add.at(touches, (r_lo[k], c_lo[k]), 1)
    np.add.at(touches, (r_lo[k], c_hi[k] + 1), -1)
    np.add.at(touches, (r_hi[k] + 1, c_lo[k]), -1)
    np.add.at(touches, (r_hi[k] + 1, c_hi[k] + 1), 1)
    touches = touches.cumsum(axis=0).cumsum(axis=1)[:nby, :nbx].reshape(-1) > 0
    bloc = (jj // BLOC) * nbx + ii // BLOC
    d = np.empty(jj.size, dtype=bool)
    sale = touches[bloc]
    d[sale] = shapely.contains_xy(geom, x[ii[sale]], y[jj[sale]])
    propre = np.flatnonzero(~sale)
    if propre.size:
        # Un témoin par bloc propre, n'importe lequel de ses centres.
        temoin = np.full(nby * nbx, -1, dtype=np.intp)
        temoin[bloc[propre]] = propre
        blocs = np.flatnonzero(temoin >= 0)
        reponse = np.zeros(nby * nbx, dtype=bool)
        t = temoin[blocs]
        reponse[blocs] = shapely.contains_xy(geom, x[ii[t]], y[jj[t]])
        d[propre] = reponse[bloc[propre]]
    return jj[d], ii[d]


def _essences_par_cellule(features, lons, lats, cle, utiles=None):
    """Valeur de `cle` de la zone qui contient chaque cellule, ou None.

    Args:
        lons, lats: coordonnées des colonnes et des lignes de la grille.
        utiles: grille booléenne des seules cellules dont la valeur sera lue,
            ou None pour toutes ; ailleurs, la valeur rendue est -1.
    """
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
        vue = valeurs[fenetre]
        # La première zone qui contient une cellule la garde : seules les
        # cellules encore sans valeur sont testées.
        libres = vue < 0
        if utiles is not None:
            libres &= utiles[fenetre]
        vue[_dedans(geom, fenetre, libres, lons, lats)] = noms.index(nom)
    return valeurs, noms


def decrire(labels, apex, hauteur, xs, ys, pas, classe, natures, noms_nature,
            essences, noms_essence, lons, lats):
    """Un dictionnaire par étiquette : position, hauteur, profil, allongement.

    `apex` : lignes et colonnes du sommet de chaque étiquette, en tableaux.
    """
    aj, ai = apex
    n_labels = len(aj)
    # Les cellules étiquetées, en lignes et colonnes : xs, ys et les
    # coordonnées peuvent être des vues diffusées, jamais recopiées en grille.
    cj, ci = _indices(labels > 0)
    lab = labels[cj, ci]
    h = hauteur[cj, ci].astype(np.float64)
    x = xs[cj, ci].astype(np.float64)
    y = ys[cj, ci].astype(np.float64)
    n = np.bincount(lab, minlength=n_labels)
    # Hauteur brute maximale du segment, pas celle du sommet lissé.
    h_max = np.zeros(n_labels); np.maximum.at(h_max, lab, h)
    ax = xs[aj, ai]; ay = ys[aj, ai]
    d = np.hypot(x - ax[lab], y - ay[lab])
    d_max = np.zeros(n_labels); np.maximum.at(d_max, lab, d)
    part = np.clip(np.floor(d / np.maximum(d_max[lab], pas) * PROFIL_BINS), 0, PROFIL_BINS - 1).astype(np.int64)
    cle = lab * PROFIL_BINS + part
    somme_h = np.bincount(cle, weights=h, minlength=n_labels * PROFIL_BINS)
    nb = np.bincount(cle, minlength=n_labels * PROFIL_BINS)
    # Moments pour l'allongement et l'axe.
    sx = np.bincount(lab, weights=x, minlength=n_labels)
    sy = np.bincount(lab, weights=y, minlength=n_labels)
    sxx = np.bincount(lab, weights=x * x, minlength=n_labels)
    syy = np.bincount(lab, weights=y * y, minlength=n_labels)
    sxy = np.bincount(lab, weights=x * y, minlength=n_labels)

    # Tout ce qui n'est qu'addition, produit, quotient, racine, minimum et
    # maximum est calculé pour tous les segments à la fois : ces opérations
    # sont exactes à l'arrondi IEEE près, en tableau comme une à une, et
    # donnent les mêmes nombres que la boucle d'un segment à l'autre qui
    # les faisait jusqu'ici. Reste un à un l'arc tangente, que numpy ne
    # calcule pas forcément comme la bibliothèque mathématique de Python ;
    # les arrondis décimaux passent par _arrondir.
    garde = np.flatnonzero(n >= HOUPPIERS_MIN_CELLULES)
    garde = garde[garde > 0]
    nk = n[garde]
    mx, my = sx[garde] / nk, sy[garde] / nk
    cxx = sxx[garde] / nk - mx * mx + pas * pas / 12
    cyy = syy[garde] / nk - my * my + pas * pas / 12
    cxy = sxy[garde] / nk - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-9)
    allongements = np.minimum(np.sqrt(l1 / l2), ALLONGEMENT_MAX)
    axes = [math.degrees(0.5 * math.atan2(a, b)) % 180
            for a, b in zip((2 * cxy).tolist(), (cxx - cyy).tolist())]
    hk = h_max[garde]
    # Profil : hauteur moyenne de chaque couronne rapportée au sommet ; une
    # couronne vide reprend la valeur de la précédente (1 avant la première).
    nb_k = nb.reshape(-1, PROFIL_BINS)[garde]
    with np.errstate(divide="ignore", invalid="ignore"):
        rapports = (somme_h.reshape(-1, PROFIL_BINS)[garde] / nb_k
                    / np.maximum(hk, 0.1)[:, None])
    profils = np.empty_like(rapports)
    dernier = np.ones(len(garde))
    for b in range(PROFIL_BINS):
        dernier = np.where(nb_k[:, b] > 0, rapports[:, b], dernier)
        profils[:, b] = np.minimum(dernier, 1.0)
    profils = _arrondir(profils.reshape(-1), 2)
    gj, gi = aj[garde], ai[garde]
    nature = natures[gj, gi]
    essence = essences[gj, gi]

    out = []
    for k, (lon, lat, h_k, r, r_max, allongement, axe, n_k, na, es) in enumerate(zip(
            _arrondir(lons[gj, gi], 7), _arrondir(lats[gj, gi], 7), _arrondir(hk, 1),
            # Rayon équivalent de la surface, et rayon réel jusqu'au bord.
            _arrondir(np.sqrt(nk * pas * pas / math.pi), 2),
            _arrondir(d_max[garde] + pas / 2, 2),
            _arrondir(allongements, 2), _arrondir(axes, 1),
            nk.tolist(), nature.tolist(), essence.tolist())):
        out.append({
            "lon": lon, "lat": lat, "h": h_k, "r": r, "r_max": r_max,
            "profil": profils[k * PROFIL_BINS:(k + 1) * PROFIL_BINS],
            "allongement": allongement, "axe_deg": axe, "n": n_k,
            "classe": classe,
            "nature": noms_nature[na] if na >= 0 else None,
            "essence": noms_essence[es] if es >= 0 else None,
        })
    return out


def _arrondir(valeurs, chiffres):
    """[round(v, chiffres) for v in valeurs], à l'identique, en tableau.

    round() de Python arrondit la valeur binaire exacte à `chiffres`
    décimales (au pair en cas d'égalité), puis rend le double le plus proche
    de ce décimal. En tableau : q = rint(v · 10^chiffres), puis q / 10^chiffres,
    dont la division IEEE rend ce même double le plus proche. Le seul écart
    possible est dans q, quand le produit arrondi et le produit exact (à une
    demi-unité de dernier rang l'un de l'autre) tombent de part et d'autre
    d'une demi-unité ; ces valeurs-là, et les non finies, passent par round()
    lui-même. Mesuré sur 286 000 valeurs : 63 ms pour round(), 7 ms ainsi.
    """
    v = np.asarray(valeurs, dtype=np.float64)
    echelle = 10.0 ** chiffres
    produit = v * echelle
    out = (np.rint(produit) / echelle).tolist()
    with np.errstate(invalid="ignore"):
        sur = np.abs(produit - np.floor(produit) - 0.5) > 4 * np.spacing(np.abs(produit))
    for k in np.flatnonzero(~sur).tolist():
        out[k] = round(float(v[k]), chiffres)
    return out


def houppiers_pour_emprise(west, south, east, north, batiments, vegetation, forets,
                           grille, exg):
    """Houppiers et masses de sursol d'une emprise, d'après le MNH à 0,5 m.

    Args:
        batiments, vegetation, forets: GeoJSON (dict) des couches BD TOPO
            bâtiment et zone_de_vegetation, et BD Forêt v2 ; None si la
            couche n'a pas répondu — le résultat le dit.
        grille, exg: grille MNH et grille ExG, partagées avec les toitures.
    Returns:
        dict: source, couvert, resolution_m, seuil_m, ortho, veg_disponible,
        foret_disponible, houppiers, masses, hauteur_max, nb_ortho.
    """
    resultat = {
        "source": grille.get("source"), "couvert": bool(grille.get("couvert")),
        "resolution_m": HOUPPIERS_RESOLUTION_M, "seuil_m": MNH_SEUIL_M,
        "ortho": exg is not None, "veg_disponible": vegetation is not None,
        "foret_disponible": forets is not None,
        "houppiers": [], "masses": [], "hauteur_max": 0.0, "nb_ortho": 0,
    }
    if grille.get("couvert"):
        resultat.update(segmenter_emprise(grille, exg, batiments, vegetation, forets))
    journal.info("Houppiers : %d houppier(s), %d masse(s), source=%s",
                 len(resultat["houppiers"]), len(resultat["masses"]), resultat["source"])
    return resultat


def segmenter_emprise(grille, exg, batiments, vegetation, forets):
    """Cœur du calcul, sans réseau ni cache : classement puis segmentation."""
    west, south, east, north = grille["bbox"]
    nx, ny = grille["width"], grille["height"]
    H = np.asarray(grille["values"], dtype=np.float32).reshape(ny, nx)
    seuil = float(grille.get("seuil_m") or MNH_SEUIL_M)
    lat0 = (south + north) / 2
    m_lon = 111320 * math.cos(math.radians(lat0))
    pas = (east - west) * m_lon / nx
    lons = west + (east - west) * (np.arange(nx) + 0.5) / nx
    lats = north - (north - south) * (np.arange(ny) + 0.5) / ny
    # Coordonnées de chaque cellule, sans les recopier ligne à ligne : une
    # colonne a la même longitude du nord au sud, une ligne la même latitude.
    LON = np.broadcast_to(lons, (ny, nx))
    LAT = np.broadcast_to(lats[:, None], (ny, nx))
    xs = np.broadcast_to(((lons - (west + east) / 2) * m_lon).astype(np.float32), (ny, nx))
    ys = np.broadcast_to(((lats - lat0) * 111320).astype(np.float32)[:, None], (ny, nx))

    sursol = H >= seuil
    # Le bâti, les zones de végétation et la forêt ne sont lus qu'au sursol
    # (`candidats` et `nb_ortho` ne regardent rien d'autre), et le bâti
    # seulement là où il n'est pas déjà : mêmes masques là où ils comptent,
    # pour une fraction des tests.
    bati = np.zeros((ny, nx), dtype=bool)
    for f in (batiments or {}).get("features", []):
        try:
            geom = shape(f["geometry"])
            fenetre = _fenetre(geom, lons, lats)
            vue = bati[fenetre]
            vue[_dedans(geom, fenetre, sursol[fenetre] & ~vue, lons, lats)] = True
        except Exception:
            continue
    natures, noms_nature = _essences_par_cellule(
        (vegetation or {}).get("features", []), lons, lats, "nature", sursol & ~bati)
    vegetal = natures >= 0
    # Ni arbre ni masse : une grue, le débord d'une tour (SURSOL_HAUTEUR_MAX_M).
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
    segments = {}
    for cle, masque, rayon_max in (("houppiers", masque_veg, RAYON_MAX_VEGETATION_M),
                                   ("masses", masque_masse, RAYON_MAX_MASSE_M)):
        if masque.any():
            rayons = rayon_cellules(lisse, pas, rayon_max)
            labels, apex_j, apex_i = _segmenter(lisse, masque, rayons, rayon_max / pas)
            segments[cle] = labels, (apex_j, apex_i)
    # L'essence n'est lue qu'au sommet de chaque segment (decrire) : la forêt
    # n'est testée qu'en ces cellules, et non sur toute la grille.
    sommets_ = np.zeros((ny, nx), dtype=bool)
    for _, (apex_j, apex_i) in segments.values():
        sommets_[apex_j[1:], apex_i[1:]] = True
    essences, noms_essence = _essences_par_cellule(
        (forets or {}).get("features", []), lons, lats, "essence", sommets_)
    for cle in ("houppiers", "masses"):
        if cle not in segments:
            out[cle] = []
            continue
        labels, apex = segments[cle]
        out[cle] = decrire(labels, apex, H, xs, ys, pas,
                           "vegetation" if cle == "houppiers" else "sursol",
                           natures, noms_nature, essences, noms_essence, LON, LAT)
    return out
