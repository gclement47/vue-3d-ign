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


def _dilater(grille):
    """Maximum sur le voisinage 3 × 3 (dilatation carrée de rayon 1)."""
    p = np.pad(grille, 1, mode="edge")
    out = grille.copy()
    for dj in range(3):
        for di in range(3):
            np.maximum(out, p[dj:dj + grille.shape[0], di:di + grille.shape[1]], out=out)
    return out


def _decales(grille, remplissage):
    """Les huit voisins de chaque cellule, sous forme de huit grilles."""
    p = np.pad(grille, 1, mode="constant", constant_values=remplissage)
    ny, nx = grille.shape
    return [p[1 + dj:1 + dj + ny, 1 + di:1 + di + nx]
            for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]


def rayon_cellules(hauteur, pas, rayon_max):
    """Rayon de recherche d'un sommet, en cellules, d'après sa hauteur."""
    r = np.clip(hauteur / 4.0, RAYON_MIN_M, min(rayon_max, RAYON_FENETRE_MAX_M))
    return np.maximum(np.round(r / pas), 1).astype(np.int32)


def sommets(lisse, masque, rayons):
    """Maxima locaux de la grille lissée, chacun dans sa propre fenêtre.

    Une cellule est un sommet si elle égale le maximum du carré de rayon
    r(h) autour d'elle. Les fenêtres sont obtenues par dilatations successives
    de rayon 1 : après k passes, la dilatation vaut le maximum du carré de
    rayon k.
    """
    base = np.where(masque, lisse, -1.0).astype(np.float32)
    courant = base.copy()
    marqueurs = np.zeros(lisse.shape, dtype=bool)
    for k in range(1, int(rayons.max()) + 1):
        courant = _dilater(courant)
        marqueurs |= (rayons == k) & masque & (base >= courant)
    return marqueurs


def segmenter(lisse, masque, rayons, etendue_max, stats=None):
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
        labels (int32, 0 = hors masque), liste d'apex (j, i) par étiquette
        (index 1..n). `etendue_max` borne l'étendue d'un segment, en cellules.
    """
    ny, nx = lisse.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    marqueurs = sommets(lisse, masque, rayons)
    apex = [(0, 0)]
    J, I = np.indices((ny, nx))

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
    for _ in range(64):
        if not restants.any():
            break
        progres = False
        for _ in range(64):
            aj = np.array([a[0] for a in apex], dtype=np.int32)
            ai = np.array([a[1] for a in apex], dtype=np.int32)
            ray = np.minimum(rayons[aj, ai] * ETENDUE_FACTEUR, etendue_max).astype(np.float32)
            ray[0] = 0
            meilleur_h = np.full((ny, nx), -1.0, dtype=np.float32)
            meilleur_l = np.zeros((ny, nx), dtype=np.int32)
            for vl, vh in zip(_decales(labels, 0), _decales(hauteur_ref, -1.0)):
                # Voisin étiqueté, sommet à portée ; priorité aux voisins qui ne
                # sont pas plus bas que la cellule.
                dist = np.hypot(J - aj[vl], I - ai[vl])
                score = vh + np.where(vh >= hauteur_ref, 1000.0, 0.0)
                ok = (vl > 0) & (dist <= ray[vl]) & (score > meilleur_h)
                meilleur_h = np.where(ok, score, meilleur_h)
                meilleur_l = np.where(ok, vl, meilleur_l)
            nouveau = restants & (meilleur_l > 0)
            if not nouveau.any():
                break
            labels[nouveau] = meilleur_l[nouveau]
            restants &= ~nouveau
            progres = True
        if not restants.any():
            break
        # Orphelins : leurs maxima locaux (3 × 3) deviennent des sommets.
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
    """Fond les segments trop petits dans leur plus haut voisin étiqueté.

    Les petits segments naissent aux lisières : une bosse qu'aucun sommet
    n'atteint. Les rattacher au voisin garde la surface du houppier entière ;
    ceux qui n'ont aucun voisin assez grand sont abandonnés (étiquette 0).
    Modifie `labels` en place.
    """
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
    """Valeur de `cle` de la zone qui contient chaque cellule, ou None."""
    valeurs = np.full(lons.shape, -1, dtype=np.int32)
    noms = []
    for f in features or []:
        try:
            geom = shape(f["geometry"])
        except Exception:
            continue
        nom = (f.get("properties") or {}).get(cle)
        if nom not in noms:
            noms.append(nom)
        dedans = shapely.contains_xy(geom, lons, lats)
        valeurs[dedans & (valeurs < 0)] = noms.index(nom)
    return valeurs, noms


def decrire(labels, apex, hauteur, xs, ys, pas, classe, natures, noms_nature,
            essences, noms_essence, lons, lats):
    """Un dictionnaire par étiquette : position, hauteur, profil, allongement."""
    n_labels = len(apex)
    flat = labels.ravel()
    ok = flat > 0
    lab = flat[ok]
    h = hauteur.ravel()[ok].astype(np.float64)
    x = xs.ravel()[ok].astype(np.float64)
    y = ys.ravel()[ok].astype(np.float64)
    n = np.bincount(lab, minlength=n_labels)
    aj = np.array([a[0] for a in apex]); ai = np.array([a[1] for a in apex])
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
            # Rayon équivalent de la surface, et rayon réel jusqu'au bord.
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
    LON, LAT = np.meshgrid(lons, lats)
    xs = ((LON - (west + east) / 2) * m_lon).astype(np.float32)
    ys = ((LAT - lat0) * 111320).astype(np.float32)

    sursol = H >= seuil
    bati = np.zeros((ny, nx), dtype=bool)
    for f in (batiments or {}).get("features", []):
        try:
            bati |= shapely.contains_xy(shape(f["geometry"]), LON, LAT)
        except Exception:
            continue
    natures, noms_nature = _essences_par_cellule(
        (vegetation or {}).get("features", []), LON, LAT, "nature")
    essences, noms_essence = _essences_par_cellule(
        (forets or {}).get("features", []), LON, LAT, "essence")
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
