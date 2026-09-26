"""Profil de toit LiDAR (vue3d/toits.py), sur une grille synthétique."""
import math

from shapely.geometry import box, Point

from vue3d.toits import profil_toit, TOITS_MIN_CELLULES


def _grille_toit_deux_pans(largeur=12, longueur=20, gout=4.0, fait=7.0, pas=0.5):
    """Maison rectangulaire : faîtage le long de l'axe X (le grand côté)."""
    cellules = []
    x = -longueur / 2
    while x <= longueur / 2:
        y = -largeur / 2
        while y <= largeur / 2:
            # Deux pans : la hauteur décroît linéairement en s'éloignant du faîtage (y=0).
            h = fait - (fait - gout) * abs(y) / (largeur / 2)
            cellules.append((x, y, h))
            y += pas
        x += pas
    return cellules, box(-longueur / 2, -largeur / 2, longueur / 2, largeur / 2)


def test_gouttiere_et_faitage_retrouves():
    cellules, emprise = _grille_toit_deux_pans()
    t = profil_toit(cellules, emprise)
    assert t is not None
    # Profil linéaire de 4 m (bord) à 7 m (faîtage) sur 6 m de demi-largeur.
    # Gouttière lue à 1 m du bord au plus (4 + 3/6 = 4.5) puis 15e percentile :
    # 5.0 au plus, contre 5.2 avec la seule érosion de 1,5 m. Le faîtage (p85 à
    # 1,5 m d'érosion) frôle 7.
    assert 4.0 <= t["gouttiere"] <= 5.0
    assert 6.2 <= t["faitage"] <= 7.0
    assert t["denivele"] > 1.5


def test_axe_du_faitage_suit_le_grand_cote():
    cellules, emprise = _grille_toit_deux_pans()
    t = profil_toit(cellules, emprise)
    assert t["axe_deg"] is not None
    # Faîtage le long de X -> angle ~0° (ou 180°, même droite).
    assert min(t["axe_deg"], 180 - t["axe_deg"]) < 5
    assert t["nettete"] >= 3


def test_toit_plat_sans_axe():
    cellules, emprise = _grille_toit_deux_pans(gout=5.0, fait=5.0)
    t = profil_toit(cellules, emprise)
    assert t["denivele"] == 0.0
    # Toutes les cellules sont « hautes » : pas de ligne, donc pas d'axe.
    assert t["axe_deg"] is None


def test_emprise_trop_petite():
    cellules, _ = _grille_toit_deux_pans()
    t = profil_toit(cellules, box(-1, -1, 1, 1))
    assert t is None


def test_les_bords_ne_tirent_pas_la_gouttiere_au_sol():
    # Contour à 0 m (cellules de sol qui chevauchent), toit à 6 m au-dedans :
    # l'érosion doit écarter le sol.
    cellules, emprise = _grille_toit_deux_pans(gout=6.0, fait=6.0)
    minx, miny, maxx, maxy = emprise.bounds
    sol = [(x, miny, 0.0) for x in range(int(minx), int(maxx) + 1)]
    t = profil_toit(cellules + sol, emprise)
    assert t["gouttiere"] >= 5.9


def _grille_maison_en_t(pas=0.5):
    """Corps A : 24 x 10 m, faîtage le long de X (y=0). Corps B : 10 x 14 m,
    accolé au milieu, faîtage le long de Y. Les deux à 4 m -> 7 m."""
    from shapely.ops import unary_union
    A = box(-12, -5, 12, 5)
    B = box(-5, 5, 5, 19)
    emprise = unary_union([A, B])
    cellules = []
    x = -12
    while x <= 12:
        y = -5
        while y <= 19:
            p = Point(x, y)
            if A.contains(p) or A.touches(p):
                h = 7 - 3 * abs(y) / 5
            elif B.contains(p) or B.touches(p):
                h = 7 - 3 * abs(x) / 5
            else:
                h = None
            if h is not None:
                cellules.append((x, y, h))
            y += pas
        x += pas
    return cellules, emprise


def test_maison_en_t_donne_deux_corps():
    cellules, emprise = _grille_maison_en_t()
    t = profil_toit(cellules, emprise)
    assert "corps" in t and len(t["corps"]) == 2
    corps = sorted(t["corps"], key=lambda c: -c["part"])
    a, b = corps
    # Le grand corps a son faîtage le long de X (~0°), le petit le long de Y (~90°).
    assert min(a["axe_deg"], 180 - a["axe_deg"]) < 8
    assert abs(b["axe_deg"] - 90) < 8
    for c in corps:
        assert 3.5 <= c["gouttiere"] <= 5.5 and 6.0 <= c["faitage"] <= 7.0
    assert a["longueur"] > b["longueur"]


def test_toit_simple_n_a_pas_de_corps():
    cellules, emprise = _grille_toit_deux_pans()
    t = profil_toit(cellules, emprise)
    assert "corps" not in t


def test_un_arbre_qui_surplombe_est_ecarte_par_le_mode_bas():
    cellules, emprise = _grille_toit_deux_pans()
    t_sain = profil_toit(cellules, emprise)
    assert t_sain["fiable"] is True and t_sain["mode_bas"] is False
    # Houppier de 12 m qui déborde sur le toit : 8 % des cellules, réparties
    # sur toute l'emprise (une sur douze) pour ne pas tomber dans la bande de
    # bord que l'érosion écarte. Le profil est rejeté (p95 dans le feuillage),
    # puis relu dans son mode bas : le toit, sans l'arbre.
    perturbe = [(x, y, 12.0) if i % 12 == 0 else (x, y, h)
                for i, (x, y, h) in enumerate(cellules)]
    t = profil_toit(perturbe, emprise)
    assert t["mode_bas"] is True and t["fiable"] is True
    assert 0.05 <= t["part_haute"] <= 0.12
    assert t["p95"] <= 7.0
    assert abs(t["gouttiere"] - t_sain["gouttiere"]) <= 0.3
    assert abs(t["faitage"] - t_sain["faitage"]) <= 0.3


def test_une_remise_a_moitie_sous_un_arbre_garde_la_hauteur_du_toit():
    # Cas mesuré sur un terrain boisé : 32 m², toit à 2,7 m,
    # feuillage à 9–12 m sur la moitié de l'emprise. Le rejet est juste, mais
    # le repli BD TOPO annonçait 8,2 m de murs.
    import random
    random.seed(1)
    cellules, emprise = _grille_toit_deux_pans(largeur=5, longueur=6.4, gout=2.5, fait=3.0)
    mixte = [(x, y, h) if x < 0 else (x, y, 9 + random.random() * 3) for x, y, h in cellules]
    t = profil_toit(mixte, emprise)
    assert t is not None            # l'érosion adoucie laisse assez de cellules
    assert t["mode_bas"] is True and t["fiable"] is True
    assert 2.4 <= t["gouttiere"] <= 3.0 and t["faitage"] <= 3.1
    assert 0.4 <= t["part_haute"] <= 0.6


def test_un_versant_continu_n_est_pas_coupe_en_deux():
    # Densité uniforme de 2 à 11 m : pas de creux, donc pas de mode bas, et
    # la mesure reste rejetée (pente impossible) plutôt que tronquée.
    cellules, emprise = _grille_toit_deux_pans(largeur=6, longueur=14, gout=2.0, fait=11.0)
    t = profil_toit(cellules, emprise)
    assert t["mode_bas"] is False and t["fiable"] is False


def test_part_verte_et_couvert():
    from vue3d.toits import part_verte, qualifier_couvert, TOITS_COUVERT_PART_VERTE
    import numpy as np
    emprise = box(-3, -3, 3, 3)
    # Grille de couleur : verte (ExG 20) à l'est, toiture (ExG -15) à l'ouest.
    xs = np.arange(-10, 11) / 2.0
    ys = xs[::-1].copy()                      # la grille descend du nord
    exg = np.where(xs[None, :] > 0, 20, -15).repeat(len(ys), axis=0).astype(np.int8)
    part = part_verte(emprise, (exg, xs, ys))
    assert 0.4 <= part <= 0.6
    assert part_verte(box(100, 100, 101, 101), (exg, xs, ys)) is None
    # Emprise entièrement à l'est : tout vert.
    assert part_verte(box(0.6, -3, 3, 3), (exg, xs, ys)) == 1.0
    assert part_verte(box(-3, -3, -0.6, 3), (exg, xs, ys)) == 0.0

    # Plateau LiDAR régulier (forme de toit plat) mais emprise verte : la
    # canopée d'une annexe. Le profil est rejeté et la hauteur inconnue.
    profil = qualifier_couvert({"fiable": True}, 0.66)
    assert profil["sous_couvert"] is True and profil["fiable"] is False
    assert profil["hauteur_inconnue"] is True
    # Toit réel : 2 % de vert, profil fiable -> rien ne change.
    profil = qualifier_couvert({"fiable": True}, 0.02)
    assert profil["sous_couvert"] is False and profil["hauteur_inconnue"] is False
    # Profil rejeté sur une emprise en partie verte : la BD TOPO est
    # contaminée par les mêmes arbres, on ne s'y replie pas.
    profil = qualifier_couvert({"fiable": False}, 0.2)
    assert profil["hauteur_inconnue"] is True
    # Profil rejeté sans vert : autre cause (voisin, antenne), repli permis.
    profil = qualifier_couvert({"fiable": False}, 0.03)
    assert profil["hauteur_inconnue"] is False
    # Sans orthophoto, rien n'est affirmé.
    profil = qualifier_couvert({"fiable": False}, None)
    assert profil["part_verte"] is None and profil["hauteur_inconnue"] is False
    assert TOITS_COUVERT_PART_VERTE > 0.2


def test_une_pente_impossible_pour_la_largeur_est_non_fiable():
    # Annexe de 6 m de large : une montée de 9 m est une pente de 72°, pas un
    # toit — c'est un arbre qui couvre une bonne part de l'emprise.
    cellules, emprise = _grille_toit_deux_pans(largeur=6, longueur=14, gout=2.0, fait=11.0)
    t = profil_toit(cellules, emprise)
    assert t["fiable"] is False
    # La même annexe avec 3 m de montée (45°) reste un toit.
    cellules, emprise = _grille_toit_deux_pans(largeur=6, longueur=14, gout=2.0, fait=5.0)
    assert profil_toit(cellules, emprise)["fiable"] is True


def test_largeur_min_reste_finie_sur_une_emprise_alignee():
    from vue3d.toits import _largeur_min
    import math
    l = _largeur_min(box(0, 0, 20, 8))
    assert math.isfinite(l) and abs(l - 8) < 0.01


def test_un_mode_bas_trop_petit_est_refuse():
    # Cabane de 11 m² entièrement sous les arbres : le LiDAR ne voit aucune
    # cellule au niveau du toit, seulement du feuillage étagé. Le mode bas
    # (8 cellules, 2 m²) n'est pas une toiture — mieux vaut ne rien affirmer
    # et retomber sur la BD TOPO.
    from vue3d.toits import _coupure_bimodale, TOITS_MIN_CELLULES
    hauteurs = sorted([5.6, 5.8, 6.1, 6.3, 6.5, 6.8, 7.1, 7.2,
                       9.4, 10.2, 11.1, 11.6, 14.3, 15.1, 15.6, 16.2, 16.8, 17.1, 17.4, 18.0, 18.2])
    assert len(hauteurs) > TOITS_MIN_CELLULES
    assert _coupure_bimodale(hauteurs) is None


def test_une_traine_d_arbres_clairsemee_laisse_lire_le_toit():
    # Hangar de 104 m² : 154 cellules de toit à 2–3 m, puis une traîne
    # d'arbres de 2 à 8 cellules par mètre jusqu'à 13 m. Le creux est ténu
    # face au pic bas : c'est bien deux modes.
    from vue3d.toits import _coupure_bimodale
    hauteurs = sorted([2.5] * 34 + [3.4] * 120
                      + [4.5] * 4 + [5.5] * 9 + [6.5] * 3 + [7.5] * 5 + [8.5] * 8
                      + [9.5] * 5 + [10.5] * 3 + [11.5] * 4 + [12.5] * 2 + [13.5] * 4)
    coupure = _coupure_bimodale(hauteurs)
    assert coupure is not None and 4.0 <= coupure <= 8.0
    bas = [h for h in hauteurs if h <= coupure]
    assert len(bas) >= 154
