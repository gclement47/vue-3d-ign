"""Profil de toit LiDAR (vue3d/toits.py), sur une grille synthétique."""
import math

import numpy as np
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
    # (8 cellules, 2 m²) n'est pas une toiture — mieux vaut ne rien affirmer :
    # toits_pour_emprise publiera un profil minimal, hauteur inconnue si
    # l'emprise est verte.
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


# --- Surface du toit ------------------------------------------------------

def _grille_surface(fonction, taille=40, pas=0.5):
    """Grille MNH carrée centrée sur l'origine : xs croissants, ys décroissants."""
    import numpy as np
    xs = (np.arange(taille) + 0.5) * pas - taille * pas / 2
    ys = -xs
    X, Y = np.meshgrid(xs, ys)
    return np.vectorize(fonction)(X, Y).astype(np.float32), xs, ys


def _decoder_surface(s):
    """Inverse de _encoder_ecarts : hauteurs en mètres, lignes depuis le nord."""
    import numpy as np
    g = np.array(s["z"], dtype=np.int64).reshape(s["h"], s["l"])
    g[:, 0] = np.cumsum(g[:, 0])
    return s["zero_m"] + np.cumsum(g, axis=1) * s["pas_m"]


def _maison_surelevee(x, y):
    """Maison de 14 x 10 m : moitié ouest à 6 m, moitié est à 4 m, sol autour."""
    if abs(x) > 7 or abs(y) > 5:
        return 0.0
    return 6.0 if x < 0 else 4.0


def test_la_surface_garde_la_marche_d_une_surelevation():
    """C'est ce que le toit à deux pans perdait : une marche de 2 m au milieu."""
    import numpy as np
    from vue3d.toits import surface_toit
    h, xs, ys = _grille_surface(_maison_surelevee)
    s = surface_toit(box(-7, -5, 7, 5), h, xs, ys, gouttiere=4.0)
    z = _decoder_surface(s)
    X, Y = np.meshgrid(xs[s["i0"]:s["i0"] + s["l"]], ys[s["j0"]:s["j0"] + s["h"]])
    dedans = (np.abs(X) < 6.5) & (np.abs(Y) < 4.5)
    assert np.allclose(z[dedans & (X < -0.5)], 6.0, atol=0.05)
    assert np.allclose(z[dedans & (X > 0.5)], 4.0, atol=0.05)


def test_la_surface_ne_tombe_pas_au_sol_sur_le_contour():
    """Les cellules du bord mêlent toit et sol : elles reprennent leurs voisines."""
    import numpy as np
    from vue3d.toits import surface_toit
    h, xs, ys = _grille_surface(lambda x, y: 5.0 if abs(x) <= 7 and abs(y) <= 5 else 0.0)
    # Un bord qui lit le sol, comme au LiDAR : la première rangée intérieure à 1 m.
    bord = (np.abs(xs) > 6.6) & (np.abs(xs) <= 7)
    h[np.ix_(np.abs(ys) <= 5, bord)] = 1.0
    s = surface_toit(box(-7, -5, 7, 5), h, xs, ys, gouttiere=5.0)
    z = _decoder_surface(s)
    X, Y = np.meshgrid(xs[s["i0"]:s["i0"] + s["l"]], ys[s["j0"]:s["j0"] + s["h"]])
    assert np.allclose(z[(np.abs(X) <= 7) & (np.abs(Y) <= 5)], 5.0, atol=0.05)


def test_la_surface_ecarte_le_feuillage_et_redresse_la_pente():
    """Une cellule verte ne soulève pas le toit ; sur un terrain qui monte de
    1 m vers l'est, le toit plat (5 m au-dessus du sol local) reste horizontal
    une fois rapporté au point le plus bas du contour."""
    import numpy as np
    from vue3d.toits import surface_toit
    sol = lambda x, y: 0.1 * (x + 7)                       # 0 à l'ouest, 1,4 m à l'est
    h, xs, ys = _grille_surface(lambda x, y: (5.0 - sol(x, y)) if abs(x) <= 7 and abs(y) <= 5 else 0.0)
    verdure = np.full(h.shape, -20, dtype=np.int8)
    j, i = len(ys) // 2, len(xs) // 2
    h[j, i] = 12.0                                         # une branche au-dessus
    verdure[j, i] = 40
    X, Y = np.meshgrid(xs, ys)
    s = surface_toit(box(-7, -5, 7, 5), h, xs, ys, gouttiere=4.0, verdure=verdure,
                     sol=sol(X, Y), sol_bas=0.0)
    z = _decoder_surface(s)
    X, Y = np.meshgrid(xs[s["i0"]:s["i0"] + s["l"]], ys[s["j0"]:s["j0"] + s["h"]])
    assert np.allclose(z[(np.abs(X) <= 6) & (np.abs(Y) <= 4)], 5.0, atol=0.1)


def test_trop_peu_de_cellules_lisibles_pas_de_surface():
    from vue3d.toits import surface_toit
    h, xs, ys = _grille_surface(lambda x, y: 5.0 if abs(x) <= 1.5 and abs(y) <= 1.5 else 0.0)
    assert surface_toit(box(-1.5, -1.5, 1.5, 1.5), h, xs, ys, gouttiere=5.0) is None


def test_encodage_des_ecarts_restitue_les_cellules_utiles():
    import numpy as np
    from vue3d.toits import _encoder_ecarts
    q = np.array([[5, 6, 7, 9], [4, 4, 8, 8], [3, 2, 1, 0]])
    utile = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool)
    z = _decoder_surface({"z": _encoder_ecarts(q, utile), "h": 3, "l": 4,
                          "zero_m": 0.0, "pas_m": 1.0})
    assert (z[utile] == q[utile]).all()


def test_une_emprise_qui_deborde_de_la_grille_n_a_pas_de_surface():
    """La grille s'arrête au bord de la scène : le toit n'en couvrirait qu'une partie."""
    from vue3d.toits import surface_toit
    h, xs, ys = _grille_surface(lambda x, y: 5.0)
    assert surface_toit(box(5, -5, 15, 5), h, xs, ys, gouttiere=5.0) is None


def _profil_et_cellules(fonction, emprise):
    """Profil LiDAR et cellules lisibles d'une grille synthétique."""
    import numpy as np
    from vue3d.toits import cellules_du_toit
    h, xs, ys = _grille_surface(fonction)
    X, Y = np.meshgrid(xs, ys)
    cellules = [(x, y, v) for x, y, v in zip(X.ravel(), Y.ravel(), h.ravel()) if v > 0]
    profil = profil_toit(cellules, emprise)
    return profil, cellules_du_toit(emprise, h, xs, ys, profil["gouttiere"])


def test_un_toit_a_deux_pans_reste_resume():
    """Le toit résumé le décrit déjà : la surface n'apporterait que du grain."""
    from vue3d.toits import ecart_au_resume, SURFACE_ECART_RESUME_M
    emprise = box(-8, -5, 8, 5)
    profil, cel = _profil_et_cellules(
        lambda x, y: 7.0 - 3.0 * abs(y) / 5 if abs(x) <= 8 and abs(y) <= 5 else 0.0, emprise)
    assert profil["axe_deg"] is not None
    assert ecart_au_resume(profil, emprise, cel) < SURFACE_ECART_RESUME_M


def test_une_surelevation_recoit_la_surface():
    """Une moitié à 7 m, l'autre à 3 m : deux pans ne savent pas la dessiner."""
    from vue3d.toits import ecart_au_resume, SURFACE_ECART_RESUME_M
    emprise = box(-7, -5, 7, 5)
    profil, cel = _profil_et_cellules(
        lambda x, y: (7.0 if x < 0 else 3.0) if abs(x) <= 7 and abs(y) <= 5 else 0.0, emprise)
    assert ecart_au_resume(profil, emprise, cel) > SURFACE_ECART_RESUME_M


def test_le_toit_resume_suit_l_axe_mesure():
    """Deux pans, faîtage le long de l'axe : au faîtage le faîtage, au bord la gouttière."""
    import numpy as np
    from vue3d.toits import hauteur_resumee
    profil = {"gouttiere": 4.0, "faitage": 7.0, "axe_deg": 0.0}
    X, Y = np.array([[0.0, 0.0, 7.9]]), np.array([[0.0, 4.99, 0.0]])
    z = hauteur_resumee(profil, box(-8, -5, 8, 5), X, Y)
    assert np.allclose(z, [[7.0, 4.0, 7.0]], atol=0.01)


def test_toits_pour_emprise_choisit_bâtiment_par_bâtiment():
    """Deux bâtiments sur une même grille : chacun reçoit son profil, et seule
    la maison surélevée reçoit une forme mesurée — ses deux niveaux, en pans
    (la boucle ne mélange pas les deux)."""
    import math
    import numpy as np
    from vue3d.toits import toits_pour_emprise
    lat0, lon0, pas = 45.0, 5.0, 0.5
    m_lon = 111320 * math.cos(math.radians(lat0))
    n = 120                                              # 60 m de côté
    xs = (np.arange(n) + 0.5) * pas - n * pas / 2
    X, Y = np.meshgrid(xs, -xs)
    h = np.zeros((n, n))
    simple = (np.abs(X + 14) <= 8) & (np.abs(Y) <= 5)
    h[simple] = 7.0 - 3.0 * np.abs(Y[simple]) / 5
    marche = (np.abs(X - 14) <= 7) & (np.abs(Y) <= 5)
    h[marche] = np.where(X[marche] < 14, 7.0, 3.0)
    demi = n * pas / 2
    bbox = (lon0 - demi / m_lon, lat0 - demi / 111320, lon0 + demi / m_lon, lat0 + demi / 111320)
    grille = {"bbox": list(bbox), "width": n, "height": n, "couvert": True,
              "source": "lidar_hd", "values": h.ravel().tolist()}
    def carre(cx, lx, ly, cle):
        c = [(lon0 + (cx + dx) / m_lon, lat0 + dy / 111320) for dx, dy in
             ((-lx, -ly), (lx, -ly), (lx, ly), (-lx, ly), (-lx, -ly))]
        return {"type": "Feature", "properties": {"cleabs": cle},
                "geometry": {"type": "Polygon", "coordinates": [c]}}
    bats = {"features": [carre(-14, 8, 5, "SIMPLE"), carre(14, 7, 5, "MARCHE")]}
    r = toits_pour_emprise(*bbox, bats, grille, None)
    assert "surface" not in r["toits"]["SIMPLE"] and "pans" not in r["toits"]["SIMPLE"]
    assert r["toits"]["MARCHE"]["pans"]["n_pans"] == 2
    assert "surface" not in r["toits"]["MARCHE"]


def _emprise_cabane(h_canopee, h_cabane):
    """Grille de 60 m avec une cabane de 2,4 × 3,0 m au centre.

    L'emprise érodée de 0,5 m (le repli d'`eroder`, buffer(-1.5) étant vide)
    fait 1,4 × 2,0 m : au plus une douzaine de cellules à 0,5 m, toujours sous
    TOITS_MIN_CELLULES — le profil est illisible quel que soit le calage.
    Rend (bbox, bats, grille, canopee) ; `canopee` est le masque |X|,|Y| ≤ 8 m.
    """
    lat0, lon0, pas = 45.0, 5.0, 0.5
    m_lon = 111320 * math.cos(math.radians(lat0))
    n = 120
    xs = (np.arange(n) + 0.5) * pas - n * pas / 2
    X, Y = np.meshgrid(xs, -xs)
    canopee = (np.abs(X) <= 8) & (np.abs(Y) <= 8)
    h = np.zeros((n, n))
    h[canopee] = h_canopee
    cabane = (np.abs(X) <= 1.2) & (np.abs(Y) <= 1.5)
    h[cabane] = np.maximum(h[cabane], h_cabane)
    demi = n * pas / 2
    bbox = (lon0 - demi / m_lon, lat0 - demi / 111320, lon0 + demi / m_lon, lat0 + demi / 111320)
    grille = {"bbox": list(bbox), "width": n, "height": n, "couvert": True,
              "source": "lidar_hd", "values": h.ravel().tolist()}
    c = [(lon0 + dx / m_lon, lat0 + dy / 111320) for dx, dy in
         ((-1.2, -1.5), (1.2, -1.5), (1.2, 1.5), (-1.2, 1.5), (-1.2, -1.5))]
    bats = {"features": [{"type": "Feature", "properties": {"cleabs": "CABANE"},
                          "geometry": {"type": "Polygon", "coordinates": [c]}}]}
    return bbox, bats, grille, canopee


def test_cabane_sous_les_arbres_publie_hauteur_inconnue():
    """Cabane illisible sous une canopée : le profil minimal doit dire
    « hauteur inconnue », pas disparaître — absent de `toits`, la vue croyait
    les altitudes BD TOPO, contaminées par les mêmes arbres (8,1 m de murs
    annoncés pour une cabane d'environ 3 m)."""
    from vue3d.toits import toits_pour_emprise
    from vue3d.ortho import EXG_SEUIL
    bbox, bats, grille, canopee = _emprise_cabane(h_canopee=12.0, h_cabane=12.0)
    exg = np.full(canopee.shape, -15, dtype=np.int8)
    exg[canopee] = EXG_SEUIL + 10   # l'orthophoto voit la canopée verte
    t = toits_pour_emprise(*bbox, bats, grille, exg)["toits"]["CABANE"]
    assert t["fiable"] is False
    assert t["hauteur_inconnue"] is True
    assert t["sous_couvert"] is True
    assert t["part_verte"] >= 0.9
    assert "gouttiere" not in t


def test_cabane_hors_canopee_garde_le_repli_bd_topo():
    """Même cabane illisible mais à découvert : rien n'accuse la BD TOPO, le
    profil minimal ne doit pas la déclarer inconnue — la vue garde son repli."""
    from vue3d.toits import toits_pour_emprise
    bbox, bats, grille, canopee = _emprise_cabane(h_canopee=0.0, h_cabane=2.5)
    exg = np.full(canopee.shape, -15, dtype=np.int8)
    t = toits_pour_emprise(*bbox, bats, grille, exg)["toits"]["CABANE"]
    assert t["fiable"] is False
    assert t["hauteur_inconnue"] is False
    assert t["sous_couvert"] is False
    assert t["part_verte"] == 0.0
    assert "gouttiere" not in t


def test_cabane_sans_orthophoto_ne_declare_rien():
    """Sans grille ExG (exg=None), pas de second avis : part_verte reste None
    et le profil minimal n'affirme rien — même repli qu'avant le correctif."""
    from vue3d.toits import toits_pour_emprise
    bbox, bats, grille, _ = _emprise_cabane(h_canopee=12.0, h_cabane=12.0)
    t = toits_pour_emprise(*bbox, bats, grille, None)["toits"]["CABANE"]
    assert t["fiable"] is False
    assert t["hauteur_inconnue"] is False
    assert t["part_verte"] is None


# --- Bâtiments coupés par le bord de la scène -------------------------------

def _scene_60m(fonction):
    """Grille de 60 m autour de (45°, 5°) : (bbox, grille, rectangle en degrés)."""
    lat0, lon0, pas, n = 45.0, 5.0, 0.5, 120
    m_lon = 111320 * math.cos(math.radians(lat0))
    xs = (np.arange(n) + 0.5) * pas - n * pas / 2
    X, Y = np.meshgrid(xs, -xs)
    demi = n * pas / 2
    bbox = (lon0 - demi / m_lon, lat0 - demi / 111320, lon0 + demi / m_lon, lat0 + demi / 111320)
    grille = {"bbox": list(bbox), "width": n, "height": n, "couvert": True,
              "source": "lidar_hd", "values": fonction(X, Y).ravel().tolist()}

    def rectangle(cle, x0, y0, x1, y1):
        c = [(lon0 + x / m_lon, lat0 + y / 111320) for x, y in
             ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
        return {"type": "Feature", "properties": {"cleabs": cle},
                "geometry": {"type": "Polygon", "coordinates": [c]}}
    return bbox, grille, rectangle


def test_un_batiment_qui_deborde_recoit_sa_forme_mesuree_une_fois_coupe():
    """Deux niveaux, 7 m au nord et 3 m au sud, sur un bâtiment qui sort de la
    grille par l'est. Entier, la grille ne l'encadre pas : il garde son toit
    résumé — le château de Versailles, avant. Coupé sur l'emprise, son morceau
    est un bâtiment ordinaire."""
    from vue3d.batiments import decouper_batiments
    from vue3d.toits import toits_pour_emprise
    bbox, grille, rectangle = _scene_60m(
        lambda X, Y: np.where((X >= 0) & (np.abs(Y) <= 8), np.where(Y > 0, 7.0, 3.0), 0.0))
    bats = {"features": [rectangle("AILE", 0, -8, 50, 8)]}
    entier = toits_pour_emprise(*bbox, bats, grille, None)["toits"]["AILE"]
    assert "pans" not in entier and "surface" not in entier
    coupe = toits_pour_emprise(*bbox, decouper_batiments(bats, *bbox), grille, None)["toits"]["AILE"]
    assert coupe["pans"]["n_pans"] == 2


def test_la_pente_d_un_toit_coupe_se_juge_a_la_largeur_du_batiment_entier():
    """Un toit à 49° de 30 m de large, faîtage nord-sud, dont le bord de la
    scène ne garde que 14 m d'un versant : trop raide pour ce morceau seul,
    plausible pour le bâtiment."""
    from vue3d.batiments import decouper_batiments
    from vue3d.toits import toits_pour_emprise
    bbox, grille, rectangle = _scene_60m(
        lambda X, Y: np.where((X >= 14.75) & (np.abs(Y) <= 15), 3.0 + 1.15 * (X - 14.75), 0.0))
    coupes = decouper_batiments({"features": [rectangle("TOIT", 14.75, -15, 44.75, 15)]}, *bbox)
    (morceau,) = coupes["features"]
    assert morceau["properties"]["coupe"]["largeur_m"] == 30.0
    assert toits_pour_emprise(*bbox, coupes, grille, None)["toits"]["TOIT"]["fiable"]
    sans = {"features": [{**morceau, "properties": {"cleabs": "TOIT"}}]}
    assert not toits_pour_emprise(*bbox, sans, grille, None)["toits"]["TOIT"]["fiable"]


def _donjon(X, Y):
    """20 × 12 m : une terrasse à 6 m, et sur sa moitié est un corps à 16 m.
    Trop de dénivelé pour un toit (10 m sur 12 de large) : le profil est
    rejeté, puis relu dans son niveau bas."""
    return np.where((np.abs(X) <= 10) & (np.abs(Y) <= 6), np.where(X > 0, 16.0, 6.0), 0.0)


def test_deux_niveaux_sans_arbre_recoivent_leur_forme_mesuree():
    """Le château de Chambord : donjon et tours au-dessus des terrasses, 33 %
    de l'emprise, et pas un arbre. Lu comme « un toit sous un arbre », il
    gardait le résumé de ses terrasses ; l'orthophoto, qui n'y voit pas de
    vert, dit que le niveau haut est le bâtiment."""
    from vue3d.toits import toits_pour_emprise
    bbox, grille, rectangle = _scene_60m(_donjon)
    bats = {"features": [rectangle("DONJON", -10, -6, 10, 6)]}
    gris = np.full((120, 120), -20, dtype=np.int8)
    t = toits_pour_emprise(*bbox, bats, grille, gris)["toits"]["DONJON"]
    assert t["deux_niveaux"] is True and t["mode_bas"] is False and t["fiable"] is True
    # Le résumé reste celui du niveau bas ; la forme mesurée porte les deux.
    assert t["faitage"] <= 6.5 and 0.4 <= t["part_haute"] <= 0.6
    assert "pans" in t or "surface" in t
    assert t["ecart_resume"] > 4


def test_deux_niveaux_sous_un_arbre_restent_lus_dans_le_mode_bas():
    """La même grille, mais l'orthophoto voit vert sur le niveau haut : c'est
    un houppier, et la surface décrirait le feuillage."""
    from vue3d.toits import toits_pour_emprise
    bbox, grille, rectangle = _scene_60m(_donjon)
    bats = {"features": [rectangle("REMISE", -10, -6, 10, 6)]}
    xs = (np.arange(120) + 0.5) * 0.5 - 30
    X, Y = np.meshgrid(xs, -xs)
    # Un quart de l'emprise est vert : sous le tiers qui dit « sous couvert »,
    # au-dessus du dixième sous lequel les arbres n'y sont pour rien.
    vert = np.where((X > 5) & (np.abs(Y) <= 6), 20, -20).astype(np.int8)
    t = toits_pour_emprise(*bbox, bats, grille, vert)["toits"]["REMISE"]
    assert t["mode_bas"] is True and not t.get("deux_niveaux")
    assert "pans" not in t and "surface" not in t
    # Sans orthophoto, rien ne dit que ce n'est pas un arbre : mode bas aussi.
    t = toits_pour_emprise(*bbox, bats, grille, None)["toits"]["REMISE"]
    assert t["mode_bas"] is True and "surface" not in t and "pans" not in t


def test_les_cellules_d_une_boite_sont_celles_de_la_grille_dans_le_meme_ordre():
    """Le raccourci par bâtiment ne doit rien changer à ce que lit profil_toit."""
    from shapely.geometry import Polygon
    from vue3d.toits import _cellules_de_la_boite, _cellules_locales, _verdure_locale
    rng = np.random.default_rng(7)
    H = np.where(rng.random((40, 60)) < 0.7, rng.random((40, 60)) * 9, 0.0)
    grille = {"bbox": [5.2, 43.9, 5.2004, 43.9002], "width": 60, "height": 40,
              "values": H.ravel().tolist()}
    lon0, lat0 = 5.2002, 43.9001
    toutes = _cellules_locales(grille, lon0, lat0)
    xs, ys = _verdure_locale(H, grille["bbox"], lon0, lat0)[1:]
    # Dans la grille, à cheval sur son bord, et hors d'elle.
    for boite in ((-8.0, -6.0, 3.0, 4.5), (5.0, -40.0, 90.0, 2.0), (200.0, 200.0, 210.0, 210.0)):
        minx, miny, maxx, maxy = boite
        attendu = [c for c in toutes if minx <= c[0] <= maxx and miny <= c[1] <= maxy]
        assert _cellules_de_la_boite(grille["values"], xs, ys, boite) == attendu
    # Emprise vide : ses bornes ne sont pas des nombres.
    assert _cellules_de_la_boite(grille["values"], xs, ys, Polygon().bounds) == []
