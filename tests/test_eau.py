"""Eau de surface (vue3d/eau.py) : filtrage et découpage, sans réseau."""
from vue3d.eau import LARGEUR_PAR_DEFAUT_M, eau_pour_emprise


def _f(geometrie, **props):
    return {"type": "Feature", "properties": props, "geometry": geometrie}


def _carre(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _ligne(*pts):
    return {"type": "LineString", "coordinates": [list(p) for p in pts]}


EMPRISE = (0.0, 0.0, 1.0, 1.0)


def test_un_lac_plus_grand_que_la_scene_est_decoupe_sur_l_emprise():
    """Le WFS rend le lac entier : seule la part dans la scène y entre."""
    eau = eau_pour_emprise(*EMPRISE, {"features": [_f(_carre(-5, -5, 0.5, 5), nature="Lac")]}, None)
    (s,) = eau["surfaces"]
    xs = [x for x, _ in s["geometrie"]["coordinates"][0]]
    assert min(xs) == 0.0 and max(xs) == 0.5 and s["nature"] == "Lac"


def test_l_axe_fictif_d_une_riviere_a_surface_n_est_pas_dessine():
    cours = {"features": [
        _f(_ligne((0.1, 0.5), (0.9, 0.5)), fictif=True, classe_de_largeur="Plus de 50 m"),
        _f(_ligne((0.1, 0.2), (0.9, 0.2)), fictif=False, classe_de_largeur="Entre 0 et 5 m"),
    ]}
    eau = eau_pour_emprise(*EMPRISE, None, cours)
    assert [c["largeur_m"] for c in eau["cours"]] == [2.5]


def test_un_cours_d_eau_souterrain_n_est_pas_dessine():
    cours = {"features": [_f(_ligne((0.1, 0.2), (0.9, 0.2)), fictif=False,
                             position_par_rapport_au_sol="-1")]}
    assert eau_pour_emprise(*EMPRISE, None, cours)["cours"] == []


def test_une_classe_de_largeur_inconnue_prend_la_plus_etroite():
    cours = {"features": [_f(_ligne((0.1, 0.2), (0.9, 0.2)), fictif=False,
                             classe_de_largeur="En attente de mise à jour")]}
    assert eau_pour_emprise(*EMPRISE, None, cours)["cours"][0]["largeur_m"] == LARGEUR_PAR_DEFAUT_M


def test_un_simple_contact_avec_l_emprise_ne_laisse_rien():
    """Une ligne qui touche le bord en un point, un polygone qui le longe."""
    eau = eau_pour_emprise(*EMPRISE, {"features": [_f(_carre(1, 0, 2, 1))]},
                           {"features": [_f(_ligne((1, 0.5), (2, 0.5)), fictif=False)]})
    assert eau == {"surfaces": [], "cours": []}


def test_le_masque_d_eau_est_en_retrait_de_la_rive():
    """Le feuillage des berges surplombe l'eau : les trois premiers mètres
    restent au sursol, la nappe au-delà en sort."""
    from shapely.geometry import Point, shape
    from vue3d.eau import MASQUE_EAU_RETRAIT_M, masque_eau
    metre = 1 / 111320                       # en degrés de latitude
    riviere = _f(_carre(0, 0, 0.01, 40 * metre), nature="Ecoulement naturel", persistance="Permanent")
    (m,) = masque_eau({"features": [riviere]}, 0, 40 * metre)["features"]
    g = shape(m["geometry"])
    assert not g.contains(Point(0.005, (MASQUE_EAU_RETRAIT_M - 1) * metre))
    assert g.contains(Point(0.005, (MASQUE_EAU_RETRAIT_M + 1) * metre))
    assert g.contains(Point(0.005, 20 * metre))


def test_le_masque_d_eau_epargne_ruisseaux_etroits_mares_a_sec_et_canaux_couverts():
    from vue3d.eau import masque_eau
    metre = 1 / 111320
    etroit = _f(_carre(0, 0, 0.01, 5 * metre), persistance="Permanent")       # 5 m de large
    a_sec = _f(_carre(0, 0, 0.01, 40 * metre), persistance="Intermittent")
    couvert = _f(_carre(0, 0, 0.01, 40 * metre), persistance="Permanent",
                 position_par_rapport_au_sol="-1")
    assert masque_eau({"features": [etroit, a_sec, couvert]}, 0, 40 * metre)["features"] == []
    assert masque_eau(None, 0, 1)["features"] == []
