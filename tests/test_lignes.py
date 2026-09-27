"""Lignes à haute tension (vue3d/lignes.py) : portées gardées et hauteurs, sans réseau."""
from vue3d.lignes import HAUTEUR_PAR_TENSION_M, lignes_pour_emprise

EMPRISE = (0.0, 0.0, 0.01, 0.01)          # ~1,1 km de côté à l'équateur


def _ligne(coords, tension="225 kV"):
    return {"type": "Feature", "properties": {"voltage": tension, "gestionnaire": "RTE"},
            "geometry": {"type": "LineString", "coordinates": [list(c) for c in coords]}}


def _pylone(lon, lat, h):
    return {"type": "Feature", "properties": {"hauteur": h},
            "geometry": {"type": "Point", "coordinates": [lon, lat]}}


def test_une_ligne_qui_ne_fait_qu_encadrer_l_emprise_est_ecartee():
    """Le WFS rend une ligne dont la boîte touche l'emprise ; elle, non."""
    en_l = _ligne([(-0.01, 0.02), (-0.01, -0.01), (0.02, -0.01)])
    assert lignes_pour_emprise(*EMPRISE, {"features": [en_l]}, None) == []


def test_la_portee_qui_sort_garde_son_support_au_dehors():
    """Le câble doit aller jusqu'au bord : le pylône extérieur est gardé, pas le suivant."""
    ligne = _ligne([(0.005, 0.005), (0.015, 0.005), (0.025, 0.005)])
    (l,) = lignes_pour_emprise(*EMPRISE, {"features": [ligne]}, None)
    assert [s[:2] for s in l["supports"]] == [[0.005, 0.005], [0.015, 0.005]]


def test_hauteur_du_pylone_sinon_mediane_de_la_tension():
    ligne = _ligne([(0.002, 0.005), (0.006, 0.005)], "400 kV")
    pylones = {"features": [_pylone(0.00201, 0.005, 61.2)]}     # à ~1 m du premier sommet
    (l,) = lignes_pour_emprise(*EMPRISE, {"features": [ligne]}, pylones)
    assert l["supports"][0][2:] == [61.2, True]
    assert l["supports"][1][2:] == [HAUTEUR_PAR_TENSION_M["400 kV"], False]


def test_deux_passages_dans_l_emprise_font_deux_suites():
    """Une ligne qui entre, sort et rentre : deux suites de portées, pas une."""
    ligne = _ligne([(0.002, 0.002), (0.002, 0.03), (0.008, 0.03), (0.008, 0.002)])
    suites = lignes_pour_emprise(*EMPRISE, {"features": [ligne]}, None)
    assert len(suites) == 2
