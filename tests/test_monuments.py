"""Monuments OSM (vue3d/monuments.py) : parsing et remplacement, sans réseau."""
from vue3d.monuments import (HAUTEUR_PAR_DEFAUT_M, METRES_PAR_NIVEAU,
                             TOIT_PART_COTE, monuments_pour_emprise)

# Un degré de longitude vaut ~73,5 km à 48,64° : les carrés ci-dessous font
# quelques dizaines de mètres, comme les parties réelles.
EMPRISE = (-1.5129, 48.6344, -1.5097, 48.6376)


def _way(tags, pts):
    return {"type": "way", "id": 1,
            "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in pts]}


def _carre(x0, y0, cote_deg):
    return [(x0, y0), (x0 + cote_deg, y0), (x0 + cote_deg, y0 + cote_deg),
            (x0, y0 + cote_deg), (x0, y0)]


def _bat(cleabs, pts, **props):
    return {"type": "Feature",
            "properties": {"cleabs": cleabs, **props},
            "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in pts]]}}


def test_une_partie_complete_est_lue_telle_quelle():
    brut = {"elements": [_way(
        {"building:part": "yes", "height": "20.5", "min_height": "3",
         "roof:shape": "gabled", "roof:height": "4", "name": "La Merveille"},
        _carre(-1.5115, 48.636, 0.0002))]}
    m = monuments_pour_emprise(*EMPRISE, brut, None)
    (p,) = m["parties"]
    assert (p["h"], p["h0"], p["nom"], p["estime"]) == (20.5, 3.0, "La Merveille", False)
    assert p["toit"] == {"forme": "deux_pans", "h": 4.0, "orientation": None}
    assert p["contour"][0] == p["contour"][-1]


def test_sans_hauteur_le_repli_est_la_hauteur_bd_topo_du_batiment_contenant():
    contour = _carre(-1.5115, 48.636, 0.0002)
    brut = {"elements": [_way({"building:part": "yes", "building:levels": "2"}, contour)]}
    bats = {"features": [_bat("BAT1", _carre(-1.5116, 48.6359, 0.0004), hauteur=15.8)]}
    (p,) = monuments_pour_emprise(*EMPRISE, brut, bats)["parties"]
    # La mesure BD TOPO prime sur la convention niveaux × 3.
    assert (p["h"], p["bat"], p["estime"]) == (15.8, "BAT1", True)


def test_sans_hauteur_ni_batiment_les_niveaux_puis_le_defaut_prennent_le_relais():
    a, b = _carre(-1.5115, 48.636, 0.0002), _carre(-1.5110, 48.636, 0.0002)
    brut = {"elements": [_way({"building:part": "yes", "building:levels": "3"}, a),
                         _way({"building:part": "yes"}, b)]}
    p1, p2 = monuments_pour_emprise(*EMPRISE, brut, None)["parties"]
    assert p1["h"] == 3 * METRES_PAR_NIVEAU
    assert p2["h"] == HAUTEUR_PAR_DEFAUT_M


def test_un_toit_en_pente_sans_roof_height_prend_la_part_mesuree_du_petit_cote():
    # Carré de 0,0002° de côté : ~14,7 m en longitude à cette latitude.
    brut = {"elements": [_way({"building:part": "yes", "height": "30",
                               "roof:shape": "pyramidal"},
                              _carre(-1.5115, 48.636, 0.0002))]}
    (p,) = monuments_pour_emprise(*EMPRISE, brut, None)["parties"]
    assert p["toit"]["forme"] == "pyramide"
    cote_attendu = 0.0002 * 73500 * TOIT_PART_COTE   # petit côté ~14,7 m
    assert abs(p["toit"]["h"] - cote_attendu) < 0.5


def test_une_forme_inconnue_reste_un_sommet_plat():
    brut = {"elements": [_way({"building:part": "yes", "height": "10",
                               "roof:shape": "dome"},
                              _carre(-1.5115, 48.636, 0.0002))]}
    (p,) = monuments_pour_emprise(*EMPRISE, brut, None)["parties"]
    assert p["toit"] is None


def test_les_outer_d_une_relation_sont_des_parties():
    brut = {"elements": [{
        "type": "relation", "id": 2, "tags": {"building:part": "yes", "height": "8"},
        "members": [
            {"type": "way", "role": "outer",
             "geometry": [{"lon": x, "lat": y} for x, y in _carre(-1.5115, 48.636, 0.0002)]},
            {"type": "way", "role": "inner",
             "geometry": [{"lon": x, "lat": y} for x, y in _carre(-1.51145, 48.63605, 0.00005)]},
        ]}]}
    m = monuments_pour_emprise(*EMPRISE, brut, None)
    assert len(m["parties"]) == 1 and m["parties"][0]["h"] == 8.0


def test_un_way_ouvert_ou_vide_est_ignore_et_sans_partie_la_couche_est_nulle():
    ouvert = _way({"building:part": "yes", "height": "5"},
                  [(-1.5115, 48.636), (-1.5113, 48.636), (-1.5113, 48.6362)])
    assert monuments_pour_emprise(*EMPRISE, {"elements": [ouvert]}, None) is None
    assert monuments_pour_emprise(*EMPRISE, {"elements": []}, None) is None


def test_une_enveloppe_qui_avalerait_des_parties_plus_basses_repart_au_repli():
    """L'église abbatiale du Mont : 78,5 m — la flèche — sur tout le vaisseau,
    qui recouvre des parties mesurées plus basses. Sa hauteur est une
    enveloppe : elle repart sur la hauteur BD TOPO. La partie haute et étroite
    qu'elle contient (la flèche), elle, garde sa mesure."""
    vaisseau = _carre(-1.5115, 48.636, 0.0004)
    clocheton = _carre(-1.5114, 48.6361, 0.0001)     # dedans, plus bas
    fleche = _carre(-1.51135, 48.63615, 0.00003)     # dedans, plus haut
    brut = {"elements": [
        _way({"building:part": "yes", "height": "78.5"}, vaisseau),
        _way({"building:part": "yes", "height": "26"}, clocheton),
        _way({"building:part": "yes", "height": "79"}, fleche),
    ]}
    bats = {"features": [_bat("EGLISE", _carre(-1.51155, 48.63595, 0.0005), hauteur=20.3)]}
    parts = monuments_pour_emprise(*EMPRISE, brut, bats)["parties"]
    assert [(p["h"], p["estime"]) for p in parts] == [
        (20.3, True), (26.0, False), (79.0, False)]


def test_le_batiment_couvert_est_remplace_le_voisin_non():
    """La règle mesurée au Mont : couvert aux 2/3 par l'union dilatée de 5 m.

    Le bâtiment sous la partie est remplacé même si leurs contours diffèrent
    d'un ou deux mètres ; la maison à 30 m n'est pas touchée.
    """
    partie = _carre(-1.5115, 48.636, 0.0002)
    sous = _carre(-1.51152, 48.63598, 0.0002)       # décalé de ~2 m
    loin = _carre(-1.5110, 48.636, 0.0002)          # à ~35 m
    brut = {"elements": [_way({"building:part": "yes", "height": "12"}, partie)]}
    bats = {"features": [_bat("SOUS", sous, hauteur=10), _bat("LOIN", loin, hauteur=10)]}
    m = monuments_pour_emprise(*EMPRISE, brut, bats)
    assert m["remplaces"] == ["SOUS"]
