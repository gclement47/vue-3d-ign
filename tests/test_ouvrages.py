"""Ouvrages BD TOPO (vue3d/ouvrages.py) : murs, ponts, voies ferrées, terrains — sans réseau."""
import numpy as np
import pytest

from vue3d import relief
from vue3d.ouvrages import (LARGEUR_PAR_VOIE_M, OUVRAGES_VERSION, PONT_LARGEUR_PAR_DEFAUT_M,
                            decouper_ligne, ouvrages_pour_emprise)

# ~100 m de côté vers 45° N ; le relief y est un plan à 100 m.
EMPRISE = (2.0, 45.0, 2.00127, 45.0009)
SOL = 100.0


def _relief(altitudes=None):
    g = np.full((8, 8), SOL) if altitudes is None else np.asarray(altitudes, dtype=float)
    return relief._quantifier(g, np.zeros(g.shape, dtype=bool), *EMPRISE)


def _pt(fx, fy, z=None):
    """Point à la fraction (fx, fy) de l'emprise, d'ouest en est et du sud au nord."""
    o, s, e, n = EMPRISE
    p = [o + (e - o) * fx, s + (n - s) * fy]
    return p if z is None else p + [z]


def _ligne(points, **props):
    return {"type": "Feature", "properties": {"etat_de_l_objet": "En service", **props},
            "geometry": {"type": "LineString", "coordinates": points}}


def _polygone(points, **props):
    return {"type": "Feature", "properties": {"etat_de_l_objet": "En service", **props},
            "geometry": {"type": "MultiPolygon", "coordinates": [[points + [points[0]]]]}}


def _ouvrages(lineaires=(), surfaciques=(), voies=(), terrains=(), **kw):
    brut = {"lineaires": {"features": list(lineaires)}, "surfaciques": {"features": list(surfaciques)},
            "voies": {"features": list(voies)}, "terrains": {"features": list(terrains)}}
    kw.setdefault("relief", _relief())
    return ouvrages_pour_emprise(*EMPRISE, brut, **kw)


def test_l_altitude_d_un_point_de_coupe_est_interpolee():
    """La ligne sort à mi-segment : le sommet de coupe prend l'altitude du milieu."""
    (morceau,) = decouper_ligne([_pt(0.5, 0.5, 110), _pt(1.5, 0.5, 130)], *EMPRISE)
    assert morceau[0][2] == 110 and morceau[-1][2] == pytest.approx(120)
    assert morceau[-1][0] == pytest.approx(EMPRISE[2])


def test_une_ligne_qui_sort_et_rentre_fait_deux_morceaux():
    ligne = [_pt(0.2, 0.5, 1), _pt(0.2, 1.5, 1), _pt(0.8, 1.5, 1), _pt(0.8, 0.5, 1)]
    assert len(decouper_ligne(ligne, *EMPRISE)) == 2


def test_une_ligne_dont_seule_la_boite_touche_l_emprise_est_ecartee():
    """Le WFS la rend quand même : constaté au Stade de France et à Feyzin."""
    assert decouper_ligne([_pt(-0.5, 0.5, 1), _pt(0.5, 1.5, 1)], *EMPRISE) == []


def test_la_hauteur_d_un_mur_est_l_altitude_de_ses_sommets_moins_le_relief():
    c = _ouvrages([_ligne([_pt(0.2, 0.5, SOL + 9), _pt(0.8, 0.5, SOL + 11)], nature="Mur",
                          toponyme="Remparts")])
    (mur,) = c["murs"]
    assert mur["h"] == pytest.approx(10, abs=0.2) and mur["nom"] == "Remparts"
    # Les sommets gardent leur altitude : c'est la vue qui pose le pied du mur.
    assert [p[2] for p in mur["ligne"]] == [SOL + 9, SOL + 11]
    assert c["version"] == OUVRAGES_VERSION and c["ponts"] == []


def test_un_mur_bas_un_mur_de_soutenement_et_un_tunnel_ne_sont_pas_dessines():
    bas = _ligne([_pt(0.2, 0.2, SOL + 1.2), _pt(0.8, 0.2, SOL + 1.2)], nature="Mur")
    soutenement = _ligne([_pt(0.2, 0.4, SOL + 6), _pt(0.8, 0.4, SOL + 6)],
                         nature="Mur de soutènement")
    tunnel = _ligne([_pt(0.2, 0.6, -1000), _pt(0.8, 0.6, -1000)], nature="Tunnel")
    sans_z = _ligne([_pt(0.2, 0.7, -1000), _pt(0.8, 0.7, -1000)], nature="Mur")
    projet = dict(_ligne([_pt(0.2, 0.8, SOL + 9), _pt(0.8, 0.8, SOL + 9)], nature="Mur"))
    projet["properties"]["etat_de_l_objet"] = "En projet"
    assert _ouvrages([bas, soutenement, tunnel, sans_z, projet]) is None


def test_un_pont_en_ligne_prend_la_largeur_de_la_route_qu_il_porte():
    """Culées au sol, tablier à 20 m : dessiné, à la largeur de la chaussée hors sol."""
    pont = _ligne([_pt(0.2, 0.5, SOL), _pt(0.5, 0.5, SOL + 20), _pt(0.8, 0.5, SOL)],
                  nature="Pont", nature_detaillee="Aqueduc")
    route = _ligne([_pt(0.1, 0.5), _pt(0.9, 0.5)], position_par_rapport_au_sol="1",
                   largeur_de_chaussee=5)
    (p,) = _ouvrages([pont], routes={"features": [route]})["ponts"]
    assert (p["h"], p["largeur_m"], p["largeur_mesuree"], p["detail"]) == (20, 5, True, "Aqueduc")
    # Sans route portée, la largeur par défaut, et la vue saura que c'est une convention.
    (p,) = _ouvrages([pont])["ponts"]
    assert (p["largeur_m"], p["largeur_mesuree"]) == (PONT_LARGEUR_PAR_DEFAUT_M, False)
    # Une route au sol qui le longe n'est pas celle qu'il porte.
    route["properties"]["position_par_rapport_au_sol"] = "0"
    assert _ouvrages([pont], routes={"features": [route]})["ponts"][0]["largeur_mesuree"] is False


def test_un_pont_que_le_relief_porte_deja_n_est_pas_dessine():
    pont = _ligne([_pt(0.2, 0.5, SOL + 0.5), _pt(0.8, 0.5, SOL - 1.3)], nature="Pont")
    assert _ouvrages([pont]) is None


def test_un_pont_surfacique_est_coupe_au_bord_et_garde_ses_altitudes():
    """Tablier qui déborde à l'est, montant de 10 à 30 m : le bord de coupe,
    à mi-montée, est à 20 m."""
    tablier = _polygone([_pt(0.5, 0.4, SOL + 10), _pt(1.5, 0.4, SOL + 30), _pt(1.5, 0.6, SOL + 30),
                         _pt(0.5, 0.6, SOL + 10)], nature="Pont")
    (p,) = _ouvrages(surfaciques=[tablier])["ponts"]
    assert "ligne" not in p and p["trous"] == []
    assert max(q[0] for q in p["contour"]) == pytest.approx(EMPRISE[2])
    assert sorted({q[2] for q in p["contour"]}) == [SOL + 10, SOL + 20]
    assert p["h"] == pytest.approx(20)
    # Escaliers et dalles sont posés sur un sol que le relief porte déjà.
    escalier = _polygone([_pt(0.1, 0.1, SOL + 5), _pt(0.2, 0.1, SOL + 5), _pt(0.2, 0.2, SOL + 5)],
                         nature="Escalier")
    assert _ouvrages(surfaciques=[escalier]) is None


def test_les_voies_ferrees_au_sol_sur_ouvrage_et_sous_le_sol():
    au_sol = _ligne([_pt(0.1, 0.2, SOL), _pt(0.9, 0.2, SOL)], nature="Voie ferrée principale",
                    position_par_rapport_au_sol="0", nombre_de_voies=2, largeur="Normale")
    viaduc = _ligne([_pt(0.1, 0.4, SOL + 8), _pt(0.9, 0.4, SOL + 8)], nature="Voie ferrée principale",
                    position_par_rapport_au_sol="1", nombre_de_voies=1, largeur="Normale")
    tunnel = _ligne([_pt(0.1, 0.6, -1000), _pt(0.9, 0.6, -1000)], nature="Métro",
                    position_par_rapport_au_sol="-1", nombre_de_voies=2, largeur="Normale")
    metrique = _ligne([_pt(0.1, 0.8, SOL), _pt(0.9, 0.8, SOL)], nature="Funiculaire ou crémaillère",
                      position_par_rapport_au_sol="0", nombre_de_voies=1, largeur="Etroite")
    voies = _ouvrages(voies=[au_sol, viaduc, tunnel, metrique])["voies"]
    assert [(v["au_sol"], v["largeur_m"]) for v in voies] == [
        (True, 2 * LARGEUR_PAR_VOIE_M["Normale"]), (False, LARGEUR_PAR_VOIE_M["Normale"]),
        (True, LARGEUR_PAR_VOIE_M["Etroite"])]
    # Au sol, la vue drape : pas d'altitude. Sur un ouvrage, elle la garde.
    assert len(voies[0]["ligne"][0]) == 2 and voies[1]["ligne"][0][2] == SOL + 8


def test_un_terrain_de_sport_est_decoupe_sur_l_emprise():
    terrain = _polygone([_pt(0.6, 0.2), _pt(1.4, 0.2), _pt(1.4, 0.6), _pt(0.6, 0.6)],
                        nature="Grand terrain de sport", nature_detaillee="Terrain de football")
    (t,) = _ouvrages(terrains=[terrain])["terrains"]
    assert (t["nature"], t["detail"]) == ("Grand terrain de sport", "Terrain de football")
    lons = [p[0] for poly in t["geometrie"]["coordinates"] for anneau in poly for p in anneau]
    assert max(lons) == pytest.approx(EMPRISE[2]) and all(len(p) == 2 for poly in
                                                          t["geometrie"]["coordinates"]
                                                          for anneau in poly for p in anneau)


def test_les_masses_sur_un_mur_ou_sous_un_tablier_sont_expliquees():
    mur = _ligne([_pt(0.1, 0.5, SOL + 10), _pt(0.9, 0.5, SOL + 10)], nature="Mur")
    o, s, e, n = EMPRISE
    metre = (n - s) / 100          # un mètre, en degrés de latitude
    masses = [{"lon": _pt(0.5, 0.5)[0], "lat": _pt(0.5, 0.5)[1] + metre},         # à 1 m du mur
              {"lon": _pt(0.5, 0.5)[0], "lat": _pt(0.5, 0.5)[1] + 6 * metre},     # à 6 m
              {"lon": _pt(0.3, 0.5)[0], "lat": _pt(0.3, 0.5)[1] - 1.5 * metre}]   # à 1,5 m
    assert _ouvrages([mur], masses=masses)["masses_expliquees"] == [0, 2]


def test_sans_relief_ni_mur_ni_pont_ni_voie_sur_ouvrage():
    """Hors couverture RGE ALTI, une altitude n'a rien à quoi se comparer ;
    les voies au sol et les terrains, drapés, restent."""
    mur = _ligne([_pt(0.1, 0.5, SOL + 10), _pt(0.9, 0.5, SOL + 10)], nature="Mur")
    viaduc = _ligne([_pt(0.1, 0.4, SOL + 8), _pt(0.9, 0.4, SOL + 8)], nature="Tramway",
                    position_par_rapport_au_sol="1", nombre_de_voies=1)
    au_sol = _ligne([_pt(0.1, 0.2, SOL), _pt(0.9, 0.2, SOL)], nature="Tramway",
                    position_par_rapport_au_sol="0", nombre_de_voies=1)
    assert _ouvrages([mur], voies=[viaduc], relief=None) is None
    c = _ouvrages([mur], voies=[viaduc, au_sol], relief=None)
    assert c["murs"] == [] and [v["au_sol"] for v in c["voies"]] == [True]


def test_une_emprise_sans_ouvrage_vaut_none():
    assert _ouvrages() is None
    assert ouvrages_pour_emprise(*EMPRISE, None) is None
