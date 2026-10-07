"""Serveur de la vue 3D.

    GET /                          la page ; `?lat=…&lon=…` pour viser un point
                                   et `&zone=…` pour le côté de la zone, en mètres
    GET /api/scene?lat=…&lon=…     la scène, JSON gzippé (construite au besoin)
    GET /api/ortho?lat=…&lon=…     l'orthophoto de la scène, en JPEG
    GET /api/monuments?lat=…&lon=…   la couche des monuments OSM, JSON gzippé
    GET /api/ouvrages?lat=…&lon=…    la couche des ouvrages BD TOPO, JSON gzippé
    GET /api/nuage?lat=…&lon=…       le bâti du nuage LiDAR HD, JSON gzippé
    GET /api/piscines?lat=…&lon=…    les piscines de l'orthophoto, si le service a un détecteur
    GET /api/vehicules?lat=…&lon=…&detecteur=…   les véhicules vus d'un détecteur du service
    GET /api/panneaux?lat=…&lon=…    les panneaux solaires du registre, si le service en a un
    GET /api/avancement?lat=…&lon=…  l'étape de la construction en cours
    GET /api/sante                 contrôle de vie, pour Docker
    POST /api/zone/tuile          construit une tuile 3D et la sauvegarde sur Geovalys
    POST /api/zone/finaliser      enregistre le GeoJSON et le manifeste de la zone

Chaque route d'API accepte `zone=` (150 à 1 000 m, arrondie à 50 m) : sans
elle, l'emprise par défaut d'environ 356 m.

La première demande d'un point construit sa scène : une vingtaine à une
trentaine de secondes, que la page annonce. Les suivantes la lisent sur disque.
"""

import logging
import base64
import copy
import math
import gzip
import os
import json
import re
import requests
import numpy as np

from flask import Flask, jsonify, request, send_from_directory

from .monuments import fetch_monuments
from .nuage import fetch_nuage
from .ouvrages import fetch_ouvrages
from .ortho import fetch_ortho_jpeg
from .panneaux import REGISTRE_LICENCE
from .panneaux import lecteur as lecteur_panneaux
from .relief import fetch_relief_anneau
from .scene import (NOM_MONUMENTS, NOM_NUAGE, NOM_ORTHO, NOM_OUVRAGES, NOM_PANNEAUX, NOM_SCENE,
                    Cache, HorsEmprise, MonumentsIndisponibles, NuageIndisponible,
                    OuvragesIndisponibles,
                    PanneauxIndisponibles, ReconstructionRefusee, SceneIncomplete,
                    VehiculesDesactives, VehiculesIndisponibles,
                    point_normalise, zone_normalisee)
from .scene import construire as construire_scene
from .toits import autoriser_bassin
from .vehicules import MODE_PAR_DEFAUT
from .vehicules import lecteur as lecteur_vehicules

# Noms de couche pour dossier_scene : les fichiers, eux, dépendent du mode et
# du détecteur.
COUCHE_VEHICULES = "vehicules"
COUCHE_PISCINES = "piscines"

# Orthophoto persistante du relief périphérique.
NOM_ORTHO_ANNEAU = "ortho_anneau.jpg"

logging.basicConfig(level=os.environ.get("VUE3D_LOG", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s : %(message)s")

ICI = os.path.dirname(os.path.abspath(__file__))


def creer_app(dossier_cache=None, construire=construire_scene, lire_monuments=fetch_monuments,
              lire_ouvrages=fetch_ouvrages, lire_vehicules=None, lire_panneaux=None,
              lire_nuage=fetch_nuage):
    """`construire`, `lire_monuments`, `lire_ouvrages`, `lire_vehicules`,
    `lire_panneaux` et `lire_nuage` sont injectables pour les tests, qui
    n'appellent ni l'IGN ni Overpass et ne chargent aucun réseau ni registre. `lire_vehicules` :
    de `vehicules.lecteur()` ; None, le service n'a ni véhicules ni piscines.
    `lire_panneaux` : de `panneaux.lecteur()` ; None, pas de panneaux."""
    app = Flask(__name__, static_folder=os.path.join(ICI, "static"), static_url_path="/static")
    # Absolu : send_from_directory résout un chemin relatif depuis le dossier
    # de l'application, pas depuis le répertoire courant — avec
    # VUE3D_CACHE=./cache, l'orthophoto répondait 404.
    cache = Cache(os.path.abspath(dossier_cache or os.environ.get("VUE3D_CACHE", "/tmp/vue3d-cache")),
                  lire_monuments=lire_monuments, lire_ouvrages=lire_ouvrages,
                  lire_vehicules=lire_vehicules, lire_panneaux=lire_panneaux,
                  lire_nuage=lire_nuage)

    def point():
        """(lat, lon, zone) de la requête ; None si l'un d'eux est illisible."""
        try:
            return (float(request.args["lat"]), float(request.args["lon"]),
                    zone_normalisee(request.args.get("zone")))
        except (KeyError, ValueError):
            return None

    MESSAGE_POINT = ("Paramètres lat et lon attendus, en degrés décimaux ; zone, "
                     "facultative, en mètres.")

    def erreur(code, message):
        return jsonify({"erreur": message}), code

    def slug_zone(nom):
        """Nom de dossier sûr pour Geovalys."""
        nom = (nom or "").strip().lower()
        nom = re.sub(r"[^a-z0-9_-]+", "-", nom)
        nom = re.sub(r"-+", "-", nom).strip("-")
        return nom[:80]

    def stockage_configure():
        return bool(
            os.environ.get("VUE3D_STORAGE_URL")
            and os.environ.get("VUE3D_STORAGE_TOKEN")
        )

    def envoyer_octets(zone, scope, nom, contenu, tile=None,
                       type_mime="application/octet-stream"):
        """Envoie un fichier vers Geovalys, en morceaux au-delà de 900 Ko."""
        url = os.environ.get("VUE3D_STORAGE_URL")
        token = os.environ.get("VUE3D_STORAGE_TOKEN")

        if not url or not token:
            raise RuntimeError("Stockage Geovalys non configuré.")

        donnees_base = {
            "token": token,
            "zone": zone,
            "scope": scope,
        }

        if tile:
            donnees_base["tile"] = tile

        if isinstance(contenu, (bytes, bytearray)):
            brut = bytes(contenu)
        else:
            position = contenu.tell()
            contenu.seek(0)
            brut = contenu.read()
            contenu.seek(position)

        taille_morceau = 900 * 1024

        def verifier(r):
            if not r.ok:
                raise RuntimeError(
                    f"Geovalys HTTP {r.status_code}: {r.text[:500]}"
                )

            try:
                retour = r.json()
            except ValueError as exc:
                corps = (r.text or "").strip().replace("\n", " ")
                corps = re.sub(r"\\s+", " ", corps)[:500]
                raise RuntimeError(
                    "Geovalys a renvoyé une réponse non JSON : "
                    + (corps or "réponse vide")
                ) from exc

            if not retour.get("ok"):
                raise RuntimeError(
                    retour.get("error", "Erreur de stockage Geovalys")
                )

            return retour

        if len(brut) <= taille_morceau:
            r = requests.post(
                url,
                data=donnees_base,
                files={"file": (nom, brut, type_mime)},
                timeout=120,
            )
            return verifier(r)

        total = math.ceil(len(brut) / taille_morceau)
        retour = None

        for index in range(total):
            debut = index * taille_morceau
            fin = min(len(brut), debut + taille_morceau)

            donnees = {
                **donnees_base,
                "filename": nom,
                "chunk_index": str(index),
                "chunk_total": str(total),
            }

            r = requests.post(
                url,
                data=donnees,
                files={
                    "file": (
                        f"{nom}.part{index:04d}",
                        brut[debut:fin],
                        type_mime,
                    )
                },
                timeout=120,
            )

            retour = verifier(r)

        return retour


    def assurer_ortho_anneau(dossier):
        """Crée l'orthophoto de l'anneau si elle n'est pas encore dans le cache.

        On essaie d'abord 4 096 px sur le plus grand côté pour une image nettement
        plus précise que le fond WMTS zoom 15 du navigateur. Si le WMS refuse cette
        taille, on retombe automatiquement à 2 048 px.
        """
        chemin = os.path.join(dossier, NOM_ORTHO_ANNEAU)

        if os.path.isfile(chemin):
            return chemin

        scene_path = os.path.join(dossier, NOM_SCENE)

        with gzip.open(scene_path, "rt", encoding="utf-8") as f:
            scene = json.load(f)

        anneau = scene.get("anneau")
        bbox = anneau.get("bbox") if isinstance(anneau, dict) else None

        if not bbox or len(bbox) != 4:
            raise RuntimeError(
                "La scène ne contient pas d'emprise d'anneau exploitable."
            )

        try:
            octets, largeur, hauteur = fetch_ortho_jpeg(
                *bbox,
                max_pixels=4096,
            )
        except Exception as exc:
            app.logger.warning(
                "Orthophoto d'anneau 4096 px refusée, repli 2048 px : %s",
                exc,
            )
            octets, largeur, hauteur = fetch_ortho_jpeg(
                *bbox,
                max_pixels=2048,
            )

        temporaire = chemin + ".tmp"

        with open(temporaire, "wb") as f:
            f.write(octets)

        os.replace(temporaire, chemin)

        app.logger.info(
            "Orthophoto d'anneau prête : %s (%d × %d, %.1f Ko)",
            NOM_ORTHO_ANNEAU,
            largeur,
            hauteur,
            len(octets) / 1024,
        )

        return chemin

                  

    def _decoder_relief(relief):
        if not relief or not relief.get("altitudes"):
            return None

        largeur = int(relief["width"])
        hauteur = int(relief["height"])

        q = np.frombuffer(
            base64.b64decode(relief["altitudes"]),
            dtype="<i2",
        ).reshape(hauteur, largeur)

        valeurs = (
            float(relief["zero_m"])
            + q.astype(np.float64) * float(relief["pas_m"])
        )

        valeurs[q == -32768] = np.nan
        return valeurs


    def _encoder_relief(valeurs, bbox, source="RGE ALTI (IGN)"):
        if valeurs is None or not np.isfinite(valeurs).any():
            return None

        valeurs = np.asarray(valeurs, dtype=np.float64)
        trous = ~np.isfinite(valeurs)

        zero = float(np.floor(np.nanmin(valeurs)))
        pas = 0.1

        q = np.rint((valeurs - zero) / pas)
        q = np.clip(q, -32767, 32767)
        q[trous] = -32768

        hauteur, largeur = valeurs.shape

        return {
            "width": int(largeur),
            "height": int(hauteur),
            "bbox": [float(v) for v in bbox],
            "zero_m": round(zero, 1),
            "pas_m": pas,
            "source": source,
            "precision": "1 m, modèle de terrain",
            "altitudes": base64.b64encode(
                q.astype("<i2").tobytes()
            ).decode("ascii"),
        }


    def _fusionner_reliefs(scenes, bbox_union, max_pixels=1024):
        reliefs = [
            s.get("relief")
            for s in scenes
            if s.get("relief")
            and s["relief"].get("altitudes")
        ]

        if not reliefs:
            return None

        pas_lon = []
        pas_lat = []

        for r in reliefs:
            w, s, e, n = map(float, r["bbox"])

            if int(r["width"]) > 1:
                pas_lon.append(
                    (e - w) / (int(r["width"]) - 1)
                )

            if int(r["height"]) > 1:
                pas_lat.append(
                    (n - s) / (int(r["height"]) - 1)
                )

        if not pas_lon or not pas_lat:
            return reliefs[0]

        west, south, east, north = map(float, bbox_union)

        dx = min(pas_lon)
        dy = min(pas_lat)

        largeur = int(round((east - west) / dx)) + 1
        hauteur = int(round((north - south) / dy)) + 1

        facteur = max(
            1.0,
            largeur / max_pixels,
            hauteur / max_pixels,
        )

        largeur = max(
            2,
            int(round((largeur - 1) / facteur)) + 1,
        )

        hauteur = max(
            2,
            int(round((hauteur - 1) / facteur)) + 1,
        )

        xs = np.linspace(west, east, largeur)
        ys = np.linspace(north, south, hauteur)

        somme = np.zeros(
            (hauteur, largeur),
            dtype=np.float64,
        )

        compte = np.zeros(
            (hauteur, largeur),
            dtype=np.uint16,
        )

        for r in reliefs:
            arr = _decoder_relief(r)

            if arr is None:
                continue

            rw, rs, re, rn = map(float, r["bbox"])
            rh, rl = arr.shape

            ix = np.where(
                (xs >= rw - 1e-12)
                & (xs <= re + 1e-12)
            )[0]

            iy = np.where(
                (ys <= rn + 1e-12)
                & (ys >= rs - 1e-12)
            )[0]

            if not len(ix) or not len(iy):
                continue

            xx = xs[ix][None, :]
            yy = ys[iy][:, None]

            fx = (
                (xx - rw)
                / (re - rw)
                * (rl - 1)
            )

            fy = (
                (rn - yy)
                / (rn - rs)
                * (rh - 1)
            )

            x0 = np.floor(fx).astype(int)
            y0 = np.floor(fy).astype(int)

            x1 = np.minimum(
                x0 + 1,
                rl - 1,
            )

            y1 = np.minimum(
                y0 + 1,
                rh - 1,
            )

            ax = fx - x0
            ay = fy - y0

            a = arr[y0, x0]
            b = arr[y0, x1]
            c = arr[y1, x0]
            d = arr[y1, x1]

            val = (
                (a * (1 - ax) + b * ax) * (1 - ay)
                + (c * (1 - ax) + d * ax) * ay
            )

            valide = np.isfinite(val)

            sous_somme = somme[np.ix_(iy, ix)]
            sous_compte = compte[np.ix_(iy, ix)]

            sous_somme[valide] += val[valide]
            sous_compte[valide] += 1

            somme[np.ix_(iy, ix)] = sous_somme
            compte[np.ix_(iy, ix)] = sous_compte

        valeurs = np.full(
            (hauteur, largeur),
            np.nan,
            dtype=np.float64,
        )

        masque = compte > 0
        valeurs[masque] = somme[masque] / compte[masque]

        return _encoder_relief(
            valeurs,
            bbox_union,
        )


    def _grille_toits_commune(scenes, bbox_union):
        grilles = [
            (s.get("toits") or {}).get("grille")
            for s in scenes
        ]

        grilles = [
            g for g in grilles
            if g and g.get("bbox")
            and g.get("width")
            and g.get("height")
        ]

        if not grilles:
            return None, []

        dxs = []
        dys = []

        for g in grilles:
            w, s, e, n = map(float, g["bbox"])

            dxs.append(
                (e - w) / int(g["width"])
            )

            dys.append(
                (n - s) / int(g["height"])
            )

        dx = min(dxs)
        dy = min(dys)

        west = min(
            float(g["bbox"][0])
            for g in grilles
        )

        south = min(
            float(g["bbox"][1])
            for g in grilles
        )

        east = max(
            float(g["bbox"][2])
            for g in grilles
        )

        north = max(
            float(g["bbox"][3])
            for g in grilles
        )

        largeur = max(
            1,
            int(round((east - west) / dx)),
        )

        hauteur = max(
            1,
            int(round((north - south) / dy)),
        )

        offsets = []

        for g in [
            (s.get("toits") or {}).get("grille")
            for s in scenes
        ]:
            if not g or not g.get("bbox"):
                offsets.append((0, 0))
                continue

            gw, gs, ge, gn = map(float, g["bbox"])

            oi = int(
                round((gw - west) / dx)
            )

            oj = int(
                round((north - gn) / dy)
            )

            offsets.append((oi, oj))

        return {
            "bbox": [west, south, east, north],
            "width": largeur,
            "height": hauteur,
        }, offsets


    def _decaler_profil_toit(
        profil,
        offset_i,
        offset_j,
        dx_m,
        dy_m,
    ):
        p = copy.deepcopy(profil)

        for cle in ("surface", "pans"):
            bloc = p.get(cle)

            if isinstance(bloc, dict):
                if "i0" in bloc:
                    bloc["i0"] = int(
                        bloc["i0"] + offset_i
                    )

                if "j0" in bloc:
                    bloc["j0"] = int(
                        bloc["j0"] + offset_j
                    )

        for corps in p.get("corps", []) or []:
            if "cx" in corps:
                corps["cx"] = (
                    float(corps["cx"]) + dx_m
                )

            if "cy" in corps:
                corps["cy"] = (
                    float(corps["cy"]) + dy_m
                )

        return p


    def _concat_features(scenes, cle):
        features = []

        for s in scenes:
            obj = s.get(cle)

            if isinstance(obj, dict):
                features.extend(
                    copy.deepcopy(
                        obj.get("features", [])
                    )
                )

        return {
            "type": "FeatureCollection",
            "features": features,
        }


    def _fusionner_scenes(tuiles):
        scenes = []
        dossiers = []

        for t in tuiles:
            lat = float(t["lat"])
            lon = float(t["lon"])
            zone = zone_normalisee(
                t.get("taille")
            )

            dossier = cache.obtenir(
                lat,
                lon,
                construire=construire,
                zone=zone,
            )

            chemin = os.path.join(
                dossier,
                NOM_SCENE,
            )

            with gzip.open(
                chemin,
                "rt",
                encoding="utf-8",
            ) as f:
                scenes.append(json.load(f))

            dossiers.append(dossier)

        if not scenes:
            raise RuntimeError(
                "Aucune tuile à fusionner."
            )

        west = min(
            float(s["bbox"][0])
            for s in scenes
        )

        south = min(
            float(s["bbox"][1])
            for s in scenes
        )

        east = max(
            float(s["bbox"][2])
            for s in scenes
        )

        north = max(
            float(s["bbox"][3])
            for s in scenes
        )

        bbox = [west, south, east, north]

        centre_lon = (west + east) / 2
        centre_lat = (south + north) / 2

        m_lon = (
            111320
            * math.cos(
                math.radians(centre_lat)
            )
        )

        grille_toits, offsets = (
            _grille_toits_commune(
                scenes,
                bbox,
            )
        )

        batiments = {
            "type": "FeatureCollection",
            "features": [],
        }

        profils = {}

        source_toits = None
        resolution_toits = None
        ortho_toits = False

        for idx, s in enumerate(scenes):
            toits = s.get("toits") or {}

            if source_toits is None:
                source_toits = toits.get(
                    "source"
                )

            if toits.get("resolution_m") is not None:
                resolution_toits = (
                    toits["resolution_m"]
                    if resolution_toits is None
                    else min(
                        resolution_toits,
                        toits["resolution_m"],
                    )
                )

            ortho_toits = (
                ortho_toits
                or bool(toits.get("ortho"))
            )

            oi, oj = (
                offsets[idx]
                if idx < len(offsets)
                else (0, 0)
            )

            sb = s["bbox"]
            tile_lon = (
                float(sb[0]) + float(sb[2])
            ) / 2

            tile_lat = (
                float(sb[1]) + float(sb[3])
            ) / 2

            shift_x = (
                tile_lon - centre_lon
            ) * m_lon

            shift_y = (
                tile_lat - centre_lat
            ) * 111320

            source_profils = (
                toits.get("toits") or {}
            )

            for j, f in enumerate(
                (s.get("batiments") or {})
                .get("features", [])
            ):
                nf = copy.deepcopy(f)
                props = nf.setdefault(
                    "properties",
                    {},
                )

                ancienne = str(
                    props.get("cleabs")
                    or f"batiment-{j}"
                )

                nouvelle = (
                    f"{ancienne}__tuile_{idx}"
                )

                props["cleabs_source"] = ancienne
                props["cleabs"] = nouvelle

                batiments["features"].append(nf)

                profil = source_profils.get(
                    ancienne
                )

                if profil is not None:
                    profils[nouvelle] = (
                        _decaler_profil_toit(
                            profil,
                            oi,
                            oj,
                            shift_x,
                            shift_y,
                        )
                    )

        relief = _fusionner_reliefs(
            scenes,
            bbox,
        )

        constructions = {
            "reservoirs": [],
            "ponctuelles": [],
            "tours": [],
        }

        for s in scenes:
            c = s.get("constructions") or {}

            for cle in constructions:
                constructions[cle].extend(
                    copy.deepcopy(
                        c.get(cle, []) or []
                    )
                )

        eau = {
            "surfaces": [],
            "cours": [],
        }

        for s in scenes:
            e = s.get("eau") or {}

            eau["surfaces"].extend(
                copy.deepcopy(
                    e.get("surfaces", []) or []
                )
            )

            eau["cours"].extend(
                copy.deepcopy(
                    e.get("cours", []) or []
                )
            )

        lignes = []

        for s in scenes:
            l = s.get("lignes")

            if isinstance(l, list):
                lignes.extend(
                    copy.deepcopy(l)
                )

        houppiers = []
        masses = []

        for s in scenes:
            houppiers.extend(
                copy.deepcopy(
                    s.get("houppiers", []) or []
                )
            )

            masses.extend(
                copy.deepcopy(
                    s.get("masses", []) or []
                )
            )

        vegetation_sources = [
            s.get("vegetation") or {}
            for s in scenes
        ]

        hauteur_max = [
            v.get("hauteur_max")
            for v in vegetation_sources
            if isinstance(
                v.get("hauteur_max"),
                (int, float),
            )
        ]

        vegetation = {
            "source": next(
                (
                    v.get("source")
                    for v in vegetation_sources
                    if v.get("source")
                ),
                None,
            ),
            "couvert": any(
                bool(v.get("couvert"))
                for v in vegetation_sources
            ),
            "resolution_m": next(
                (
                    v.get("resolution_m")
                    for v in vegetation_sources
                    if v.get("resolution_m")
                    is not None
                ),
                None,
            ),
            "seuil_m": next(
                (
                    v.get("seuil_m")
                    for v in vegetation_sources
                    if v.get("seuil_m")
                    is not None
                ),
                None,
            ),
            "ortho": any(
                bool(v.get("ortho"))
                for v in vegetation_sources
            ),
            "veg_disponible": any(
                bool(v.get("veg_disponible"))
                for v in vegetation_sources
            ),
            "foret_disponible": any(
                bool(v.get("foret_disponible"))
                for v in vegetation_sources
            ),
            "hauteur_max": (
                max(hauteur_max)
                if hauteur_max
                else None
            ),
            "nb_ortho": sum(
                int(v.get("nb_ortho") or 0)
                for v in vegetation_sources
            ),
        }

        scene = {
            "version": max(
                int(s.get("version") or 0)
                for s in scenes
            ),
            "bbox": bbox,
            "batiments": batiments,
            "toits": {
                "source": source_toits,
                "resolution_m": resolution_toits,
                "toits": profils,
                "ortho": ortho_toits,
                "grille": grille_toits,
            },
            "routes": _concat_features(
                scenes,
                "routes",
            ),
            "constructions": constructions,
            "houppiers": houppiers,
            "masses": masses,
            "vegetation": vegetation,
            "relief": relief,
            "anneau": None,
            "eau": eau,
            "lignes": lignes,
        }

        largeur_m = (
            (east - west)
            * 111320
            * math.cos(
                math.radians(centre_lat)
            )
        )

        hauteur_m = (
            (north - south)
            * 111320
        )

        marge_m = 1000.0

        demi_lat_m = (
            hauteur_m / 2
            + marge_m
        )

        demi_lon_m = (
            largeur_m / 2
            + marge_m
        )

        dlat = demi_lat_m / 111320

        dlon = (
            demi_lon_m
            / (
                111320
                * math.cos(
                    math.radians(
                        centre_lat
                    )
                )
            )
        )

        anneau_bbox = [
            centre_lon - dlon,
            centre_lat - dlat,
            centre_lon + dlon,
            centre_lat + dlat,
        ]

        try:
            scene["anneau"] = (
                fetch_relief_anneau(
                    *anneau_bbox
                )
            )
        except Exception as exc:
            app.logger.warning(
                "Relief de zone périphérique indisponible : %s",
                exc,
            )

        def ortho_zone(emprise):
            try:
                return fetch_ortho_jpeg(
                    *emprise,
                    max_pixels=4096,
                )[0]
            except Exception as exc:
                app.logger.warning(
                    "Orthophoto 4096 px refusée, repli 2048 px : %s",
                    exc,
                )

                return fetch_ortho_jpeg(
                    *emprise,
                    max_pixels=2048,
                )[0]

        ortho = ortho_zone(bbox)
        ortho_anneau = ortho_zone(
            anneau_bbox
        )

        octets_scene = gzip.compress(
            json.dumps(
                scene,
                separators=(",", ":"),
            ).encode("utf-8"),
            6,
        )

        return {
            "scene": octets_scene,
            "ortho": ortho,
            "ortho_anneau": ortho_anneau,
            "centre_lat": centre_lat,
            "centre_lon": centre_lon,
            "largeur_m": largeur_m,
            "hauteur_m": hauteur_m,
            "taille_vue_m": int(
                math.ceil(
                    max(
                        largeur_m,
                        hauteur_m,
                    )
                    / 50
                )
                * 50
            ),
            "bbox": bbox,
            "anneau_bbox": anneau_bbox,
        }


    def dossier_scene(couche=None, prelire=False, detecteur=None):
        p = point()
        if p is None:
            return None, erreur(400, MESSAGE_POINT)
        *p, zone = p
        try:
            if couche == NOM_MONUMENTS:
                return cache.obtenir_monuments(*p, construire=construire, zone=zone), None
            if couche == NOM_OUVRAGES:
                return cache.obtenir_ouvrages(*p, construire=construire, zone=zone), None
            if couche == NOM_NUAGE:
                return cache.obtenir_nuage(*p, construire=construire, zone=zone), None
            if couche == COUCHE_PISCINES:
                return cache.obtenir_piscines(*p, construire=construire, zone=zone), None
            if couche == NOM_PANNEAUX:
                return cache.obtenir_panneaux(*p, construire=construire, zone=zone), None
            if couche == COUCHE_VEHICULES:
                return cache.obtenir_vehicules(*p, detecteur, construire=construire,
                                              zone=zone), None
            if prelire:
                # Les couches à part d'abord, en tâche de fond : leurs sources
                # répondent pendant que la scène se construit, et elles sont
                # souvent prêtes quand la page les demande.
                cache.prelire_monuments(*p, zone=zone)
                cache.prelire_ouvrages(*p, zone=zone)
                cache.prelire_vehicules(*p, zone=zone)
                cache.prelire_panneaux(*p, zone=zone)
            return cache.obtenir(*p, construire=construire, zone=zone), None
        except HorsEmprise as exc:
            return None, erreur(422, str(exc))
        except SceneIncomplete as exc:
            # Rien n'a été mis en cache : la prochaine demande réessaiera.
            app.logger.warning("Scène incomplète pour %s : %s", p, exc)
            return None, erreur(503, "Un service de l'IGN n'a pas répondu ("
                                f"{exc}). Rien n'a été mis en cache : réessayez "
                                "dans quelques instants.")
        except MonumentsIndisponibles as exc:
            app.logger.warning("Monuments OSM indisponibles pour %s : %s", p, exc)
            return None, erreur(503, "OpenStreetMap n'a pas répondu ("
                                f"{exc}). Rien n'a été mis en cache : réessayez "
                                "dans quelques instants.")
        except OuvragesIndisponibles as exc:
            app.logger.warning("Ouvrages indisponibles pour %s : %s", p, exc)
            return None, erreur(503, "Un service de l'IGN n'a pas répondu ("
                                f"{exc}). Rien n'a été mis en cache : réessayez "
                                "dans quelques instants.")
        except NuageIndisponible as exc:
            app.logger.warning("Nuage LiDAR HD indisponible pour %s : %s", p, exc)
            return None, erreur(503, "Un service de l'IGN n'a pas répondu ("
                                f"{exc}). Rien n'a été mis en cache : réessayez "
                                "dans quelques instants.")
        except VehiculesIndisponibles as exc:
            app.logger.warning("Détection indisponible pour %s : %s", p, exc)
            return None, erreur(503, "La détection sur l'orthophoto n'a pas abouti ("
                                f"{exc}). Rien n'a été mis en cache : réessayez "
                                "dans quelques instants.")
        except PanneauxIndisponibles as exc:
            app.logger.warning("Panneaux indisponibles pour %s : %s", p, exc)
            return None, erreur(503, f"Le registre des panneaux solaires n'a pas pu être lu ({exc}). "
                                "Rien n'a été mis en cache : réessayez dans quelques instants.")
        except VehiculesDesactives as exc:
            return None, erreur(400, f"{exc} : détecteurs de ce service : "
                                f"{', '.join(cache.lire_vehicules.detecteurs) or 'aucun'}.")

    # Tout ce qui sort du dossier d'une scène est revalidé à chaque demande :
    # une scène et ses couches peuvent être reconstruites sous la même adresse
    # (Cache.reconstruire), et une couche change avec le détecteur du service
    # et avec sa version. Gardée un jour par le navigateur, la scène
    # reconstruite n'apparaissait pas, et la couche des véhicules montrait
    # encore les véhicules sans les piscines après une reconstruction de
    # l'image. Le validateur est le fichier lui-même, son nom et l'instant de
    # son écriture : un 304 sans corps tant qu'il n'a pas été réécrit.
    def validateur(chemin):
        etat = os.stat(chemin)
        return f"{os.path.basename(chemin)}-{etat.st_mtime_ns}-{etat.st_size}"

    def servir_gzip(dossier, nom):
        chemin = os.path.join(dossier, nom)
        etag = validateur(chemin)
        if request.if_none_match.contains(etag):
            reponse = app.response_class(status=304)
        else:
            with open(chemin, "rb") as f:
                charge = f.read()
            # Déjà gzippé : annoncé tel quel, le navigateur le décompresse.
            reponse = app.response_class(charge, mimetype="application/json")
            reponse.headers["Content-Encoding"] = "gzip"
        reponse.set_etag(etag)
        reponse.headers["Cache-Control"] = "no-cache"
        return reponse

    @app.get("/")
    def page():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/scene")
    def scene():
        dossier, err = dossier_scene(prelire=True)
        if err:
            return err
        return servir_gzip(dossier, NOM_SCENE)

    @app.get("/api/monuments")
    def monuments():
        """La couche OSM, demandée par la page une fois la scène affichée :
        `null` quand l'emprise n'a aucune partie, le cas de presque partout."""
        dossier, err = dossier_scene(couche=NOM_MONUMENTS)
        if err:
            return err
        return servir_gzip(dossier, NOM_MONUMENTS)

    @app.get("/api/ouvrages")
    def ouvrages():
        """Murs, ponts, voies ferrées et terrains de sport, demandés par la
        page une fois la scène affichée : `null` quand l'emprise n'en a aucun."""
        dossier, err = dossier_scene(couche=NOM_OUVRAGES)
        if err:
            return err
        return servir_gzip(dossier, NOM_OUVRAGES)

    @app.get("/api/nuage")
    def nuage():
        """Le bâti du nuage LiDAR HD, demandé par la page une fois la scène
        affichée : `null` hors couverture LiDAR HD ou sans relief."""
        dossier, err = dossier_scene(couche=NOM_NUAGE)
        if err:
            return err
        return servir_gzip(dossier, NOM_NUAGE)

    def sans_detecteur(**vide):
        """Réponse des couches de l'orthophoto quand le service n'a pas de
        détecteur : le mode, que la page lit, et rien à dessiner. Ce n'est
        pas une erreur. Jamais gardée : le service peut être relancé avec un
        détecteur."""
        reponse = jsonify({"mode": MODE_PAR_DEFAUT, **vide})
        reponse.headers["Cache-Control"] = "no-store"
        return reponse

    @app.get("/api/piscines")
    def piscines():
        """Les piscines de l'orthophoto, demandées par la page une fois la
        scène affichée : une demi-seconde de détection, la première couche
        prête."""
        if not cache.lire_vehicules:
            return sans_detecteur(piscines=[])
        trouve, err = dossier_scene(couche=COUCHE_PISCINES)
        if err:
            return err
        return servir_gzip(*trouve)

    @app.get("/api/vehicules")
    def vehicules():
        """Les véhicules vus d'un détecteur (`detecteur=`), demandés par la
        page dans l'ordre que /api/sante lui donne, du rapide au lent : elle
        les réunit à mesure. Sans le paramètre, ou avec un détecteur que le
        service n'a pas : 400."""
        if not cache.lire_vehicules:
            return sans_detecteur(detecteur=None, vehicules=[])
        detecteur = request.args.get("detecteur")
        if not detecteur:
            return erreur(400, "Paramètre detecteur attendu : "
                          f"{', '.join(cache.lire_vehicules.detecteurs)}.")
        trouve, err = dossier_scene(couche=COUCHE_VEHICULES, detecteur=detecteur)
        if err:
            return err
        return servir_gzip(*trouve)

    @app.get("/api/panneaux")
    def panneaux():
        """Les panneaux solaires du registre OpenPVMapper, demandés par la
        page une fois la scène affichée. Sans registre, la réponse le dit
        (`actif: false`) : ce n'est pas une erreur."""
        if not cache.lire_panneaux:
            reponse = jsonify({"actif": False, "panneaux": []})
            reponse.headers["Cache-Control"] = "no-store"
            return reponse
        dossier, err = dossier_scene(couche=NOM_PANNEAUX)
        if err:
            return err
        return servir_gzip(dossier, NOM_PANNEAUX)
    @app.post("/api/zone/tuile")
    def generer_tuile_zone():
        """Construit une scène puis l'enregistre sur Geovalys."""
        if not stockage_configure():
            return erreur(503, "Stockage Geovalys non configuré.")

        data = request.get_json(silent=True) or {}

        nom = data.get("zone")
        slug = slug_zone(nom)

        if not slug:
            return erreur(400, "Nom de zone invalide.")

        try:
            lat = float(data["lat"])
            lon = float(data["lon"])
            zone = zone_normalisee(data.get("taille"))
        except (KeyError, TypeError, ValueError):
            return erreur(
                400,
                "lat, lon et taille sont attendus."
            )

        try:
            # Construction de la scène si elle n'existe pas encore.
            dossier = cache.obtenir(
                lat,
                lon,
                construire=construire,
                zone=zone,
            )
        except HorsEmprise as exc:
            return erreur(422, str(exc))
        except SceneIncomplete as exc:
            return erreur(503, str(exc))
        except Exception as exc:
            app.logger.exception(
                "Erreur génération tuile %.6f %.6f",
                lat,
                lon,
            )
            return erreur(500, str(exc))

        try:
            latn, lonn = point_normalise(lat, lon)

            tile_id = (
                f"{latn:.4f}_{lonn:.4f}"
                f"_z{zone or 'default'}"
            )

            # Prépare aussi l'orthophoto persistante du relief périphérique.
            assurer_ortho_anneau(dossier)

            envoyes = []

            # Fichiers indispensables à une scène archivée complète.
            for nom_fichier, mime in (
                (NOM_SCENE, "application/gzip"),
                (NOM_ORTHO, "image/jpeg"),
                (NOM_ORTHO_ANNEAU, "image/jpeg"),
            ):
                chemin = os.path.join(
                    dossier,
                    nom_fichier,
                )

                if not os.path.isfile(chemin):
                    continue

                with open(chemin, "rb") as f:
                    envoyer_octets(
                        slug,
                        "tile",
                        nom_fichier,
                        f,
                        tile=tile_id,
                        type_mime=mime,
                    )

                envoyes.append(nom_fichier)

            # Ajoute aussi toutes les couches déjà calculées.
            for nom_fichier in sorted(os.listdir(dossier)):
                if nom_fichier in envoyes:
                    continue

                chemin = os.path.join(
                    dossier,
                    nom_fichier,
                )

                if not os.path.isfile(chemin):
                    continue

                with open(chemin, "rb") as f:
                    envoyer_octets(
                        slug,
                        "tile",
                        nom_fichier,
                        f,
                        tile=tile_id,
                    )

                envoyes.append(nom_fichier)

            return jsonify({
                "ok": True,
                "tile": tile_id,
                "lat": latn,
                "lon": lonn,
                "taille": zone,
                "fichiers": envoyes,
            })         
        except Exception as exc:
            app.logger.exception(
                "Erreur sauvegarde Geovalys pour la tuile %.6f %.6f",
                lat,
                lon,
            )
            return erreur(
                502,
                f"Sauvegarde Geovalys impossible : {exc}"
            )

    @app.post("/api/zone/finaliser")
    def finaliser_zone():
        """Enregistre le GeoJSON et le manifeste de la zone sur Geovalys."""
        if not stockage_configure():
            return erreur(503, "Stockage Geovalys non configuré.")

        data = request.get_json(silent=True) or {}

        nom = data.get("nom")
        slug = slug_zone(nom)

        geojson = data.get("geojson")
        tuiles = data.get("tuiles", [])
        taille = data.get("taille")

        if not slug or not geojson:
            return erreur(
                400,
                "Nom et GeoJSON sont obligatoires."
            )

        zone_bytes = json.dumps(
            geojson,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        manifest = {
            "nom": nom,
            "slug": slug,
            "taille_tuile": taille,
            "nombre_tuiles": len(tuiles),
            "tuiles": tuiles,
        }

        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        try:
            fusion = _fusionner_scenes(tuiles)

            manifest["fusion"] = {
                "actif": True,
                "centre_lat": fusion["centre_lat"],
                "centre_lon": fusion["centre_lon"],
                "largeur_m": round(fusion["largeur_m"], 1),
                "hauteur_m": round(fusion["hauteur_m"], 1),
                "taille_vue_m": fusion["taille_vue_m"],
                "bbox": fusion["bbox"],
            }

            manifest_bytes = json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")

            envoyer_octets(
                slug,
                "zone",
                "zone.geojson",
                zone_bytes,
                type_mime="application/geo+json",
            )

            envoyer_octets(
                slug,
                "zone",
                "manifest.json",
                manifest_bytes,
                type_mime="application/json",
            )

            envoyer_octets(
                slug,
                "zone",
                "zone_scene.json.gz",
                fusion["scene"],
                type_mime="application/gzip",
            )

            envoyer_octets(
                slug,
                "zone",
                "zone_ortho.jpg",
                fusion["ortho"],
                type_mime="image/jpeg",
            )

            envoyer_octets(
                slug,
                "zone",
                "zone_ortho_anneau.jpg",
                fusion["ortho_anneau"],
                type_mime="image/jpeg",
            )
        except Exception as exc:
            app.logger.exception(
                "Erreur finalisation de la zone %s",
                slug,
            )
            return erreur(
                502,
                f"Finalisation Geovalys impossible : {exc}"
            )

        return jsonify({
            "ok": True,
            "zone": slug,
            "nombre_tuiles": len(tuiles),
            "fusion": True,
            "taille_vue_m": manifest["fusion"]["taille_vue_m"],
        })

        
    @app.get("/api/ortho")
    def ortho():
        dossier, err = dossier_scene()
        if err:
            return err
        # Revalidée comme la scène (servir_gzip) : send_from_directory donne
        # déjà son validateur, tiré de l'instant d'écriture du fichier.
        reponse = send_from_directory(dossier, NOM_ORTHO, mimetype="image/jpeg")
        reponse.headers["Cache-Control"] = "no-cache"
        return reponse

    @app.post("/api/reconstruire")
    def reconstruire():
        """Le bouton « Reconstruire la scène » de la page : la scène et ses
        couches sont mises de côté, et la page, rechargée, les reconstruit
        d'après les données de l'IGN du moment (Cache.reconstruire)."""
        p = point()
        if p is None:
            return erreur(400, MESSAGE_POINT)
        *p, zone = p
        try:
            cache.reconstruire(*p, zone=zone)
        except HorsEmprise as exc:
            return erreur(422, str(exc))
        except ReconstructionRefusee as exc:
            return erreur(exc.code, f"Reconstruction refusée : {exc}.")
        reponse = jsonify({"etat": "a_reconstruire"})
        reponse.headers["Cache-Control"] = "no-store"
        return reponse, 202

    @app.get("/api/avancement")
    def avancement():
        """Pour la page qui attend sa scène : jamais mis en cache, il change
        d'une seconde à l'autre."""
        p = point()
        if p is None:
            return erreur(400, MESSAGE_POINT)
        try:
            reponse = jsonify(cache.avancement(*p))
        except HorsEmprise as exc:
            return erreur(422, str(exc))
        reponse.headers["Cache-Control"] = "no-store"
        return reponse

    @app.get("/api/sante")
    def sante():
        """Contrôle de vie, et ce que le service sait détecter sur
        l'orthophoto : la page y lit les détecteurs à demander, dans
        l'ordre."""
        lecteur = cache.lire_vehicules
        reponse = jsonify({"ok": True, "vehicules": {
            "mode": lecteur.mode if lecteur else MODE_PAR_DEFAUT,
            "detecteurs": list(lecteur.detecteurs) if lecteur else []},
            "panneaux": {"actif": bool(cache.lire_panneaux),
                         "source": REGISTRE_LICENCE if cache.lire_panneaux else None}})
        reponse.headers["Cache-Control"] = "no-store"
        return reponse

    return app


# Les toitures se calculent sur un bassin de processus (vue3d/toits.py), que
# le service autorise : gunicorn et flask gardent leur script principal, que
# chaque processus du bassin réexécute. Il naît dès maintenant, dans un fil à
# part, et non à la première scène, qui le trouve prêt. Sauf dans le
# processus qui surveille les fichiers sous `flask run --debug` : le
# rechargeur de werkzeug y importe aussi ce module, sans jamais y servir de
# scène, et un second bassin y naissait (22 processus au lieu de 11,
# constaté le 2 octobre 2026). Le processus qui sert porte WERKZEUG_RUN_MAIN.
autoriser_bassin(chauffer=not (os.environ.get("FLASK_DEBUG") == "1"
                               and "WERKZEUG_RUN_MAIN" not in os.environ))

# Le détecteur de VUE3D_VEHICULES et le registre de VUE3D_PANNEAUX sont
# chargés ici, une fois : demandés sans leur réseau ou leur base, ils
# arrêtent le démarrage, avec la commande qui les produit. Sur deux lignes :
# une trace d'erreur sur la ligne commune laissait croire que les panneaux
# étaient en cause quand c'était le réseau des véhicules qui manquait.
lire_vehicules = lecteur_vehicules()
lire_panneaux = lecteur_panneaux()
app = creer_app(lire_vehicules=lire_vehicules, lire_panneaux=lire_panneaux)
