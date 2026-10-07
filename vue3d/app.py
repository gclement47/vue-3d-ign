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

        return jsonify({
            "ok": True,
            "zone": slug,
            "nombre_tuiles": len(tuiles),
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
