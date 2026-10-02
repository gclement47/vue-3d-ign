#!/usr/bin/env bash
# Lance le serveur sans Docker, avec le Python du .venv : le même service que
# `docker compose up -d`, sur le même port, au choix de l'un ou de l'autre.
# Les variables sont celles de docker-compose.yml, avec les mêmes défauts,
# sauf VUE3D_VEHICULES : `tous` ici, comme run_docker.sh, puisque faire
# tourner les détecteurs sur CoreML est la raison de lancer ce script.
#
#   ./run_macOS_CoreML.sh                        # les deux détecteurs, sur CoreML
#   VUE3D_VEHICULES=rtmdet ./run_macOS_CoreML.sh # aucun | rtmdet | yolo | tous
#   VUE3D_PANNEAUX=oui ./run_macOS_CoreML.sh     # ou le chemin de la base
#   VUE3D_PORT=8081 ./run_macOS_CoreML.sh        # à côté du conteneur, qui garde le 8080
#   VUE3D_MOTEUR=processeur ./run_macOS_CoreML.sh  # sans CoreML
#
# Sur un Mac, les détecteurs tournent ici sur CoreML, que le conteneur n'a
# pas (vue3d/vehicules.py, MOTEURS) : c'est la raison de lancer sans Docker.
#
# Seul prérequis : uv (brew install uv) ou python3.12. Le script crée le
# .venv s'il n'existe pas, et y installe à chaque lancement les versions
# exactes de requirements.txt (et de requirements-vehicules.txt avec un
# détecteur) : rien à faire après un git pull qui en change une.
#
# Les réseaux des détecteurs ne sont pas dans le dépôt : s'il en manque dans
# ./modeles, le script les exporte d'abord (outils/preparer_modeles.sh, une
# à deux minutes et environ 1 Go à télécharger, une seule fois). Le cache est
# ./cache, pas le volume du conteneur : sans copie, chaque scène est
# reconstruite. Copier ceux d'un conteneur évite l'un et l'autre :
#
#   docker cp vue-3d-ign-vue3d-1:/cache ./cache
#   docker cp vue-3d-ign-vue3d-1:/modeles ./modeles
#
# Revenir à Docker : Ctrl-C ici, puis `docker compose start`.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${VUE3D_PORT:-8080}"
export VUE3D_VEHICULES="${VUE3D_VEHICULES:-tous}"
export VUE3D_MODELES="${VUE3D_MODELES:-./modeles}"
export VUE3D_CACHE="${VUE3D_CACHE:-./cache}"
# docker-compose.yml dit « oui », le serveur attend le chemin de la base.
if [ "${VUE3D_PANNEAUX:-}" = "oui" ]; then
    export VUE3D_PANNEAUX=./donnees/panneaux.sqlite
fi

# Le .venv, en Python 3.12 comme l'image : en 3.14, avec les mêmes versions
# de numpy et de shapely, la segmentation des arbres rend 0 houppier sans la
# moindre erreur, et le cache, qui ne périme pas, garderait ces scènes.
if [ ! -x .venv/bin/python ]; then
    echo "Création du .venv (Python 3.12)…"
    if command -v uv >/dev/null 2>&1; then
        uv venv -q --python 3.12 .venv
    elif command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv .venv
    else
        echo "Python 3.12 introuvable : installez uv (brew install uv) ou python3.12." >&2
        exit 1
    fi
fi
VERSION="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$VERSION" != "3.12" ]; then
    echo "Le .venv est en Python $VERSION ; il faut 3.12 (en 3.14, 0 houppier)." \
         "Le recréer : rm -rf .venv && $0" >&2
    exit 1
fi
# Un .venv créé par uv n'a pas pip ; uv installe aussi dans un .venv qui l'a.
if command -v uv >/dev/null 2>&1; then
    INSTALLER=(uv pip install -q --python .venv/bin/python)
elif .venv/bin/python -c "import pip" 2>/dev/null; then
    INSTALLER=(.venv/bin/python -m pip install -q)
else
    echo "Ni uv ni pip pour installer les dépendances : brew install uv." >&2
    exit 1
fi
# Versions exactes, donc sans réseau quand elles sont déjà là (0,07 s avec uv).
DEPENDANCES=(-r requirements.txt)
[ "$VUE3D_VEHICULES" != "aucun" ] && DEPENDANCES+=(-r requirements-vehicules.txt)
"${INSTALLER[@]}" "${DEPENDANCES[@]}"
# Les réseaux du mode, exportés au premier lancement s'ils manquent : un clone
# neuf n'en a pas, l'image Docker les fabrique à sa construction.
case "$VUE3D_VEHICULES" in
    tous) RESEAUX="rtmdet yolo" ;;
    aucun) RESEAUX="" ;;
    *) RESEAUX="$VUE3D_VEHICULES" ;;
esac
MANQUANTS=""
for r in $RESEAUX; do
    [ -f "$VUE3D_MODELES/vehicules-$r.onnx" ] || MANQUANTS="$MANQUANTS $r"
done
if [ -n "$MANQUANTS" ]; then
    echo "Réseaux absents de $VUE3D_MODELES :$MANQUANTS. Export avant le démarrage" \
         "(sans détecteur : VUE3D_VEHICULES=aucun $0)."
    MODE_EXPORT="tous"
    [ "$(echo $MANQUANTS)" = "rtmdet" ] && MODE_EXPORT="rtmdet"
    [ "$(echo $MANQUANTS)" = "yolo" ] && MODE_EXPORT="yolo"
    if ! outils/preparer_modeles.sh "$MODE_EXPORT" "$VUE3D_MODELES"; then
        echo "L'export des réseaux a échoué. Pour démarrer sans détecteur :" \
             "VUE3D_VEHICULES=aucun $0" >&2
        exit 1
    fi
fi
# Le contrôle n'est pas laissé à Flask : macOS accepte 127.0.0.1:8080 pendant
# que Docker écoute *:8080 (essayé le 2026-10-01), et le navigateur tombe
# alors sur l'un ou sur l'autre selon qu'il résout localhost en IPv4 ou IPv6.
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Le port $PORT est déjà écouté (le conteneur ?) : docker compose stop," \
         "ou VUE3D_PORT=8081 $0" >&2
    exit 1
fi

exec .venv/bin/flask --app vue3d.app run --port "$PORT"
