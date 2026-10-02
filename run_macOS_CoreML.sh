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
# Le cache est ./cache, pas le volume du conteneur : sans copie, chaque scène
# est reconstruite. Les réseaux de l'image évitent de refaire l'export :
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

if [ ! -x .venv/bin/flask ]; then
    echo "Pas de .venv : python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi
# Un .venv créé par uv n'a pas pip : la commande proposée est celle qui marche.
if .venv/bin/python -c "import pip" 2>/dev/null; then
    INSTALLER=".venv/bin/python -m pip install"
else
    INSTALLER="uv pip install --python .venv/bin/python"
fi
if [ "$VUE3D_VEHICULES" != "aucun" ] && ! .venv/bin/python -c "import onnxruntime" 2>/dev/null; then
    echo "VUE3D_VEHICULES=$VUE3D_VEHICULES demande onnxruntime :" \
         "$INSTALLER -r requirements-vehicules.txt" >&2
    exit 1
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
