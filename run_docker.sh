#!/usr/bin/env bash
# Repart de zéro dans Docker : `-v` efface le volume des scènes, chaque lieu
# sera reconstruit. Image avec les deux détecteurs de véhicules.
set -euo pipefail
cd "$(dirname "$0")"

docker compose down -v
VUE3D_VEHICULES=tous docker compose up -d --build
