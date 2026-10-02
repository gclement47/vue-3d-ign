#!/usr/bin/env bash
# Exporte en ONNX les réseaux des détecteurs de véhicules, sans Docker : ce
# que fait l'étage `export` du Dockerfile, avec les mêmes versions, dans un
# environnement Python jetable effacé à la fin (1,3 Go une fois installé).
# Seuls les .onnx restent, dans le dossier donné. Les poids d'origine sont
# téléchargés chez leurs auteurs et ne sont jamais gardés : ceux de YOLO sont
# sous AGPL-3.0, et les deux réseaux sont entraînés sur DOTA, réservé à un
# usage académique.
#
#   outils/preparer_modeles.sh tous ./modeles      # rtmdet | yolo | tous
#
# Exporté ainsi sur un Mac M4 le 2 octobre 2026, en une à deux minutes une
# fois les paquets téléchargés : vehicules-rtmdet.onnx identique à l'octet à
# celui de l'image Docker, vehicules-yolo.onnx identique hormis la date
# d'export qu'Ultralytics inscrit dans ses métadonnées.
set -euo pipefail

MODE="${1:-tous}"
DOSSIER="${2:-./modeles}"
case "$MODE" in
    rtmdet|yolo|tous) ;;
    *) echo "Détecteur « $MODE » : attendu rtmdet, yolo ou tous" >&2; exit 1 ;;
esac
OUTILS="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DOSSIER"
DOSSIER="$(cd "$DOSSIER" && pwd)"

TRAVAIL="$(mktemp -d)"
trap 'rm -rf "$TRAVAIL"' EXIT

# Python 3.12, celui de l'image. Un venv créé par uv n'a pas pip sans --seed.
if command -v uv >/dev/null 2>&1; then
    uv venv -q --seed --python 3.12 "$TRAVAIL/venv"
elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$TRAVAIL/venv"
else
    echo "Python 3.12 introuvable : installez uv (brew install uv) ou python3.12." >&2
    exit 1
fi
PY="$TRAVAIL/venv/bin/python"
installer() { "$PY" -m pip install -q --no-cache-dir "$@"; }

echo "Export des réseaux ($MODE) vers $DOSSIER : environ 1 Go à télécharger (torch)…"
# L'index CPU de PyTorch, comme le Dockerfile : sur x86-64, le torch de PyPI
# tire deux gigaoctets de CUDA.
installer torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
# Le script d'export écrit les poids téléchargés à côté des réseaux : il
# travaille ici, et seuls les .onnx passent dans le dossier donné.
mkdir "$TRAVAIL/sortie"
cd "$OUTILS"
if [ "$MODE" != yolo ]; then
    installer -r requirements-export-rtmdet.txt
    installer --no-deps mmdet==3.0.0 mmrotate==1.0.0rc1
    "$PY" exporter_vehicules.py rtmdet "$TRAVAIL/sortie"
fi
if [ "$MODE" != rtmdet ]; then
    installer -r requirements-export-yolo.txt
    "$PY" exporter_vehicules.py yolo "$TRAVAIL/sortie"
fi
mv "$TRAVAIL"/sortie/*.onnx "$DOSSIER"/
echo "Réseaux prêts : $(cd "$DOSSIER" && ls *.onnx | tr '\n' ' ')"
