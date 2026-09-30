# Détecteur de véhicules embarqué (vue3d/vehicules.py) : aucun, rtmdet, yolo
# ou tous. Par défaut aucun : l'image est alors celle du service sans la
# couche des véhicules, sans moteur d'inférence ni réseau.
ARG VEHICULES=aucun
# Registre des panneaux solaires (vue3d/panneaux.py) : vide, ou « oui » pour
# télécharger le registre OpenPVMapper (211 Mo, CC-BY 4.0) et en faire la base
# de la couche.
ARG PANNEAUX=

# --- Étage d'export -------------------------------------------------------------
# Produit les réseaux au format ONNX, et rien d'autre n'en sort : torch,
# MMRotate, Ultralytics et les poids d'origine restent ici. Sans détecteur,
# l'étage ne fait que créer un dossier vide.
FROM python:3.12-slim AS export
ARG VEHICULES
ARG PANNEAUX
WORKDIR /export
COPY outils/exporter_vehicules.py outils/requirements-export-rtmdet.txt outils/requirements-export-yolo.txt ./
# Le registre des panneaux : lu une fois, rangé dans une base SQLite. Le
# script importe le module de la couche, qui ne dépend que de la bibliothèque
# standard pour cela.
COPY outils/preparer_panneaux.py outils/
COPY vue3d/__init__.py vue3d/panneaux.py vue3d/
RUN set -e; mkdir /donnees; \
    if [ -n "$PANNEAUX" ]; then python outils/preparer_panneaux.py /donnees/panneaux.sqlite; fi
RUN set -e; mkdir /modeles; \
    case "$VEHICULES" in \
      aucun) exit 0 ;; \
      rtmdet|yolo|tous) ;; \
      *) echo "VEHICULES=$VEHICULES : attendu aucun, rtmdet, yolo ou tous" >&2; exit 1 ;; \
    esac; \
    # OpenCV, dépendance de mmcv-lite et d'Ultralytics, veut ces bibliothèques.
    apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*; \
    # L'index CPU : sur x86-64, le torch de PyPI tire deux gigaoctets de CUDA.
    pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 \
      --index-url https://download.pytorch.org/whl/cpu; \
    if [ "$VEHICULES" != yolo ]; then \
      pip install --no-cache-dir -r requirements-export-rtmdet.txt; \
      pip install --no-cache-dir --no-deps mmdet==3.0.0 mmrotate==1.0.0rc1; \
      python exporter_vehicules.py rtmdet /modeles; \
    fi; \
    if [ "$VEHICULES" != rtmdet ]; then \
      pip install --no-cache-dir -r requirements-export-yolo.txt; \
      python exporter_vehicules.py yolo /modeles; \
    fi; \
    # Seuls les .onnx passent à l'image finale.
    find /modeles -type f ! -name '*.onnx' -delete

# --- Image du service -----------------------------------------------------------
FROM python:3.12-slim
ARG VEHICULES
ARG PANNEAUX

# Pas de .pyc, sortie non tamponnée : les journaux arrivent tout de suite dans
# `docker compose logs`. Le détecteur et le registre sont ceux avec lesquels
# l'image a été construite : le serveur refuse de démarrer si on lui en
# demande d'autres.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VUE3D_CACHE=/cache \
    VUE3D_MODELES=/modeles \
    VUE3D_VEHICULES=$VEHICULES \
    VUE3D_PANNEAUX=${PANNEAUX:+/donnees/panneaux.sqlite}

WORKDIR /app
COPY requirements.txt requirements-vehicules.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$VEHICULES" != aucun ]; then pip install --no-cache-dir -r requirements-vehicules.txt; fi

COPY --from=export /modeles /modeles
COPY --from=export /donnees /donnees
COPY vue3d ./vue3d

# Utilisateur sans privilège, propriétaire du cache.
RUN useradd --create-home --uid 1000 vue3d && mkdir -p /cache && chown vue3d /cache
USER vue3d

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/sante', timeout=4)"

# La première construction d'une scène dure 20 à 40 s : le délai de gunicorn
# doit la couvrir largement. Des fils plutôt que des processus, pour que le
# verrou par point empêche deux constructions simultanées de la même scène.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", \
     "--timeout", "180", "--access-logfile", "-", "vue3d.app:app"]
