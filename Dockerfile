FROM python:3.12-slim

# Pas de .pyc, sortie non tamponnée : les journaux arrivent tout de suite dans
# `docker compose logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VUE3D_CACHE=/cache

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
