"""Prépare la base des panneaux solaires (vue3d/panneaux.py) depuis le registre
OpenPVMapper.

    python outils/preparer_panneaux.py donnees/panneaux.sqlite [individual-regions.zip]

Sans archive en argument, elle est téléchargée depuis Zenodo (211 Mo,
CC-BY 4.0). Les douze GeoJSON régionaux sont lus, reprojetés en WGS84 et
rangés dans la base, avec leur index spatial : 460 755 installations,
quelques minutes. Tourne dans l'étage `export` du Dockerfile quand l'image
est construite avec `VUE3D_PANNEAUX=oui`.
"""

import os
import sys
import tempfile
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vue3d.panneaux import REGISTRE_URL, preparer  # noqa: E402


def main(sortie, archive=None):
    with tempfile.TemporaryDirectory() as tmp:
        if archive is None:
            archive = os.path.join(tmp, "individual-regions.zip")
            print(f"Téléchargement de {REGISTRE_URL} …")
            urllib.request.urlretrieve(REGISTRE_URL, archive)
        with zipfile.ZipFile(archive) as z:
            noms = [n for n in z.namelist()
                    if n.endswith(".geojson.gz") and not os.path.basename(n).startswith("._")]
            z.extractall(tmp, noms)
        os.makedirs(os.path.dirname(os.path.abspath(sortie)), exist_ok=True)
        n = preparer([os.path.join(tmp, nom) for nom in sorted(noms)], sortie)
    print(f"{sortie} : {n} installation(s), {os.path.getsize(sortie) / 1e6:.0f} Mo")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
