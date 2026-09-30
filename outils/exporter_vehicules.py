"""Exporte en ONNX les détecteurs de véhicules de vue3d/vehicules.py.

    python outils/exporter_vehicules.py rtmdet modeles/
    python outils/exporter_vehicules.py yolo modeles/

Le serveur ne lit que le fichier .onnx, avec onnxruntime : ni torch, ni
MMRotate, ni Ultralytics n'entrent dans l'image finale. Ce script tourne une
fois, dans l'étage de construction du Dockerfile, avec les dépendances de
outils/requirements-export-*.txt. Les poids sont téléchargés ici et ne sont
jamais dans le dépôt : ceux de YOLO sont sous AGPL-3.0, ceux de RTMDet-R sous
Apache-2.0, et les deux réseaux sont entraînés sur DOTA, dont les images sont
réservées à un usage académique.

Les deux exports prennent la même entrée — une tuile RGB de COTE pixels,
valeurs de 0 à 1, en (1, 3, COTE, COTE) — et chacun rend ses boîtes orientées
dans sa disposition, que vue3d/vehicules.py sait lire.

RTMDet-R : MMRotate 1.0.0rc1 demande mmcv complet, dont les opérations
compilées ne s'installent pas sans chaîne de compilation. Le réseau n'en a
pas besoin : seule la suppression des doublons en dépend, et elle est refaite
côté serveur. On installe donc mmcv-lite, et `mmcv.ops` est remplacé par un
module factice le temps de l'import. L'export a été vérifié contre le réseau
d'origine : les boîtes à un centième de pixel près, les scores à 1e-5 près.
"""

import os
import sys
import urllib.request

# Côté de la tuile présentée au réseau : celui de l'entraînement sur DOTA.
COTE = 1024

RTMDET_POIDS = ("https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/"
                "rotated_rtmdet_s-3x-dota/rotated_rtmdet_s-3x-dota-11f6ccf5.pth")
RTMDET_CONFIG = "rotated_rtmdet_s-3x-dota.py"
YOLO_POIDS = "yolo11s-obb.pt"


def _neutraliser_mmcv_ops():
    """Rend importables mmdet et mmrotate sans les opérations compilées de mmcv.

    Tout module `mmcv.ops.*` (et `e2cnn`, dont dépend un réseau qu'on
    n'utilise pas) devient un module dont chaque attribut est une classe qui
    accepte tout : héritage, appel, attributs.
    """
    import collections
    import collections.abc
    import importlib.abc
    import importlib.machinery
    import types

    # MMRotate 1.0.0rc1 importe encore collections.Sequence, retiré de
    # Python 3.10.
    collections.Sequence = collections.abc.Sequence

    class Meta(type):
        def __getattr__(cls, nom):
            if nom.startswith("__"):
                raise AttributeError(nom)
            return factice(nom)

    def factice(nom):
        def attribut(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            return factice(n)()
        return Meta(nom, (), {"__init__": lambda self, *a, **k: None,
                              "__getattr__": attribut,
                              "__call__": lambda self, *a, **k: factice(nom)(),
                              "__iter__": lambda self: iter(())})

    class Module(types.ModuleType):
        def __getattr__(self, nom):
            if nom.startswith("__"):
                raise AttributeError(nom)
            return factice(nom)

    class Chercheur(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, nom, path=None, target=None):
            if nom.split(".")[0] == "e2cnn" or nom == "mmcv.ops" or nom.startswith("mmcv.ops."):
                return importlib.machinery.ModuleSpec(nom, self, is_package=True)

        def create_module(self, spec):
            module = Module(spec.name)
            module.__path__ = []
            return module

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, Chercheur())
    import mmcv
    # mmdet 3.0.0 refuse mmcv ≥ 2.1 ; mmcv-lite n'existe plus en 2.0 pour les
    # Python récents, et rien de ce qu'on utilise n'a changé entre les deux.
    mmcv.__version__ = "2.0.1"


def charger_rtmdet(dossier):
    """Le réseau RTMDet-R d'origine, poids chargés, prêt à l'inférence."""
    import torch
    _neutraliser_mmcv_ops()
    import mmdet.models  # noqa: F401
    import mmrotate
    import mmrotate.models  # noqa: F401
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmrotate.registry import MODELS

    poids = os.path.join(dossier, os.path.basename(RTMDET_POIDS))
    if not os.path.exists(poids):
        urllib.request.urlretrieve(RTMDET_POIDS, poids)
    cfg = Config.fromfile(os.path.join(os.path.dirname(mmrotate.__file__), ".mim", "configs",
                                       "rotated_rtmdet", RTMDET_CONFIG))
    init_default_scope("mmrotate")
    cfg.model.backbone.init_cfg = None            # pas de poids ImageNet à télécharger
    modele = MODELS.build(cfg.model)
    # weights_only=False : le fichier d'OpenMMLab embarque ses métadonnées.
    etat = torch.load(poids, map_location="cpu", weights_only=False)["state_dict"]
    modele.load_state_dict(etat, strict=True)     # strict : une clé en trop ou en moins arrête tout
    return modele.eval(), cfg


def reseau_rtmdet(modele, cfg):
    """Le réseau enveloppé : RGB 0-1 en entrée, boîtes décodées en sortie.

    Sortie (1, N, 20) : cx, cy, largeur, hauteur (pixels de la tuile), angle
    (radians), puis les 15 scores de classe, dans l'ordre de MMRotate.
    """
    import torch
    from mmrotate.structures.bbox import distance2obb

    class Reseau(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modele = modele
            pre = cfg.model.data_preprocessor
            # Le réseau a été entraîné en BGR, valeurs de 0 à 255.
            self.register_buffer("moyenne", torch.tensor(pre.mean).view(1, 3, 1, 1))
            self.register_buffer("ecart", torch.tensor(pre.std).view(1, 3, 1, 1))
            points = []
            for pas in (8, 16, 32):
                n = COTE // pas
                ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
                points.append(torch.stack([xs, ys], -1).reshape(-1, 2).float() * pas)
            self.register_buffer("points", torch.cat(points))

        def forward(self, x):
            x = (x.flip(1) * 255.0 - self.moyenne) / self.ecart
            cls, reg, ang = self.modele.bbox_head(self.modele.extract_feat(x))
            a_plat = lambda niveaux, c: torch.cat(                         # noqa: E731
                [t.permute(0, 2, 3, 1).reshape(1, -1, c) for t in niveaux], 1)
            scores = a_plat(cls, cls[0].shape[1]).sigmoid()
            boites = distance2obb(self.points, torch.cat([a_plat(reg, 4), a_plat(ang, 1)], -1)[0],
                                  "le90")
            return torch.cat([boites.unsqueeze(0), scores], -1)

    return Reseau().eval()


def exporter_rtmdet(dossier):
    import torch
    reseau = reseau_rtmdet(*charger_rtmdet(dossier))
    sortie = os.path.join(dossier, "vehicules-rtmdet.onnx")
    with torch.no_grad():
        torch.onnx.export(reseau, torch.zeros(1, 3, COTE, COTE), sortie, input_names=["image"],
                          output_names=["boites"], opset_version=17, dynamo=False)
    return sortie


def exporter_yolo(dossier):
    """L'export officiel d'Ultralytics : sortie (1, 20, N), cx, cy, largeur,
    hauteur, 15 scores, angle."""
    from ultralytics import YOLO
    ici = os.getcwd()
    os.chdir(dossier)                 # Ultralytics télécharge et exporte sur place
    try:
        produit = YOLO(YOLO_POIDS).export(format="onnx", imgsz=COTE, opset=17, simplify=False)
    finally:
        os.chdir(ici)
    sortie = os.path.join(dossier, "vehicules-yolo.onnx")
    os.replace(os.path.join(dossier, produit), sortie)
    return sortie


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("rtmdet", "yolo"):
        sys.exit(__doc__)
    os.makedirs(sys.argv[2], exist_ok=True)
    chemin = {"rtmdet": exporter_rtmdet, "yolo": exporter_yolo}[sys.argv[1]](
        os.path.abspath(sys.argv[2]))
    print(f"{chemin} : {os.path.getsize(chemin) / 1e6:.1f} Mo")
