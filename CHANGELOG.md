# Journal des changements

Le projet n'a pas de versions numérotées : ce qui tient lieu de version est le
**format de la scène** (`SCENE_VERSION` dans `vue3d/scene.py`). L'incrémenter
invalide tout le cache — chaque lieu est reconstruit à sa première ouverture —
et c'est donc à lui que ce journal s'accroche. Les changements qui ne touchent
pas la scène (page, outils, documentation) sont rangés sous la version en
cours à leur date.

Les plus récents en premier. Les chiffres cités sont ceux mesurés au moment du
changement, sur des lieux publics ; le détail et la méthode sont dans les
messages de commit et dans les commentaires des modules.

## Scène v13 — 30 septembre 2026

### Corrigé

- **Le château de Chambord n'est plus réduit à ses terrasses.** Son donjon et
  ses tours, un tiers de l'emprise, étaient pris pour « un arbre au-dessus du
  toit » et écartés. Là où l'orthophoto ne voit pas de vert, ce niveau haut
  est le bâtiment : il reçoit sa forme mesurée au LiDAR. 26 bâtiments sur
  1 516 y gagnent, dont un immeuble d'Annecy de 22,5 m dessiné à 3,7 m.
- **Un toit résumé ne dépasse plus de son bâtiment.** Posé sur sa boîte
  entière, il débordait de 31 % de l'emprise en médiane (391 toits, huit
  lieux) : vu du ciel, un rectangle de photo aérienne plus grand que la
  maison. Il est découpé sur l'emprise, pignons compris — une tour ronde
  d'OpenStreetMap porte un toit rond.
- **Les toits plats portent la photo aérienne**, comme ce que les corps de
  toit laissent à découvert, au lieu de la couleur des murs.
- Les marches d'une surface mesurée prennent la teinte du toit, plus celle
  des murs : les tours coniques ne se mouchettent plus d'orange.

### Ajouté

- `outils/verifier-geometrie.mjs` : exécute sous Node les fonctions
  géométriques de la page et vérifie orientation, fermeture et volumes.

## Scène v12 — 30 septembre 2026

### Corrigé

- **Plus de végétation plus haute que les monuments.** À Notre-Dame de Paris,
  les flèches des grues du chantier, au-dessus des arbres des quais, sortaient
  en houppiers de 52 à 88 m — plus hauts que les tours (69 m) —, et le débord
  des tours sur leur emprise en houppiers et en masses de 60 à 66 m. Hors des
  forêts BD TOPO, où les vrais arbres montent à 43 m (sapins des Vosges), le
  sursol de plus de 40 m n'est plus dessiné. Seuil fixé sur 22 lieux, dont six
  forêts.
- **Plus d'arbres dans l'eau.** Le laser ne revient pas de l'eau : entre deux
  quais, le modèle de hauteur lit leur hauteur en pleine rivière. 438 des
  1 175 houppiers de la scène de Notre-Dame étaient plantés dans la Seine,
  verte à l'orthophoto ; il en reste 807, sur les quais et dans les squares.
  Les étendues d'eau permanentes sortent du sursol, en retrait de 3 m sur la
  rive pour garder le feuillage qui la surplombe. Même effet à Strasbourg, à
  Lyon, au Pont du Gard.

### Limites connues

- En forêt, où rien n'est plafonné, un arbre accroché à une falaise garde une
  hauteur comptée depuis le pied de la paroi : 7 houppiers de 42 à 60 m à
  Rocamadour.
- Le plafond efface aussi ce qui dépasse d'un bâtiment hors de son emprise,
  comme le toit du Stade de France, qui n'était qu'une couronne de masses.

## Scène v11 — 30 septembre 2026

### Ajouté

- **Réservoirs et constructions ponctuelles** de la BD TOPO, dans la scène
  (`vue3d/constructions.py`) : citernes et châteaux d'eau montés à leur
  hauteur, torchères, cheminées, antennes et mâts d'éclairage. Hauteur BD
  TOPO, à défaut LiDAR ; l'infobulle dit laquelle. Bouton **Réservoirs, mâts**.
- **Les citernes sortent du sursol.** Sur quatre parcs de stockage, 2 426
  houppiers et masses sur 5 400 étaient posés sur l'un des 156 réservoirs ; à
  Feyzin, il en reste 82 sur 545.
- **Couche des ouvrages**, chargée après la scène (`vue3d/ouvrages.py`,
  `GET /api/ouvrages`) : murs, ponts, voies ferrées et terrains de sport. La
  hauteur d'un mur ou d'un pont est l'altitude de ses sommets BD TOPO moins le
  relief : 30 murs de rempart de 3 à 25 m à Carcassonne, l'aqueduc du Pont du
  Gard à 47,8 m du Gardon. Les masses de sursol qu'un mur ou un tablier
  explique lui laissent la place. Bouton **Murs, ponts, rails**.
- La couche des ouvrages a sa propre version (`OUVRAGES_VERSION`), dans le nom
  de son fichier de cache : la changer ne reconstruit aucune scène.
- Deux lieux d'exemple : le Pont du Gard et le parc de stockage de la
  raffinerie de Feyzin.
- `outils/mesure_constructions.py`, pour refaire les mesures qui ont fixé les
  seuils.

### Modifié

- La construction d'une scène compte 18 étapes au lieu de 16.
- Le cache sert les couches chargées après la scène — monuments OSM, ouvrages
  — par un même chemin, avec un bassin de fils par source.
- Extrait OpenStreetMap des lieux d'exemple rafraîchi au 30 septembre : le
  Mont-Saint-Michel y passe de 37 à 31 parties, OSM ayant retiré la Porte du
  Roi et cinq parties sans nom. Les règles de remplacement tiennent.

### Limites connues

- Un pont n'est que son tablier : la BD TOPO ne décrit ni piles ni arches.
- Éoliennes, croix, calvaires et murs de soutènement ne sont pas dessinés.

## Scène v10 — 30 septembre 2026

### Modifié

- **Bâtiments coupés au bord de la scène** (`vue3d/batiments.py`). Le WFS rend
  un bâtiment entier dès qu'il touche l'emprise, mais la grille des hauteurs
  s'arrête à son bord : le château de Versailles, une emprise plus grande que
  la scène, n'était qu'une dalle sous la photo aérienne. Coupé à 1,25 m du
  bord, il est mesuré comme les autres ; sa fiche le dit.
- Les cours intérieures restent ouvertes, murs compris.
- Le lieu par défaut devient le village de Gordes.
- Le bouton « Monuments OSM » n'apparaît que si la vue en dessine des parties.

## Scène v9 — 28 septembre 2026

### Modifié

- **Les monuments OSM deviennent une couche à part** (`GET /api/monuments`),
  demandée une fois la scène affichée : OpenStreetMap répond de 0,6 s à plus
  de 100 s et tombe parfois, la scène ne l'attend plus et n'échoue plus avec
  lui.
- Extrait OSM embarqué pour les lieux d'exemple du README : ils n'attendent
  jamais Overpass.

### Corrigé

- L'orbite démarre avec la scène, et non avec la page.

## Scène v8 — 28 septembre 2026

### Ajouté

- **Toits en pans** (`vue3d/pans.py`) : là où le toit résumé s'écarte du LiDAR
  de plus de 0,7 m, des plans ajustés au modèle de hauteur, fermés par leurs
  murs. Un volume est vérifié fermé ou n'est pas publié. 87 % des toits
  proposés à Gordes, 55 % à Strasbourg, à 8-12 cm du LiDAR.
- Suivi de la construction d'une scène, étape par étape
  (`GET /api/avancement`).
- Guide du contributeur (`CONTRIBUTING.md`) et outils de mesure du prototype
  LoD2 des toitures.

### Modifié

- Le bouton « Toits mesurés » est allumé au départ.

### Corrigé

- « Ma position » recentre la scène quand on en est sorti.
- Chemin de cache absolu : en local, l'orthophoto répondait 404.

## Scène v7 — 28 septembre 2026

### Corrigé

- Profil minimal publié pour les bâtiments illisibles : une cabane sous les
  arbres retombait sur des hauteurs BD TOPO contaminées par la canopée (8,1 m
  de murs pour 3 m).
- Les hauteurs BD TOPO ne sont plus relevées du terrain médian : la pente
  était comptée deux fois.

## Scènes v1 à v6 — 27 septembre 2026

### Ajouté

- **Première version** : bâtiments BD TOPO et toits mesurés au LiDAR HD,
  houppiers segmentés arbre par arbre sur le modèle de hauteur à 0,5 m, relief
  RGE ALTI drapé de la photo aérienne, course du soleil et ombres portées.
  Cache disque par point : une scène est complète ou n'existe pas.
- v2 : anneau de relief grossier sur 2 km de côté autour de la scène.
- v3 : surface mesurée du toit, là où le toit résumé s'écarte du LiDAR.
- v4 : étendues et cours d'eau de la BD TOPO.
- v5 : lignes à haute tension et hauteur de leurs pylônes.
- v6 : monuments en 3D d'OpenStreetMap (`building:part`), là où le LiDAR
  manque — l'abbaye du Mont-Saint-Michel, flèche comprise.
- Routes en rubans sur le relief, fond Plan IGN, crédits selon les sources
  affichées, lieux à essayer.
