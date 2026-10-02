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

## Scène v14 — 1er octobre 2026

### Ajouté

- **Les tours de refroidissement.** La BD TOPO n'en donne qu'un bâtiment
  rond et une hauteur, et le LiDAR ne voit pas leur coque : à Gardanne, la
  tour de 139 m sortait en bâtiment d'un étage « sous les arbres », avec une
  aiguille de 135 m en son centre. Un bâtiment rond de plus de 60 m que le
  LiDAR ne voit pas est désormais dessiné en coque hyperbolique, au galbe de
  convention (col à 0,58 fois le pied, aux quatre cinquièmes de la hauteur).
  Les centrales nucléaires restent absentes : la BD TOPO n'en contient rien.
- **Les détecteurs sur CoreML, hors conteneur.** Lancé sans Docker sur un
  Mac, le serveur fait tourner les réseaux des véhicules et des piscines sur
  CoreML : à Gordes, 0,8 s au lieu de 2,8 pour `rtmdet`, 5 s au
  lieu de 16 pour `yolo`, et des couches identiques à l'octet. Sur une zone
  de 1 000 m, piscines et véhicules sont prêts avec la scène (114 s) ; dans
  le conteneur, sur un autre lieu, `yolo` arrivait dix minutes après elle.
  `VUE3D_MOTEUR=processeur` s'en passe ; le conteneur, qui n'a pas CoreML, ne
  change pas.
- **Deux scripts de lancement** : `run_docker.sh` repart de zéro dans
  Docker avec les deux détecteurs ; `run_macOS_CoreML.sh` lance le
  service sans Docker, détecteurs sur CoreML, et refuse de démarrer si le
  conteneur tient déjà le port.

### Modifié

- **Toitures et houppiers se calculent dix fois plus vite.** Chaque
  bâtiment relisait la grille MNH entière, en Python, et la segmentation des
  arbres la parcourait toute à chaque passe. Sur Gordes en
  zone de 1 000 m (672 bâtiments, 19 312 houppiers), le calcul passe de 100 s
  à 8,5 s, et la scène entière, lectures comprises, de 114 s à 17 s. La
  scène est la même à l'octet : son format et son cache ne changent pas.
- **Les toitures se calculent sur tous les cœurs, du centre vers le bord.**
  Chaque bâtiment part, avec sa fenêtre des grilles, vers un bassin de
  processus créé une fois par le service (« forkserver » dans le
  conteneur, « spawn » sur macOS) ; le calcul lui-même ne fait plus ce qui
  coûtait sans rien changer — un objet shapely par cellule testée, une
  boucle Python par distance au faîtage, des scalaires numpy dans la
  croissance des pans. À Strasbourg en zone de 1 000 m (1 376 bâtiments),
  sur un Mac chargé par d'autres calculs, les toitures passent de 53 s à
  10 s sur un cœur et à 3 s sur le bassin (1,2 s au mieux) ; dans le
  conteneur, de 67 s à 3,6 s ; à Gordes en zone de 1 000 m, de 6,2 s à
  0,5 s. Servie par le réseau, la scène de Strasbourg arrive en 19 s au lieu
  de 71 s dans les mêmes conditions. Elle est la même à l'octet (horodatage
  des réponses WFS mis à part), vérifiée sur quatre lieux sur macOS et
  dans le conteneur : format et cache ne changent pas.
  `VUE3D_TOITS_PROCESSUS` règle le nombre de processus (1 : aucun bassin) ;
  s'il ne peut pas démarrer ou casse en route, les toitures se calculent
  dans le service, comme avant, et ses processus s'arrêtent avec lui.

### Corrigé

- **Une cheminée posée sur un bâtiment n'est plus effacée.** Un point de la
  BD TOPO dans une emprise bâtie était laissé au toit du bâtiment, quelle que
  soit sa hauteur : la cheminée de 295 m de la centrale de Provence, à
  Gardanne, disparaissait dans son socle de 32,8 m. Elle n'est plus écartée
  que si elle ne dépasse pas le bâtiment (de 15 %) ou si la hauteur du
  bâtiment est inconnue. Sur 17 points en bâtiment de hauteur déclarée autour
  de quatorze sites industriels, un seul tenait vraiment sous son toit.
- **Les très hautes cheminées gardent leur largeur.** Le LiDAR en perd le
  sommet (Gardanne culmine à 155 m dans le MNH, Porcheville à 92 m pour
  220 m) : leur rayon mesuré était jeté, et la page les dessinait en aiguille
  de 4 m de rayon au plus. Le fût, isolé, donne désormais sa largeur —
  environ 9 m de rayon à Gardanne.

## Scène v13 — 30 septembre 2026

### Ajouté

- **Une zone plus grande, au choix.** Le sélecteur « Zone » du panneau, ou
  `zone=` dans l'URL, fixe le côté nord-sud de la scène, de 150 à 1 000 m
  (arrondi à 50 m). L'emprise par défaut (~356 m) ne change pas et garde son
  cache. Autour d'un site de la vallée de la chimie à Lyon, 1 000 m donnent 301
  bâtiments, 17 réservoirs et 4 constructions élevées contre 50, 7 et 2 ; la
  construction prend environ cinq minutes. L'anneau de relief, le brouillard,
  le recul de la caméra et les ombres suivent la taille de la zone.

### Corrigé

- **Une lecture WFS n'est plus tronquée à 5 000 objets.** Le service s'arrête
  là sans erreur (5 000 bâtiments rendus sur 9 948 au centre de Paris).
  L'emprise est désormais coupée en quatre jusqu'à ce que chaque morceau
  tienne en une réponse.
- **Les véhicules d'une zone élargie sont détectés à 0,2 m.** L'orthophoto
  des détections était plafonnée à 2 048 px : à 1 000 m elle revenait à
  0,49 m, les voitures y étaient 2,5 fois trop petites et presque toutes
  écartées (une dizaine). Lue en tuiles, elle garde sa résolution : 301
  véhicules au même endroit.
- **La mosaïque d'orthophoto garde ses proportions au-delà de 2 048 px.** Ses
  deux côtés étaient plafonnés chacun de son côté, ce qui l'aurait étirée.

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
- **Chaque détection s'affiche dès qu'elle est prête.** La page attendait
  la couche entière — véhicules des deux détecteurs et piscines, jusqu'à
  quarante secondes dans le conteneur avec `tous` — avant de rien montrer.
  Les piscines ont maintenant leur fichier et leur route (`/api/piscines`),
  chaque détecteur de véhicules le sien (`/api/vehicules?detecteur=…`), et
  la page les demande l'un après l'autre, du rapide au lent, en dessinant
  chaque couche à son arrivée ; `/api/sante` lui dit les détecteurs du
  service. La couche des véhicules passe en version 3.
- **La couche des véhicules et des piscines n'est plus gardée un jour par le
  navigateur.** À la même adresse, elle change avec le détecteur du service
  et avec sa version : après une reconstruction de l'image, un lieu déjà
  visité montrait encore ses véhicules sans ses piscines. Elle est
  maintenant revalidée à chaque demande, le serveur répondant 304 tant que
  rien n'a changé.

### Ajouté

- **Un lien « Recentrer la scène sur ce bâtiment »** : la scène est
  toujours cadrée sur son point, et un bâtiment coupé par le bord se lit
  mieux au centre de la sienne. Un clic sur un bâtiment épingle son
  infobulle, qui ne suit plus la souris et porte ce lien et celui de Street
  View, jusqu'au prochain clic sur la scène ; la fiche du panneau les a
  aussi.
- **Les véhicules de l'orthophoto, en option.** Lancé avec
  `VUE3D_VEHICULES=rtmdet`, `yolo` ou `tous`, le service lit les véhicules
  sur l'orthophoto à 0,2 m avec un réseau à boîtes orientées, et la vue les
  pose en volume, à leur couleur, sur la pente. Une couche à part
  (`/api/vehicules`), calculée après la scène et gardée dans un fichier au nom
  du détecteur ; la scène ne change pas, son cache non plus. Par défaut
  (`aucun`), rien ne change : ni dépendance, ni poids, même image. Mesuré sur
  Gordes et Carcassonne : 91 et 178 véhicules pour `rtmdet` en 3 à 6 s, 146
  et 140 pour `yolo` en 14 à 35 s, 169 et 188 pour les deux ensemble. Le
  LiDAR, lui, ne voit pas les véhicules (0,0 m sur un parking plein), et
  l'analyse d'image classique en retrouvait 8 sur 61.
- **Les panneaux solaires, en option** (`VUE3D_PANNEAUX=oui`). Pas un
  réseau de plus : le registre OpenPVMapper (G. Kasmi, CC-BY 4.0), où
  DeepPVMapper a relevé 471 449 installations en toiture sur la BD ORTHO de
  toute la France, est téléchargé à la construction de l'image et rangé
  dans une base SQLite à index spatial (119 Mo). Chaque installation est
  posée sur le toit tel que la vue le dessine, avec sa surface, sa puissance
  et l'année de la photo. Les poids publiés du réseau lui-même ont été
  essayés d'abord : 79 % d'exactitude sur leur propre jeu de test, rien de
  trouvé sur Gordes ni Carcassonne — on lit le résultat des auteurs.
- **Les piscines avec.** La même option lit aussi les piscines de
  l'orthophoto — la BD TOPO n'a pas celles des particuliers — et la vue les
  pose en bassins, à la couleur de leur eau : 13 à Gordes et 9 à Carcassonne
  pour `rtmdet`, 8 et 8 pour `yolo`, 14 et 9 pour les deux, en une
  demi-seconde de plus. Des 21 taches bleues des deux orthophotos, toutes
  des piscines, `rtmdet` en couvre 20. La couche passe en version 2 : elle
  est recalculée à la première ouverture de chaque lieu, les scènes non.
- `outils/exporter_vehicules.py` convertit les réseaux en ONNX à la
  construction de l'image — leurs poids ne sont pas dans le dépôt —, et
  `outils/mesure_vehicules.py` rejoue la mesure sur des lieux réels.
- `outils/verifier-geometrie.mjs` : exécute sous Node les fonctions
  géométriques de la page et vérifie orientation, fermeture et volumes ; il
  couvre aussi les volumes des véhicules et leur pose sur une pente.

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
