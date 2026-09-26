# Vue 3D IGN

Une vue 3D de n'importe quel lieu de France métropolitaine, reconstruite à partir
des **données ouvertes de l'IGN** : on donne une latitude et une longitude, on
obtient les bâtiments avec leurs vrais toits, les arbres un par un, le relief et
la photo aérienne, sous un soleil qui suit sa vraie course.

![Le village de Gordes, au 21 décembre à 13 h](docs/capture-gordes.jpg)

*Gordes (Vaucluse) au 21 décembre à 13 h : 249 bâtiments, 2 160 houppiers,
84 m de relief. Orthophoto et données © IGN.*

Aucune clé d'API, aucune base de données : un conteneur Docker, un cache disque,
et les services publics de la [Géoplateforme](https://geoservices.ign.fr/).

## Démarrer

```bash
docker compose up -d
```

Puis ouvrir <http://localhost:8080/?lat=43.9116&lon=5.2003>, ou saisir un point
dans le panneau. Sans paramètre, la page s'ouvre sur la cour du château de
Versailles.

La **première** ouverture d'un lieu construit sa scène : 20 à 40 secondes, le
temps de télécharger une grille de hauteurs à 0,5 m et d'y segmenter les arbres.
Les ouvertures suivantes sont instantanées, la scène étant gardée sur disque dans
le volume `scenes`. Pour changer de port : `VUE3D_PORT=9000 docker compose up -d`.

Sans Docker :

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
VUE3D_CACHE=./cache flask --app vue3d.app run --port 8080
```

## Ce que montre la vue

- **Les bâtiments et leurs toits.** Les murs montent à la gouttière et le toit
  rejoint le faîtage, **mesurés au LiDAR HD** quand la mesure est fiable, sinon
  déclarés par la BD TOPO. Un bâtiment en ailes reçoit un toit par corps, chacun
  sur son propre faîtage. Le bâtiment qui contient le point visé est en orange ;
  un clic ouvre sa fiche BD TOPO et un lien Street View orienté depuis la rue.
- **Les arbres, un par un.** La végétation est segmentée houppier par houppier
  sur le modèle de hauteur à 0,5 m. Chaque arbre garde sa hauteur, son emprise,
  son allongement et son profil mesurés : un pin parasol a un sommet plat, un
  cyprès une pointe.
- **Le relief**, drapé de la photo aérienne, avec une exagération réglable.
- **Le soleil**, à l'heure et au jour choisis : un curseur pour l'heure, un
  autre pour la saison, avec des crans aux solstices et à l'équinoxe. Les ombres
  portées suivent. Le panneau chiffre le **masque solaire au sud**, l'élévation
  du plus haut obstacle vu depuis la toiture du bâtiment visé.

## Les sources

Toutes servies sans clé par la Géoplateforme de l'IGN, sous
[Licence Ouverte Etalab 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).

| Donnée | Service | Ce qu'elle apporte |
|---|---|---|
| BD TOPO, bâtiments | WFS `BDTOPO_V3:batiment` | Emprises, usage, hauteurs et altitudes de toit déclarées |
| BD TOPO, végétation | WFS `BDTOPO_V3:zone_de_vegetation` | Où est la végétation (haie, bois, forêt…), jamais sa hauteur |
| BD Forêt v2 | WFS `LANDCOVER.FORESTINVENTORY.V2:formation_vegetale` | Essence dominante d'un massif, pour la couleur des arbres |
| BD TOPO, routes | WFS `BDTOPO_V3:troncon_de_route` | Point de vue Street View posé sur la rue |
| LiDAR HD, MNH | WMS, grille BIL à 0,5 m | Hauteur de tout ce qui dépasse du sol : toits et arbres |
| MNS − MNT | WMS, repli photogrammétrique | Le même, hors couverture LiDAR HD, en moins net |
| RGE ALTI | WMS, grille BIL | Le relief du terrain |
| Orthophoto | WMS | La photo aérienne, et l'indice de verdure qui reconnaît le feuillage |

three.js est chargé depuis jsDelivr. Le lien Street View ouvre Google Maps.

## La méthode, en bref

Chaque règle a été fixée après une mesure sur des données réelles. Les
principales :

- **Toitures.** Gouttière au 15e centile et faîtage au 85e des hauteurs LiDAR
  de l'emprise érodée, plutôt que le minimum et le maximum : un arbre qui
  surplombe gonfle le maximum (16 m lus sur une maison de 4 m). La mesure est
  rejetée si le haut du profil n'est pas un toit, et une toiture à moitié sous
  un arbre est relue dans le mode bas de son profil. Les altitudes de toit BD
  TOPO manquent sur près d'un tiers des bâtiments d'un site mesuré, et sont
  bruitées dans les deux sens ailleurs.
- **Bâtiments sous les arbres.** Le LiDAR y lit la canopée : une annexe de
  12 m² sous les feuillages lisait un toit plat à 12 m qui passait tous les tests
  de forme. L'orthophoto tranche : au-delà d'un tiers de l'emprise verte, la
  hauteur est déclarée inconnue, et le bâtiment est dessiné pâle, sans toit.
- **Végétation.** Une cellule du modèle de hauteur est de la végétation si elle
  tombe dans une zone BD TOPO, ou si l'orthophoto y est verte (indice
  ExG = 2G − R − B, seuil 4 : 93 % de la végétation reconnue pour 2 % de toits
  pris pour du vert). Les houppiers sont les bassins de la grille lissée,
  descendus depuis leurs sommets. Leur emprise au sol est une ellipse d'aire et
  d'allongement mesurés, inscrite à 80 % dans son segment pour laisser voir les
  trouées.
- **Port des arbres.** L'essence BD Forêt ne décrit qu'un massif : quand elle
  dit « mixte », c'est le profil mesuré de l'arbre qui choisit son port — un
  sommet qui se maintient loin du centre est un pin, une cime effilée un
  conifère, un dôme un feuillu.
- **Une scène est complète ou n'existe pas.** Le cache ne périme pas, les
  données sources ne changeant qu'au rythme des campagnes IGN. Une scène
  construite pendant une panne resterait donc fausse pour toujours : si une
  seule source ne répond pas, rien n'est mis en cache et le serveur répond 503.
  Les lectures réessaient trois fois, refus compris, car le service renvoie des
  400 sporadiques sur des requêtes valides.

## Limites

- **France métropolitaine seulement.** Hors de cette emprise, le serveur refuse
  le point.
- **Couverture LiDAR HD en cours.** Environ 77 % des bâtiments tirés au hasard
  sont couverts ; ailleurs, le repli photogrammétrique lisse les cimes et les
  faîtages, et le panneau le signale.
- **Une scène couvre environ 356 m de côté** autour du point.
- **La règle « sous les arbres » est sévère** dans les tissus denses et arborés,
  où l'orthophoto, qui n'est pas une vraie orthophoto, décale les feuillages
  sur les emprises voisines.

## API

| Route | Réponse |
|---|---|
| `GET /?lat=…&lon=…` | La page |
| `GET /api/scene?lat=…&lon=…` | La scène, en JSON gzippé |
| `GET /api/ortho?lat=…&lon=…` | L'orthophoto de la scène, en JPEG |
| `GET /api/sante` | `{"ok": true}` |

Codes d'erreur : 400 sans coordonnées valides, 422 hors de France métropolitaine,
503 si un service de l'IGN n'a pas répondu (rien n'est mis en cache, réessayer).

## Développer

```bash
pip install -r requirements.txt pytest
pytest
```

Les tests Python ne voient pas le rendu. Pour la page elle-même, un essai dans un
vrai navigateur charge un lieu, clique le bâtiment visé, change de saison et
échoue à la moindre erreur JavaScript :

```bash
npm install puppeteer-core
node outils/essai-navigateur.mjs "http://localhost:8080/?lat=43.9116&lon=5.2003" capture.png
```

## Licence

Code sous licence [MIT](LICENSE). Les données affichées appartiennent à l'IGN et
sont diffusées sous Licence Ouverte Etalab 2.0.
