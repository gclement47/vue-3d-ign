// Vérifie la recherche d'un lieu de la page en exécutant son vrai code : les
// fonctions sont extraites de vue3d/static/index.html et appelées sous Node,
// sur des réponses réelles du géocodage de la Géoplateforme enregistrées dans
// outils/geocodage-exemples.json — des lieux publics seulement.
//
//   node outils/verifier-recherche.mjs
//
// Les contrôles :
// - coordonnées tapées ou collées (lireCoordonnees) : reconnues dans les deux
//   ordres et avec la virgule décimale, refusées hors de France métropolitaine
//   et en cours de frappe (« 43,5 », « 45 5 ») ;
// - suggestions (suggestionsDe) : ordre du service gardé, outre-mer écarté,
//   commune présente une seule fois, nom et détail lisibles.
//
// Pour rafraîchir les exemples, rejouer les quatre requêtes du fichier sur
// https://data.geopf.fr/geocodage/search?autocomplete=1&index=address,poi&limit=15&q=…
//
// Sort en erreur si un contrôle est manqué, après les avoir tous faits.
import fs from 'fs';
import { fileURLToPath } from 'url';

const lire = chemin => fs.readFileSync(fileURLToPath(new URL(chemin, import.meta.url)), 'utf8');
const src = lire('../vue3d/static/index.html');
const EXEMPLES = JSON.parse(lire('./geocodage-exemples.json'));
function extraire(nom) {
  const i = src.indexOf(`function ${nom}(`);
  if (i < 0) throw new Error('introuvable dans la page : ' + nom);
  let n = 0;
  for (let j = src.indexOf('{', src.indexOf(')', i)); j < src.length; j++) {
    if (src[j] === '{') n++;
    else if (src[j] === '}' && --n === 0) return src.slice(i, j + 1);
  }
}
const { lireCoordonnees, suggestionsDe } = new Function(
  `${extraire('lireCoordonnees')}\n${extraire('suggestionsDe')}\nreturn { lireCoordonnees, suggestionsDe };`)();

let tout = true;
function ok(nom, vrai, detail = '') {
  console.log(`${vrai ? 'OK ' : 'ÉCHEC'} ${nom}${detail ? ' : ' + detail : ''}`);
  tout &&= vrai;
}
const memes = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// --- Coordonnées ----------------------------------------------------------------
for (const [texte, attendu] of [
  ['43.4715, 5.4861', { lat: 43.4715, lon: 5.4861 }],
  ['43,4715 5,4861', { lat: 43.4715, lon: 5.4861 }],
  ['43,4715, 5,4861', { lat: 43.4715, lon: 5.4861 }],
  ['43.4715;5.4861', { lat: 43.4715, lon: 5.4861 }],
  ['  5.2003 43.9116 ', { lat: 43.9116, lon: 5.2003 }],          // longitude d'abord
  ['48.8530,-4.4860', { lat: 48.853, lon: -4.486 }],             // Finistère, longitude négative
  ['41.9192 8.7386', { lat: 41.9192, lon: 8.7386 }],             // Ajaccio
  ['43,5', null], ['45 5', null], ['13120', null],               // en cours de frappe
  ['40.7128, -74.0060', null],                                   // hors de France
  ['14.6413, -61.0702', null],                                   // Martinique : non couverte
  ['place du château gordes', null], ['', null],
]) {
  const lu = lireCoordonnees(texte);
  ok(`coordonnées « ${texte} »`, memes(lu, attendu), JSON.stringify(lu));
}

// --- Suggestions -----------------------------------------------------------------
const toutes = Object.values(EXEMPLES).flatMap(r => suggestionsDe(r, 6));
ok('au plus six suggestions', Object.values(EXEMPLES).every(r => suggestionsDe(r, 6).length <= 6));
ok('aucune suggestion sans nom ni position', toutes.every(s =>
  s.nom && Number.isFinite(s.lat) && Number.isFinite(s.lon)));
ok('rien hors de France métropolitaine', toutes.every(s =>
  s.lat >= 41 && s.lat <= 51.2 && s.lon >= -5.3 && s.lon <= 9.7),
  toutes.filter(s => s.lat < 41).map(s => s.nom).join(', '));

const gordes = suggestionsDe(EXEMPLES['gordes'], 6);
ok('« gordes » : la commune en tête', gordes[0].nom === 'Gordes' && gordes[0].detail === 'commune · 84220',
   JSON.stringify(gordes[0]));
ok('« gordes » : la commune une seule fois (trois fois dans la réponse)',
   gordes.filter(s => s.nom === 'Gordes' && s.detail.includes('84220')).length === 1);
ok('« gordes » : le lieu-dit homonyme de Berre-l\'Étang reste',
   gordes.some(s => s.nom === 'Gordes' && s.detail.includes("Berre-l'Étang")));

const fdf = suggestionsDe(EXEMPLES['fort de france'], 6);
ok('« fort de france » : la Martinique écartée, le fort de Colmars en tête',
   fdf.length >= 1 && fdf[0].detail.includes('Colmars'), fdf.map(s => s.nom).join(', '));

const chambord = suggestionsDe(EXEMPLES['chateau de chambord'], 6);
ok('« chateau de chambord » : le château en tête', chambord[0].nom === 'Château de Chambord'
   && chambord[0].detail === 'château · Chambord · 41250', JSON.stringify(chambord[0]));

const place = suggestionsDe(EXEMPLES['place du chateau gordes'], 6);
ok('« place du chateau gordes » : la place, avec code postal et commune',
   place[0].nom === 'Place du Château' && place[0].detail === '84220 Gordes'
   && Math.abs(place[0].lat - 43.9112) < 0.001 && Math.abs(place[0].lon - 5.1997) < 0.001,
   JSON.stringify(place[0]));

ok('réponse vide ou mal formée : aucune suggestion',
   [null, {}, { features: null }, { features: [{}] }].every(r => suggestionsDe(r, 6).length === 0));

process.exit(tout ? 0 : 1);
