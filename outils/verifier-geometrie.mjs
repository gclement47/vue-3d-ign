// Vérifie la géométrie de la page en exécutant son vrai code : les fonctions
// sont extraites de vue3d/static/index.html et appelées sous Node, avec le
// three.js de la page. La relecture ne suffit pas — c'est ainsi qu'ont été
// trouvées des facettes retournées, puis un pignon sur deux manquant au toit
// découpé (un côté du contour posé sur le bord de sa boîte, jeté dehors par
// l'arrondi).
//
//   npm install three@0.160.0          # ignoré par git, comme puppeteer-core
//   node outils/verifier-geometrie.mjs
//
// Deux familles de contrôles :
// - volumes des ouvrages (prismeLeLong, dalle) : fermés — chaque arête portée
//   par deux triangles, en sens opposés — et de volume positif, donc orientés
//   vers l'extérieur ;
// - toit résumé découpé sur l'emprise (geometrieToitDecoupe) : pans tournés
//   vers le haut, d'aire égale à celle de l'emprise, pignons tournés vers
//   l'extérieur, volume égal à l'intégrale de la hauteur du toit ;
// - véhicules (blocsVehicule, repereVehicule) : chaque bloc fermé, tourné vers
//   l'extérieur, dans le gabarit unité ; le repère d'un véhicule posé sur une
//   pente reste orthonormé, direct, le toit vers le haut.
//
// Sort en erreur au premier contrôle manqué.
import * as THREE from 'three';
import fs from 'fs';
import { fileURLToPath } from 'url';

const page = fileURLToPath(new URL('../vue3d/static/index.html', import.meta.url));
const src = fs.readFileSync(page, 'utf8');
function extraire(nom) {
  const i = src.indexOf(`function ${nom}(`);
  if (i < 0) throw new Error('introuvable dans la page : ' + nom);
  let n = 0;
  for (let j = src.indexOf('{', src.indexOf(')', i)); j < src.length; j++) {
    if (src[j] === '{') n++;
    else if (src[j] === '}' && --n === 0) return src.slice(i, j + 1);
  }
}
// Projection locale de la page, autour d'un point à 45° N.
const M = 111320, MLON = 111320 * Math.cos(45 * Math.PI / 180);
const toLocal = (lon, lat) => [(lon - 2) * MLON, (lat - 45) * M];
const NOMS = ['couperPolygone', 'enveloppeConvexe', 'rectangleMin', 'rectangleSelonAxe',
              'geometrieToitDecoupe', 'stationsLeLong', 'prismeLeLong', 'dalle',
              'blocsVehicule', 'repereVehicule'];
const { rectangleMin, rectangleSelonAxe, geometrieToitDecoupe, stationsLeLong, prismeLeLong, dalle,
        blocsVehicule, repereVehicule } =
  new Function('THREE', 'toLocal', NOMS.map(extraire).join('\n') + `\nreturn { ${NOMS.join(', ')} };`)(THREE, toLocal);
let tout = true;

// --- Volumes des ouvrages ---------------------------------------------------
{
const ll3 = (x, y, z) => [2 + x / MLON, 45 + y / M, z];

// Hauteur d'un sommet { z, dz }, sol plat à 100 m : comme maillagePose, plancher 0.
const SOL = 100;
const pos = s => [s.x, (s.z == null ? 0 : Math.max(s.z - SOL, 0)) + s.dz, -s.y];
function verifierVolume(nom, tris) {
  const cle = p => p.map(v => v.toFixed(4)).join(',');
  const aretes = new Map();
  let volume = 0, degeneres = 0;
  for (const t of tris) {
    const [a, b, c] = t.map(pos);
    const n = [(b[1]-a[1])*(c[2]-a[2]) - (b[2]-a[2])*(c[1]-a[1]), (b[2]-a[2])*(c[0]-a[0]) - (b[0]-a[0])*(c[2]-a[2]), (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])];
    if (Math.hypot(...n) < 1e-9) { degeneres++; }
    volume += (a[0] * (b[1]*c[2] - b[2]*c[1]) - a[1] * (b[0]*c[2] - b[2]*c[0]) + a[2] * (b[0]*c[1] - b[1]*c[0])) / 6;
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      const k = cle(u) + '>' + cle(v);
      aretes.set(k, (aretes.get(k) || 0) + 1);
    }
  }
  let ouvertes = 0, doubles = 0;
  for (const [k, n] of aretes) {
    const [u, v] = k.split('>');
    if (u === v) continue;               // arête d'un triangle dégénéré
    if (n !== 1) doubles++;
    if ((aretes.get(v + '>' + u) || 0) !== 1) ouvertes++;
  }
  const ok = ouvertes === 0 && doubles === 0 && volume > 0;
  console.log(`${ok ? 'OK ' : 'KO '} ${nom} : ${tris.length} triangles, volume ${volume.toFixed(1)} m³, arêtes sans vis-à-vis ${ouvertes}, arêtes doublées ${doubles}, triangles dégénérés ${degeneres}`);
  return ok;
}
const haut = dz => s => ({ z: s.z, dz }), sol = () => ({ z: null, dz: -0.5 });
// Mur droit, mur coudé, mur en épingle, mur à sommet doublé.
const lignes = {
  'mur droit': [ll3(0, 0, 110), ll3(30, 0, 112)],
  'mur coudé': [ll3(0, 0, 110), ll3(20, 0, 110), ll3(20, 15, 115), ll3(40, 30, 108)],
  'mur en épingle': [ll3(0, 0, 110), ll3(20, 0, 110), ll3(0, 1.5, 110)],
  'mur à sommet doublé': [ll3(0, 0, 110), ll3(10, 0, 110), ll3(10, 0, 110), ll3(10, 12, 111)],
  'mur vers le sud-ouest': [ll3(30, 30, 108), ll3(0, 0, 108)],
};
for (const [nom, l] of Object.entries(lignes)) {
  const st = stationsLeLong(l, 2);
  tout = verifierVolume(nom, prismeLeLong(st, 1, haut(0), sol)) && tout;
  tout = verifierVolume(nom + ' (tablier)', prismeLeLong(st, 4, haut(0), haut(-1))) && tout;
}
// Dalles : contour dans les deux sens, avec et sans trou, fermé par son premier point.
const carre = [ll3(0, 0, 105), ll3(40, 0, 105), ll3(40, 20, 107), ll3(0, 20, 107), ll3(0, 0, 105)];
const trou = [ll3(10, 5, 105), ll3(20, 5, 105), ll3(20, 12, 106), ll3(10, 12, 106), ll3(10, 5, 105)];
const enL = [ll3(0, 0, 105), ll3(30, 0, 105), ll3(30, 10, 105), ll3(10, 10, 106), ll3(10, 30, 107), ll3(0, 30, 107), ll3(0, 0, 105)];
tout = verifierVolume('dalle, sens trigonométrique', dalle(carre, [], 1)) && tout;
tout = verifierVolume('dalle, sens des aiguilles', dalle(carre.slice().reverse(), [], 1)) && tout;
tout = verifierVolume('dalle trouée', dalle(carre, [trou], 1)) && tout;
tout = verifierVolume('dalle trouée, trou dans l\'autre sens', dalle(carre, [trou.slice().reverse()], 1)) && tout;
tout = verifierVolume('dalle en L', dalle(enL, [], 1)) && tout;
tout = verifierVolume('dalle en L, sens des aiguilles', dalle(enL.slice().reverse(), [], 1)) && tout;
}

// --- Toit résumé découpé sur l'emprise ----------------------------------------
{
const ll = ([x, y]) => [2 + x / MLON, 45 + y / M];
const fermer = a => a.concat([a[0]]);
const tourner = (pts, deg) => { const c = Math.cos(deg * Math.PI / 180), s = Math.sin(deg * Math.PI / 180); return pts.map(([x, y]) => [x * c - y * s, x * s + y * c]); };
const aire = a => { let s = 0; for (let k = 0; k < a.length; k++) { const p = a[k], q = a[(k + 1) % a.length]; s += p[0] * q[1] - q[0] * p[1]; } return Math.abs(s / 2); };
const dans = (p, a) => { let d = false; for (let i = 0, j = a.length - 1; i < a.length; j = i++) { const [xi, yi] = a[i], [xj, yj] = a[j]; if ((yi > p[1]) !== (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi) d = !d; } return d; };

function verifier(nom, contour, trous, boitesDe, hMur) {
  const pts = contour, rects = boitesDe(pts);
  const geo = geometrieToitDecoupe(fermer(contour).map(ll), trous.map(t => fermer(t).map(ll)), rects, hMur);
  if (!geo) { console.log('KO ', nom, ': aucune géométrie'); return false; }
  const pos = geo.attributes.position.array;
  let aireHaut = 0, volume = 0, versBas = 0, pignonsDedans = 0, hMax = -Infinity, hMin = Infinity, nPignons = 0;
  // centre de gravité grossier de l'emprise, pour juger du sens des pignons d'une emprise convexe
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  for (let i = 0; i < pos.length; i += 9) {
    const a = [pos[i], pos[i + 1], pos[i + 2]], b = [pos[i + 3], pos[i + 4], pos[i + 5]], c = [pos[i + 6], pos[i + 7], pos[i + 8]];
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    const l = Math.hypot(...n);
    volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
    if (Math.abs(n[1]) > 1e-6 * l) {                 // un pan
      if (n[1] < 0) versBas++;
      aireHaut += n[1] / 2;
      for (const p of [a, b, c]) { hMax = Math.max(hMax, p[1]); hMin = Math.min(hMin, p[1]); }
    } else {                                         // un pignon, vertical
      nPignons++;
      if (l < 1e-9) continue;                        // triangle aplati au bas d'un pignon
      // Un pas vers où regarde le pignon : on doit sortir du bâtiment (ou entrer dans une cour).
      const m = [(a[0] + b[0] + c[0]) / 3 + 0.05 * n[0] / l, -(a[2] + b[2] + c[2]) / 3 - 0.05 * n[2] / l];
      if (dans(m, pts) && !trous.some(t => dans(m, t))) pignonsDedans++;
    }
  }
  // Volume attendu : intégrale de (toit − hMur) sur l'emprise, par échantillonnage ; le toit d'un
  // point est le plus haut des boîtes qui le couvrent, comme des pans qui se traversent.
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const [xa, xb, ya, yb] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const N = 500; let attendu = 0, couvert = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const p = [xa + (xb - xa) * (i + 0.5) / N, ya + (yb - ya) * (j + 0.5) / N];
    if (!dans(p, pts) || trous.some(t => dans(p, t))) continue;
    let somme = 0, un = false;
    for (const b of rects) {
      const { ang, x0, x1, y0, y1 } = b.rect, co = Math.cos(ang), si = Math.sin(ang);
      const q = [p[0] * co + p[1] * si, -p[0] * si + p[1] * co];
      if (q[0] < x0 || q[0] > x1 || q[1] < y0 || q[1] > y1) continue;
      const mu = (x0 + x1) / 2, mw = (y0 + y1) / 2, du = (x1 - x0) / 2, dw = (y1 - y0) / 2;
      const sx = b.suivantX ?? (x1 - x0 >= y1 - y0);
      const h = b.pyramide ? b.g + (b.f - b.g) * (1 - Math.max(Math.abs(q[0] - mu) / du, Math.abs(q[1] - mw) / dw))
              : sx ? b.g + (b.f - b.g) * (1 - Math.abs(q[1] - mw) / dw) : b.g + (b.f - b.g) * (1 - Math.abs(q[0] - mu) / du);
      somme += h - hMur; un = true;
    }
    if (un) { couvert++; attendu += somme; }
  }
  const cellule = (xb - xa) * (yb - ya) / (N * N);
  attendu *= cellule; const aireAttendue = couvert * cellule * (rects.length === 1 ? 1 : NaN);
  // Le volume des seuls pans et pignons, ouvert par-dessous : on le ferme par la base à hMur.
  const base = rects.length === 1 ? hMur * couvert * cellule : NaN;
  const okAire = rects.length > 1 || Math.abs(aireHaut - aireAttendue) < 0.01 * aireAttendue;
  // La base, tournée vers le bas à la hauteur hMur, compte pour −hMur × aire / 3 dans la somme.
  const volumeFerme = volume - (rects.length === 1 ? base / 3 : 0);
  const okVolume = rects.length > 1 || Math.abs(volumeFerme - attendu) < 0.02 * attendu + 0.5;
  const ok = versBas === 0 && pignonsDedans === 0 && okAire && okVolume;
  console.log(`${ok ? 'OK ' : 'KO '} ${nom} : ${pos.length / 9} triangles ; pans vers le bas ${versBas} ; aire des pans ${aireHaut.toFixed(1)} m² (emprise sous les boîtes ${rects.length === 1 ? aireAttendue.toFixed(1) : '—'}) ; hauteurs ${hMin.toFixed(2)} à ${hMax.toFixed(2)} ; pignons ${nPignons} dont vers l'intérieur ${pignonsDedans} ; volume ${rects.length === 1 ? volumeFerme.toFixed(1) + ' m³ pour ' + attendu.toFixed(1) : '—'}`);
  return ok;
}
const rect = [[0, 0], [20, 0], [20, 10], [0, 10]];
const enL = [[0, 0], [20, 0], [20, 8], [8, 8], [8, 18], [0, 18]];
const cercle = Array.from({ length: 24 }, (_, k) => [6 * Math.cos(k * Math.PI / 12), 6 * Math.sin(k * Math.PI / 12)]);
const cour = [[0, 0], [30, 0], [30, 24], [0, 24]], trou = [[10, 8], [20, 8], [20, 16], [10, 16]];
const min = extra => pts => [{ rect: rectangleMin(pts), g: 4, f: 7, ...extra }];
const axe = (deg, extra) => pts => [{ rect: rectangleSelonAxe(pts, deg * Math.PI / 180), g: 4, f: 7, suivantX: true, ...extra }];
for (const sens of ['', ' (sens des aiguilles)']) {
  const o = a => sens ? a.slice().reverse() : a;
  tout = verifier('rectangle, deux pans' + sens, o(rect), [], min({}), 4) && tout;
  tout = verifier('rectangle tourné de 30°' + sens, o(tourner(rect, 30)), [], min({}), 4) && tout;
  tout = verifier('rectangle, faîtage mesuré en biais à 25°' + sens, o(rect), [], axe(25), 4) && tout;
  tout = verifier('maison en L' + sens, o(enL), [], min({}), 4) && tout;
  tout = verifier('maison en L, faîtage à 40°' + sens, o(tourner(enL, 10)), [], axe(40), 4) && tout;
  tout = verifier('tour ronde, pyramide' + sens, o(cercle), [], min({ pyramide: true }), 4) && tout;
  tout = verifier('rectangle, pyramide' + sens, o(rect), [], min({ pyramide: true }), 4) && tout;
  tout = verifier('îlot à cour' + sens, o(cour), [o(trou)], min({}), 4) && tout;
  tout = verifier('faîtage le long du petit côté' + sens, o(rect), [], pts => [{ rect: rectangleMin(pts), g: 4, f: 7, suivantX: false }], 4) && tout;
  tout = verifier('murs plus bas que la gouttière' + sens, o(rect), [], min({}), 3) && tout;
}
// Deux corps sur une maison en L : chacun sa boîte, pans vers le haut, pignons dehors (emprise non convexe : sens non jugé).
const corps = [{ rect: { ang: 0, x0: 0, x1: 20, y0: 0, y1: 8 }, g: 4, f: 7, suivantX: true },
               { rect: { ang: Math.PI / 2, x0: 8, x1: 18, y0: -8, y1: 0 }, g: 4.5, f: 6.5, pyramide: true }];
const g2 = geometrieToitDecoupe(fermer(enL).map(ll), [], corps, 4);
let bas = 0; const p2 = g2.attributes.position.array;
for (let i = 0; i < p2.length; i += 9) { const ny = (p2[i + 5] - p2[i + 2]) * (p2[i + 6] - p2[i]) - (p2[i + 3] - p2[i]) * (p2[i + 8] - p2[i + 2]); if (ny < -1e-9) bas++; }
console.log(`${bas === 0 ? 'OK ' : 'KO '} deux corps sur une maison en L : ${p2.length / 9} triangles, pans vers le bas ${bas}`);
// Une boîte hors de l'emprise ne donne rien.
const rien = geometrieToitDecoupe(fermer(rect).map(ll), [], [{ rect: { ang: 0, x0: 100, x1: 120, y0: 0, y1: 10 }, g: 4, f: 7 }], 4);
console.log(`${rien === null ? 'OK ' : 'KO '} boîte hors de l'emprise : ${rien === null ? 'null' : 'géométrie'}`);
tout = tout && bas === 0 && rien === null;
}

// --- Véhicules ------------------------------------------------------------------
{
for (const gabarit of ['voiture', 'fourgon', 'car']) {
  const blocs = blocsVehicule(gabarit);
  let ouvertes = 0, dehors = 0, retournes = 0, volume = 0;
  for (const b of blocs) {
    const aretes = new Map();
    let v = 0;
    const cle = p => p.map(c => c.toFixed(5)).join(',');
    for (const [a, bb, c] of b.tris) {
      v += (a[0] * (bb[1] * c[2] - bb[2] * c[1]) - a[1] * (bb[0] * c[2] - bb[2] * c[0]) + a[2] * (bb[0] * c[1] - bb[1] * c[0])) / 6;
      for (const [u, w] of [[a, bb], [bb, c], [c, a]]) aretes.set(cle(u) + '>' + cle(w), (aretes.get(cle(u) + '>' + cle(w)) || 0) + 1);
      for (const p of [a, bb, c]) if (Math.abs(p[0]) > 0.5 + 1e-9 || p[1] < -1e-9 || p[1] > 1 + 1e-9 || Math.abs(p[2]) > 0.52) dehors++;
    }
    for (const [k, n] of aretes) { const [u, w] = k.split('>'); if (n !== 1 || (aretes.get(w + '>' + u) || 0) !== 1) ouvertes++; }
    if (v <= 0) retournes++;
    volume += v;
  }
  const teintes = new Set(blocs.map(b => b.teinte));
  const ok = ouvertes === 0 && dehors === 0 && retournes === 0 && teintes.has('caisse') && teintes.has('vitre') && teintes.has('roue');
  console.log(`${ok ? 'OK ' : 'KO '} véhicule « ${gabarit} » : ${blocs.length} blocs, volume ${volume.toFixed(2)} du gabarit, arêtes sans vis-à-vis ${ouvertes}, blocs retournés ${retournes}, sommets hors gabarit ${dehors}`);
  tout = tout && ok;
}
// Repère : caps tous les 15°, pentes jusqu'à 100 % en long et en travers.
let mauvais = 0, essais = 0, pire = 0;
const scal = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
for (let cap = 0; cap < 360; cap += 15) for (const pl of [-1, -0.2, 0, 0.2, 1]) for (const pt of [-1, -0.2, 0, 0.2, 1]) {
  const est = Math.sin(cap * Math.PI / 180), nord = Math.cos(cap * Math.PI / 180), L = 4.3, W = 1.8;
  const { X, Y, Z } = repereVehicule(est, nord, L, W, pl * L / 2, -pl * L / 2, -pt * W / 2, pt * W / 2);
  const det = scal(X, [Y[1] * Z[2] - Y[2] * Z[1], Y[2] * Z[0] - Y[0] * Z[2], Y[0] * Z[1] - Y[1] * Z[0]]);
  const ecart = Math.max(Math.abs(scal(X, X) - 1), Math.abs(scal(Y, Y) - 1), Math.abs(scal(Z, Z) - 1), Math.abs(scal(X, Y)), Math.abs(scal(Y, Z)), Math.abs(det - 1));
  // L'avant pointe vers le cap et monte avec la pente ; la droite descend si le sol descend à droite.
  const sens = X[0] * est - X[2] * nord > 0 && Math.sign(X[1]) === Math.sign(pl) && Z[0] * nord + Z[2] * est > 0;
  essais++; pire = Math.max(pire, ecart);
  if (ecart > 1e-9 || Y[1] <= 0 || !sens) mauvais++;
}
const plat = repereVehicule(0, 1, 4.3, 1.8, 0, 0, 0, 0);       // cap nord, sol plat
const platOk = [plat.X, plat.Y, plat.Z].flat().every((c, i) => Math.abs(c - [0, 0, -1, 0, 1, 0, 1, 0, 0][i]) < 1e-12);
console.log(`${mauvais === 0 && platOk ? 'OK ' : 'KO '} repère d'un véhicule : ${essais} poses, ${mauvais} fausses, écart à l'orthonormé ${pire.toExponential(1)}, à plat cap nord ${platOk ? 'avant au nord, droite à l\'est' : 'FAUX'}`);
tout = tout && mauvais === 0 && platOk;
}

process.exit(tout ? 0 : 1);
