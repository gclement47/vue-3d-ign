// Essai de la page dans un vrai navigateur : charge un point, attend la scène,
// survole et clique le bâtiment visé, bascule au 21 décembre, et relève toute
// erreur JavaScript ou requête en échec. C'est le seul moyen de vérifier les
// chemins d'exécution du rendu, que les tests Python ne voient pas.
//
//   npm install puppeteer-core
//   node outils/essai-navigateur.mjs "http://localhost:8080/?lat=43.9116&lon=5.2003" capture.png
//
// CHROME désigne l'exécutable de Chrome ou Chromium ; par défaut celui de macOS.
import puppeteer from 'puppeteer-core';
const url = process.argv[2] || 'http://localhost:8080/';
const sortie = process.argv[3] || 'capture.png';
const navigateur = await puppeteer.launch({
  executablePath: process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const page = await navigateur.newPage();
await page.setViewport({ width: 1400, height: 850 });
const erreurs = [];
page.on('pageerror', e => erreurs.push('pageerror: ' + e.message));
page.on('console', m => { if (m.type() === 'error') erreurs.push('console: ' + m.text()); });
page.on('requestfailed', r => erreurs.push('échec requête: ' + r.url().slice(0, 100) + ' ' + (r.failure()?.errorText || '')));
page.on('response', r => { if (r.status() >= 400) erreurs.push(`HTTP ${r.status()} ${r.url().slice(0, 100)}`); });
await page.goto(url, { waitUntil: 'domcontentloaded' });
// Attend la scène, puis la végétation et le relief.
await page.waitForFunction(() => document.getElementById('attente').hidden
                          || /indisponible/.test(document.getElementById('attente').textContent),
                          { timeout: 120000 });
await new Promise(r => setTimeout(r, 12000));
const etat = await page.evaluate(() => ({
  attente: document.getElementById('attente').hidden ? '(masqué)' : document.getElementById('attente').textContent,
  batiments: document.getElementById('note').textContent,
  terrain: document.getElementById('s-src').textContent + ' · ' + document.getElementById('s-amp').textContent,
  vegetation: document.getElementById('v-note').textContent.slice(0, 160),
  soleil: document.getElementById('s-date').textContent + ' · ' + document.getElementById('s-hauteur').textContent,
}));
console.log(JSON.stringify(etat, null, 1));
// Arrête l'orbite, puis survole et clique au centre (bâtiment visé).
await page.click('#t-orbit');
// L'orbite avance d'un cran par image : là où elle s'arrête dépend de la vitesse
// du rendu, et un arbre peut alors masquer le bâtiment visé (constaté à Gordes,
// un feuillu de 9 m devant lui). La végétation est masquée pour le clic.
if (await page.$eval('#t-veg', e => e.classList.contains('on'))) await page.click('#t-veg');
const cadre = await page.$eval('#scene canvas', c => { const r = c.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
// La fiche du bâtiment visé s'ouvre d'elle-même : on la relève, puis on la
// vide pour que le clic soit réellement éprouvé.
console.log('fiche à l\'ouverture :', (await page.$eval('#fiche-batiment', e => e.textContent.trim())).slice(0, 120) || '(vide)');
console.log('note :', await page.$eval('#note-batiment', e => e.textContent));
await page.$eval('#fiche-batiment', e => { e.innerHTML = ''; });
let fiche = '';
for (const [fx, fy] of [[0.5, 0.5], [0.5, 0.45], [0.48, 0.52], [0.52, 0.5], [0.45, 0.48]]) {
  const x = cadre.x + cadre.w * fx, y = cadre.y + cadre.h * fy;
  await page.mouse.move(x, y);
  await new Promise(r => setTimeout(r, 300));
  await page.mouse.click(x, y);
  await new Promise(r => setTimeout(r, 400));
  fiche = await page.$eval('#fiche-batiment', e => e.textContent.trim());
  if (fiche) break;
}
console.log('fiche après clic :', fiche.slice(0, 200) || '(vide)');
// Curseur de saison au 21 décembre, et heure à 13 h.
await page.evaluate(() => document.querySelector('.saison-cran[data-mois="12"]').click());
await page.evaluate(() => { const h = document.getElementById('heure'); h.value = 13; h.dispatchEvent(new Event('input')); });
console.log('soleil au 21 décembre, 13 h :', await page.$eval('#s-date', e => e.textContent), '·', await page.$eval('#s-hauteur', e => e.textContent));
await new Promise(r => setTimeout(r, 800));
await page.screenshot({ path: sortie });
console.log('ERREURS :', erreurs.length ? '\n  ' + erreurs.join('\n  ') : 'aucune');
await navigateur.close();
process.exit(erreurs.length ? 1 : 0);
