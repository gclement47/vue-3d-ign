// Chronologie des requêtes de la page, relevée dans un vrai navigateur : à
// quel instant (depuis le début de la navigation) chaque requête au serveur
// part et arrive, et quand la scène est affichée. Pour voir ce que la page
// attend, et si elle l'attend en série ou en parallèle.
//
//   npm install puppeteer-core
//   node outils/chrono-page.mjs "http://localhost:8080/?lat=43.9116&lon=5.2003" [tours] [froid]
//
// `froid` : cache du navigateur vidé avant chaque tour (three.js compris) ;
// sinon le premier tour le remplit et les suivants le trouvent. ATTENTE_MS :
// temps laissé aux couches à part après l'affichage de la scène (4 s).
// La dernière ligne de chaque tour est un résumé JSON : `affichee` est le
// premier instant où la page, libre, a masqué son message d'attente — après
// la construction des maillages, que le rendu logiciel allonge beaucoup.
import puppeteer from 'puppeteer-core';

const url = process.argv[2] || 'http://localhost:8080/';
const tours = Number(process.argv[3] || 1);
const froid = process.argv[4] === 'froid';
const navigateur = await puppeteer.launch({
  executablePath: process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  protocolTimeout: 0,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const page = await navigateur.newPage();
await page.setViewport({ width: 1400, height: 850 });
// La scène se construit parfois longtemps, et le rendu logiciel occupe la
// page plusieurs secondes : pas de délai sur la navigation.
page.setDefaultNavigationTimeout(0);
page.setDefaultTimeout(0);
const cdp = await page.createCDPSession();
const erreurs = [];
page.on('pageerror', e => erreurs.push('pageerror: ' + e.message));
page.on('console', m => { if (m.type() === 'error') erreurs.push('console: ' + m.text()); });
for (let tour = 0; tour < tours; tour++) {
  if (froid) await cdp.send('Network.clearBrowserCache');
  // La page précédente quittée d'abord : occupée à construire ses maillages,
  // elle retardait le départ de la suivante, et sa chronologie avec.
  if (tour) await page.goto('about:blank');
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.getElementById('attente').hidden
                            || /indisponible/.test(document.getElementById('attente').textContent),
                            { timeout: 600000, polling: 10 });
  const affichee = await page.evaluate(() => performance.now());
  await new Promise(r => setTimeout(r, Number(process.env.ATTENTE_MS || 4000)));
  const entrees = await page.evaluate(() => performance.getEntriesByType('resource')
    .filter(e => e.name.startsWith(location.origin) || /three/.test(e.name))
    .map(e => ({ nom: e.name.replace(location.origin + '/', '').replace(/^api\/(\w+)\?.*?(detecteur=\w+)?$/, '$1 $2').trim()
                   .replace(/^https:\/\/.*\//, ''),
                 depart: Math.round(e.startTime), fin: Math.round(e.responseEnd) })));
  const resume = { tour, affichee: Math.round(affichee), avancement: 0 };
  for (const e of entrees) {
    if (e.nom === 'avancement') { resume.avancement++; continue; }
    if (resume[e.nom]) { resume[e.nom + ' (bis)'] = [e.depart, e.fin]; continue; }
    resume[e.nom] = [e.depart, e.fin];
  }
  for (const e of entrees) if (e.nom !== 'avancement') console.log(`${String(e.depart).padStart(7)} -> ${String(e.fin).padStart(7)} ms  ${e.nom}`);
  console.log(JSON.stringify(resume));
}
console.log(erreurs.length ? erreurs.join('\n') : 'aucune erreur JavaScript');
await navigateur.close();
