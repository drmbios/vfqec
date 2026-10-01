'use strict';

const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const METHODS = {
  unencoded: { label: 'Unencoded', color: '#92a19a', dash: '5 4' },
  standard: { label: 'Standard QEC', color: '#d48255' },
  twirled: { label: 'QEC + twirling', color: '#a082bd' },
  calibrated: { label: 'Learned field', color: '#087b61' },
  adaptive: { label: 'Adaptive field', color: '#4f85b7', dash: '6 3' },
  oracle: { label: 'Oracle', color: '#b49528', dash: '2 4' },
};
let result;
let candidates = [];
let fullScale = false;
let playback = null;
const percent = (p, digits = 2) => `${(100 * p).toFixed(digits)}%`;

function svgNode(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function add(svg, tag, attrs, text) {
  const node = svgNode(tag, attrs, text);
  svg.append(node);
  return node;
}

function path(points) {
  return points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
}

function wilson(errors, shots) {
  const z = 1.959963984540054;
  const p = errors / shots;
  const denom = 1 + z * z / shots;
  const center = (p + z * z / (2 * shots)) / denom;
  const half = z * Math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / denom;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

function drawCalibration() {
  const svg = $('calibration-chart');
  svg.replaceChildren();
  const step = Number($('step-slider').value);
  const row = candidates[step];
  const w = 540, h = 195, left = 47, right = 12, top = 12, bottom = 35;
  const max = Math.max(...candidates.map((r) => r.cost)) * 1.18;
  const x = (i) => left + i / (candidates.length - 1) * (w - left - right);
  const y = (cost) => top + (1 - cost / max) * (h - top - bottom);
  for (let i = 0; i <= 3; i++) {
    const value = max * i / 3;
    add(svg, 'line', { x1: left, x2: w - right, y1: y(value), y2: y(value), class: 'grid-line' });
    add(svg, 'text', { x: left - 8, y: y(value) + 3, 'text-anchor': 'end' }, percent(value, 1));
  }
  [0, 10, 20, 30, 40, 50].forEach((i) => add(svg, 'text', { x: x(i), y: h - 12, 'text-anchor': 'middle' }, String(i)));
  add(svg, 'path', { d: path(candidates.map((r, i) => [x(i), y(r.cost)])), fill: 'none', stroke: '#dce5da', 'stroke-width': 2 });
  add(svg, 'path', { d: path(candidates.slice(0, step + 1).map((r, i) => [x(i), y(r.cost)])), fill: 'none', stroke: '#087b61', 'stroke-width': 2.5, 'stroke-linejoin': 'round' });
  add(svg, 'line', { x1: x(step), x2: x(step), y1: top, y2: h - bottom, class: 'cursor-line' });
  add(svg, 'circle', { cx: x(step), cy: y(row.cost), r: 5, fill: '#087b61', stroke: 'white', 'stroke-width': 2 });
  $('syndrome-cost').textContent = percent(row.cost, 3);
  $('step-label').textContent = `Step ${row.step} / ${result.config.iterations}`;
  $('step-value').textContent = String(row.step);
  row.theta.forEach((theta, i) => {
    const fill = $(`field-fill-${i}`);
    // Display the full permitted control range [-0.7, 0.7] without clipping candidates.
    const width = Math.min(50, Math.abs(theta) / 0.7 * 50);
    fill.style.width = `${width}%`;
    fill.style.left = `${theta < 0 ? 50 - width : 50}%`;
    $(`field-value-${i}`).textContent = `${theta >= 0 ? '+' : ''}${theta.toFixed(4)}`;
  });
}

function drawResults() {
  const svg = $('results-chart');
  svg.replaceChildren();
  const round = Number($('round-slider').value);
  const w = 650, h = 300, left = 48, right = 14, top = 12, bottom = 40;
  const n = result.config.rounds;
  const encodedMax = Math.max(...Object.entries(result.methods).filter(([k]) => k !== 'unencoded').flatMap(([, value]) => value.logical_error));
  const max = fullScale ? 1 : Math.max(0.01, encodedMax * 1.25);
  const x = (i) => left + i / (n - 1) * (w - left - right);
  const y = (probability) => top + (1 - probability / max) * (h - top - bottom);
  const defs = add(svg, 'defs');
  const clip = add(defs, 'clipPath', { id: 'plot-clip' });
  add(clip, 'rect', { x: left, y: top, width: w - left - right, height: h - top - bottom });
  for (let i = 0; i <= 4; i++) {
    const value = max * i / 4;
    add(svg, 'line', { x1: left, x2: w - right, y1: y(value), y2: y(value), class: 'grid-line' });
    add(svg, 'text', { x: left - 8, y: y(value) + 3, 'text-anchor': 'end' }, `${(value * 100).toFixed(fullScale ? 0 : 2)}%`);
  }
  [1, 10, 20, 30, 40, 50, 60].forEach((r) => add(svg, 'text', { x: x(r - 1), y: h - 20, 'text-anchor': 'middle' }, String(r)));
  add(svg, 'text', { x: (left + w - right) / 2, y: h - 1, 'text-anchor': 'middle' }, 'QEC round');
  const plot = add(svg, 'g', { 'clip-path': 'url(#plot-clip)' });
  const learned = result.methods.calibrated;
  const bands = learned.errors.map((errors) => wilson(errors, learned.shots));
  const upper = bands.map((band, i) => [x(i), y(band[1])]);
  const lower = bands.map((band, i) => [x(i), y(band[0])]).reverse();
  add(plot, 'path', { d: `${path([...upper, ...lower])} Z`, fill: '#087b61', 'fill-opacity': 0.07 });
  // Draw each method; distinct dash patterns keep overlapping learned/oracle traces legible.
  for (const [key, meta] of Object.entries(METHODS)) {
    const series = result.methods[key].logical_error;
    const attrs = { d: path(series.map((p, i) => [x(i), y(p)])), fill: 'none', stroke: meta.color, 'stroke-width': key === 'calibrated' ? 2.8 : 1.8, 'stroke-linejoin': 'round' };
    if (meta.dash) attrs['stroke-dasharray'] = meta.dash;
    add(plot, 'path', attrs);
    if (series[round - 1] <= max) add(plot, 'circle', { cx: x(round - 1), cy: y(series[round - 1]), r: 3, fill: meta.color, stroke: 'white', 'stroke-width': 1 });
  }
  add(svg, 'line', { x1: x(round - 1), x2: x(round - 1), y1: top, y2: h - bottom, class: 'cursor-line' });
  $('round-value').textContent = String(round);
  for (const [prefix, method] of [['standard', 'standard'], ['learned', 'calibrated'], ['oracle', 'oracle']]) {
    const data = result.methods[method];
    $(prefix + '-error').textContent = percent(data.logical_error[round - 1]);
    $(prefix + '-count').textContent = `${data.errors[round - 1]} / ${data.shots.toLocaleString()} failures`;
  }
  $('results-table').replaceChildren();
  for (const [key, meta] of Object.entries(METHODS)) {
    const data = result.methods[key];
    const errors = data.errors[round - 1];
    const interval = wilson(errors, data.shots);
    const tr = document.createElement('tr');
    [meta.label, `${errors} / ${data.shots.toLocaleString()}`, percent(data.logical_error[round - 1], 3), `${percent(interval[0], 3)} – ${percent(interval[1], 3)}`].forEach((value) => {
      const td = document.createElement('td');
      td.textContent = value;
      tr.append(td);
    });
    $('results-table').append(tr);
  }
  $('scale-note').textContent = (fullScale ? 'Full range shows all six curves, including coherent oscillations in the unencoded memory.' : 'All six curves are drawn; the unencoded curve extends above this encoded-detail view.') + ' Shading: learned-field 95% Wilson interval.';
}

function stopPlayback() {
  clearInterval(playback);
  playback = null;
  $('play').textContent = '▶ Replay';
  $('play').setAttribute('aria-label', 'Play calibration replay');
}

function validate(data) {
  if (data.config.experiment !== 'E1' || data.config.rounds !== 60 || data.config.iterations !== 50 || data.remote_jobs.length !== 458) throw new Error('Unexpected benchmark configuration.');
  for (const key of Object.keys(METHODS)) {
    const method = data.methods[key];
    if (!method || method.logical_error.length !== data.config.rounds || method.errors.length !== data.config.rounds || method.shots !== 4000 || method.logical_error.some((p) => !Number.isFinite(p) || p < 0 || p > 1)) throw new Error('Invalid benchmark series.');
  }
  if (data.remote_jobs.some((job) => job.status !== 'complete')) throw new Error('Incomplete provider jobs.');
}

async function start() {
  try {
    const response = await fetch('./data/result.json');
    if (!response.ok) throw new Error(`Dataset request returned ${response.status}.`);
    result = await response.json();
    validate(result);
    candidates = result.optimizer.filter((row) => row.calibration_round === 0 && ['initial', 'candidate'].includes(row.phase));
    if (candidates.length !== result.config.iterations + 1) throw new Error('Calibration trace is incomplete.');
    const finalErrors = (method) => result.methods[method].errors.at(-1);
    $('improvement').textContent = `${(finalErrors('standard') / finalErrors('calibrated')).toFixed(0)}×`;
    $('job-count').textContent = result.remote_jobs.length.toLocaleString();
    $('round-count').textContent = String(result.config.rounds);
    $('deployed-theta').textContent = result.updates[0].theta.map((t) => `${t >= 0 ? '+' : ''}${t.toFixed(4)}`).join(' / ') + ' rad';
    for (let i = 0; i < result.config.distance; i++) {
      const row = document.createElement('div');
      row.className = 'field-row';
      const name = document.createElement('span');
      name.textContent = `Q${i + 1}`;
      const track = document.createElement('div');
      track.className = 'field-track';
      const fill = document.createElement('span');
      fill.className = 'field-fill';
      fill.id = `field-fill-${i}`;
      track.append(fill);
      const output = document.createElement('output');
      output.id = `field-value-${i}`;
      output.setAttribute('aria-label', `Qubit ${i + 1} candidate correction in radians`);
      row.append(name, track, output);
      $('field-bars').append(row);
    }
    for (const meta of Object.values(METHODS)) {
      const label = document.createElement('span');
      const line = document.createElement('i');
      line.className = 'legend-line';
      line.style.background = meta.color;
      label.append(line, document.createTextNode(meta.label));
      $('method-legend').append(label);
    }
    $('step-slider').max = String(candidates.length - 1);
    $('step-slider').disabled = false;
    $('round-slider').disabled = false;
    $('play').disabled = false;
    $('load-status').textContent = 'Verified experiment loaded.';
    $('load-status').classList.add('loaded');
    drawCalibration();
    drawResults();
  } catch (error) {
    $('load-status').classList.add('error');
    $('load-status').textContent = 'The recorded dataset could not be loaded. Please reload, or open the full report below.';
    console.error(error);
  }
}

$('step-slider').addEventListener('input', () => { stopPlayback(); drawCalibration(); });
$('round-slider').addEventListener('input', drawResults);
$('play').addEventListener('click', () => {
  if (playback) { stopPlayback(); return; }
  if (Number($('step-slider').value) === candidates.length - 1) $('step-slider').value = '0';
  $('play').textContent = 'Ⅱ Pause';
  $('play').setAttribute('aria-label', 'Pause calibration replay');
  playback = setInterval(() => {
    const step = Number($('step-slider').value) + 1;
    $('step-slider').value = String(step);
    drawCalibration();
    if (step >= candidates.length - 1) stopPlayback();
  }, 180);
});

for (const [id, full] of [['encoded-scale', false], ['full-scale', true]]) {
  $(id).addEventListener('click', () => {
    if (!result) return;
    fullScale = full;
    $('encoded-scale').classList.toggle('active', !full);
    $('full-scale').classList.toggle('active', full);
    $('encoded-scale').setAttribute('aria-pressed', String(!full));
    $('full-scale').setAttribute('aria-pressed', String(full));
    drawResults();
  });
}

$('copy-command').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText('vfqec run E1 --backend local');
    $('copy-command').textContent = 'Copied!';
  } catch {
    $('copy-command').textContent = 'Select command';
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('.command code'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }
  setTimeout(() => { $('copy-command').textContent = 'Copy'; }, 2000);
});

document.addEventListener('visibilitychange', () => { if (document.hidden) stopPlayback(); });
start();
