'use strict';

(async function loadResearch() {
  const find = (id) => document.getElementById(id);
  const names = {
    unencoded: 'Unencoded', standard: 'Standard QEC', twirled: 'QEC + twirling',
    oracle: 'Single-qubit-field oracle', spsa_budget: 'Budget SPSA',
    spsa_reference: 'Reference SPSA', fourier: 'Validated Fourier',
    adaptive_spsa: 'Adaptive SPSA', adaptive_fourier: 'Adaptive Fourier',
  };
  const panels = {
    heldout_static: 'Primary · 24 held-out static cases', development: 'Development · excluded from inference',
    phase_memory: 'Complementary X memory', measurement_noise: 'Measurement noise q = 2%',
    distance_5: 'Repetition distance 5', distance_7: 'Repetition distance 7',
    drift: 'Drift · static and adaptive controllers',
    'surface_Z_zz0.00': 'Surface Z memory · no ZZ', 'surface_Z_zz0.08': 'Surface Z memory · ZZ = 0.08',
    'surface_X_zz0.00': 'Surface X memory · no ZZ', 'surface_X_zz0.08': 'Surface X memory · ZZ = 0.08',
  };
  const pc = (v, digits = 3) => `${(v * 100).toFixed(digits)}%`;
  try {
    const response = await fetch('./research/summary.json');
    if (!response.ok) throw new Error('Study data unavailable');
    const study = await response.json();
    if (!study.primary.complete || study.primary.cases !== 24 || study.cases !== 48) throw new Error('Incomplete study');
    const rows = study.case_results.filter((r) => !r.cloud_verification);
    const groupSelect = find('research-group');
    const caseSelect = find('research-case');
    const primary = study.groups.heldout_static.methods;
    find('research-counts').textContent = `${primary.fourier.total_final_errors} vs. ${primary.spsa_reference.total_final_errors}`;
    const comparison = study.primary.comparisons.fourier_vs_spsa_reference;
    find('research-interval').textContent = `Each out of ${primary.fourier.total_evaluation_shots.toLocaleString()} readouts. Fourier − reference: ${(comparison.mean_difference * 100).toFixed(4)} percentage points; 95% interval [${comparison.ci95.map((p) => (p * 100).toFixed(4)).join(', ')}].`;
    const cloud = study.cloud_verification;
    find('research-cloud').textContent = cloud.status === 'complete' ? `$${Number(cloud.actual_usd).toFixed(2)} spent` : 'Verification in progress';
    find('research-cloud-note').textContent = cloud.status === 'complete' ? `${cloud.cases} predeclared cases · ${cloud.jobs} completed CPU jobs. Authorization: at most $10. Hybrid simulation; local QEC instrument.` : 'Preview data; cloud verification must finish before publication.';
    for (const [key, label] of Object.entries(panels)) {
      const option = document.createElement('option');
      option.value = key; option.textContent = label; groupSelect.append(option);
    }
    function showCase() {
      const row = rows.find((r) => r.id === caseSelect.value);
      if (!row) throw new Error('Unknown study case');
      const c = row.config;
      find('research-config').textContent = `${c.code} · distance ${c.distance} · ${c.basis} memory · ${c.rounds} rounds · p=${c.p} · q=${c.q} · ZZ=${c.zz}.`;
      const body = find('research-results'); body.replaceChildren();
      for (const [name, value] of Object.entries(row.methods)) {
        const tr = document.createElement('tr');
        if (name === 'fourier' || name === 'adaptive_fourier') tr.className = 'fourier-row';
        const initial = row.fits[name]?.shots_used;
        [names[name] || name, `${value.final_errors} / ${value.shots.toLocaleString()}`, pc(value.final_error), initial === undefined ? '—' : initial.toLocaleString()].forEach((v) => {
          const td = document.createElement('td'); td.textContent = v; tr.append(td);
        }); body.append(tr);
      }
      find('research-case-note').textContent = c.drift_amplitude ? 'Adaptive arms reuse the initial fit and recalibrate at the stated cadence. Full executed and unused reference calibration costs are retained in the study report.' : 'Zero observed failures have nonzero uncertainty. The oracle only cancels single-qubit fields; it cannot cancel ZZ crosstalk.';
    }
    function showGroup() {
      caseSelect.replaceChildren();
      rows.filter((r) => r.group === groupSelect.value).sort((a, b) => a.seed - b.seed).forEach((row) => {
        const option = document.createElement('option'); option.value = row.id;
        option.textContent = `Seed ${row.seed}`; caseSelect.append(option);
      }); showCase();
    }
    groupSelect.addEventListener('change', showGroup);
    caseSelect.addEventListener('change', showCase);
    groupSelect.disabled = false; caseSelect.disabled = false;
    showGroup(); find('research-status').hidden = true;
    const legacyResponse = await fetch('./research/legacy-summary.json');
    if (!legacyResponse.ok) throw new Error('Legacy evidence unavailable');
    const legacy = await legacyResponse.json();
    if (legacy.status !== 'complete') throw new Error('Incomplete legacy evidence');
    const legacyBody = find('research-legacy-results');
    for (const [suite, panel] of Object.entries(legacy.suites)) {
      for (const run of panel.runs) {
        const c = run.config;
        const tr = document.createElement('tr');
        const settings = `d=${c.distance}, ${c.basis}, p=${c.p}, K=${c.cadence}, probe shots=${c.shots}`;
        const values = [suite, settings, ...['unencoded', 'standard', 'twirled', 'calibrated', 'adaptive', 'oracle'].map((m) => `${run.final_methods[m].errors}/${run.final_methods[m].shots}`)];
        values.forEach((value) => { const td = document.createElement('td'); td.textContent = value; tr.append(td); });
        legacyBody.append(tr);
      }
    }
  } catch (error) {
    find('research-status').textContent = 'Study explorer could not load. The full report and raw results remain available below.';
    console.error('Research data failed validation.');
  }
})();
