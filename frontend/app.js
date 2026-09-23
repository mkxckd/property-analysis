/* SouqPulse dashboard rendering. Called as SouqPulse.init(DATA) once the
   data (either fetched from ./data/*.json or embedded inline) is ready. */

const SouqPulse = (() => {
  const palette = { dune: "#c9a15a", gulf: "#4d9186", rust: "#c1573f", sandDim: "#a89b86", line: "#362e23" };
  const hasChart = () => typeof Chart !== "undefined";

  function fmtCompact(n) { return new Intl.NumberFormat("en", { maximumFractionDigits: 0 }).format(n); }
  function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }
  function fmtAED(n) { return "AED " + fmtCompact(n); }

  function chartOptions(yLabel, extra) {
    return Object.assign({
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: palette.sandDim, maxTicksLimit: 6, font: { family: "IBM Plex Mono", size: 10 } },
          grid: { color: palette.line },
          title: { display: !!extra?.xLabel, text: extra?.xLabel, color: palette.sandDim, font: { size: 10 } },
        },
        y: {
          ticks: { color: palette.sandDim, font: { family: "IBM Plex Mono", size: 10 } },
          grid: { color: palette.line },
          title: { display: !!yLabel, text: yLabel, color: palette.sandDim, font: { size: 10 } },
        },
      },
    }, extra?.override || {});
  }

  function chartFallback(canvasId, message) {
    const el = document.getElementById(canvasId);
    if (el) el.closest("div").innerHTML = `<p style="color:var(--sand-dim);font-size:13px">${message}</p>`;
  }

  // ---------------- Hero: live scoring feed ----------------

  let feedTimer;

  function flagLabel(flag) {
    if (flag === "underpriced") return "under value";
    if (flag === "overpriced") return "over value";
    return "at value";
  }

  function renderHero(data) {
    const m = data.metrics;
    document.getElementById("stat-listings").textContent = fmtCompact(m.dedup.listings_scored);
    document.getElementById("stat-r2").textContent = m.model.r2.toFixed(3);
    document.getElementById("stat-mape").textContent = fmtPct(m.model.mape);

    const feed = data.live_feed;
    const el = document.getElementById("liveFeed");
    const caption = document.getElementById("feed-caption");
    const btn = document.getElementById("feedBtn");
    let i = 0;

    function reset() {
      el.innerHTML = "";
      i = 0;
    }

    function step() {
      if (i >= feed.length) { clearInterval(feedTimer); btn.textContent = "Replay"; caption.textContent = `${feed.length} listings scored`; return; }
      const row = feed[i];
      const div = document.createElement("div");
      div.className = "feed-item";
      div.innerHTML = `
        <span class="ftitle">${row.title}</span>
        <span class="fprice">${fmtAED(row.price_aed_month)}</span>
        <span class="feed-flag ${row.flag}">${flagLabel(row.flag)}</span>`;
      el.prepend(div);
      while (el.children.length > 12) el.removeChild(el.lastChild);
      caption.textContent = `scoring… ${row.area} · predicted ${fmtAED(row.predicted_price_aed_month)} vs listed ${fmtAED(row.price_aed_month)}`;
      i++;
    }

    caption.textContent = "press play to watch listings get scored as they arrive";
    btn.addEventListener("click", () => {
      clearInterval(feedTimer);
      reset();
      btn.textContent = "Scoring…";
      feedTimer = setInterval(step, 260);
    });
  }

  // ---------------- Ticker ----------------

  function renderTicker(data) {
    const d = data.metrics.dedup;
    const mo = data.metrics.model;
    const fr = data.metrics.fraud_detection;
    const st = data.metrics.streaming;

    const items = [
      { val: fmtPct(d.precision), lbl: "dedup precision", cls: "good" },
      { val: mo.r2.toFixed(3), lbl: "fair-value model R²", cls: "good" },
      { val: fmtPct(mo.mape), lbl: "fair-value model MAPE", cls: "" },
      { val: fmtPct(fr.model_based.f1), lbl: "fraud detection F1 (model)", cls: "good" },
      { val: st.avg_latency_ms.toFixed(2) + " ms", lbl: "avg live scoring latency", cls: "" },
    ];
    document.getElementById("ticker").innerHTML = items.map((it) => `
      <div class="ticker-item"><span class="val ${it.cls}">${it.val}</span><span class="lbl">${it.lbl}</span></div>`).join("");
  }

  // ---------------- Model performance ----------------

  function renderModelPerformance(data) {
    const m = data.metrics.model;
    document.getElementById("model-summary").innerHTML =
      `Trained on <span class="mono">${fmtCompact(m.n_train)}</span> cleaned listings (duplicates collapsed, fraud excluded), evaluated on <span class="mono">${fmtCompact(m.n_test)}</span> held out. Mean absolute error <span class="mono">${fmtAED(m.mae_aed)}</span>/month, <span class="mono">${fmtPct(m.mape)}</span> MAPE, R² <span class="mono">${m.r2.toFixed(3)}</span>.`;

    if (!hasChart()) { chartFallback("scatterChart", "Chart library unavailable — see the summary above for the numbers."); }
    else {
      const pts = data.test_predictions.map((r) => ({ x: r.actual, y: r.predicted }));
      const maxV = Math.max(...pts.map((p) => Math.max(p.x, p.y))) * 1.05;
      new Chart(document.getElementById("scatterChart").getContext("2d"), {
        type: "scatter",
        data: { datasets: [
          { data: pts, backgroundColor: "rgba(77,145,134,0.55)", pointRadius: 2.5 },
          { data: [{ x: 0, y: 0 }, { x: maxV, y: maxV }], type: "line", borderColor: palette.sandDim, borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
        ] },
        options: chartOptions("predicted (AED/mo)", { xLabel: "actual (AED/mo)" }),
      });
    }

    const el = document.getElementById("importanceRows");
    el.innerHTML = data.metrics.feature_importances.map((r) => `
      <div class="importance-row">
        <div class="importance-row-head"><span class="field">${r.feature}</span><span class="mono">${fmtPct(r.importance)}</span></div>
        <div class="importance-bar-track"><span class="importance-bar-fill" style="width:${(r.importance * 100).toFixed(1)}%"></span></div>
      </div>`).join("");
  }

  // ---------------- Market leaderboard ----------------

  function renderLeaderboard(data) {
    const rows = [...data.area_summary].sort((a, b) => b.avg_price_per_sqft_annual - a.avg_price_per_sqft_annual);
    const max = rows[0].avg_price_per_sqft_annual;
    document.getElementById("leaderboard").innerHTML = rows.map((r) => `
      <div class="leaderboard-row">
        <span>${r.area}</span>
        <span class="leaderboard-bar-track"><span class="leaderboard-bar-fill" style="width:${(r.avg_price_per_sqft_annual / max * 100).toFixed(1)}%"></span></span>
        <span class="val">${Math.round(r.avg_price_per_sqft_annual)}</span>
      </div>`).join("");
  }

  // ---------------- Warehouse queries ----------------

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function fmtCell(v) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") return Number.isInteger(v) ? fmtCompact(v) : v.toFixed(1);
    return escapeHtml(v);
  }

  function renderWarehouse(data) {
    const queries = data.warehouse_queries;
    const tabs = document.getElementById("queryTabs");

    function show(i) {
      const q = queries[i];
      tabs.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
      document.getElementById("queryDesc").textContent = q.description;
      document.getElementById("querySql").textContent = q.sql;
      const numeric = q.columns.map((_, c) => q.rows.every((r) => r[c] === null || typeof r[c] === "number"));
      document.getElementById("queryTable").innerHTML =
        `<thead><tr>${q.columns.map((c, k) => `<th${numeric[k] ? ' style="text-align:right"' : ""}>${escapeHtml(c)}</th>`).join("")}</tr></thead>` +
        `<tbody>${q.rows.map((r) => `<tr>${r.map((v, k) => `<td${numeric[k] ? ' class="num"' : ""}>${fmtCell(v)}</td>`).join("")}</tr>`).join("")}</tbody>`;
    }

    tabs.innerHTML = queries.map((q, i) => `<button class="toggle-btn" role="tab" data-i="${i}">${escapeHtml(q.title)}</button>`).join("");
    tabs.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => show(Number(b.dataset.i))));
    show(0);
  }

  // ---------------- Trust engine ----------------

  const CLUSTER_PAGE = 12;

  function renderClusters(data) {
    const clusters = data.duplicate_clusters;
    const total = data.duplicate_clusters_total || clusters.length;
    const countEl = document.getElementById("cluster-count");
    const grid = document.getElementById("clusterGrid");
    const moreBtn = document.getElementById("clusterMoreBtn");
    let shown = 0;

    function showMore() {
      const next = clusters.slice(shown, shown + CLUSTER_PAGE);
      grid.insertAdjacentHTML("beforeend", next.map(clusterCard).join(""));
      shown += next.length;
      countEl.textContent = `showing ${shown} of ${total} duplicate clusters — collapsed to 1 row each before training`;
      moreBtn.hidden = shown >= clusters.length;
    }

    grid.innerHTML = "";
    moreBtn.addEventListener("click", showMore);
    showMore();
  }

  function clusterCard(c) {
    const topScore = Math.max(...c.pair_scores.map((p) => p.score));
    const rows = c.members.map((m) => `
      <div class="listing-row">
        <span class="title">${m.title}</span>
        <span class="meta">${m.area} · ${m.bedrooms}BR · ${m.size_sqft} sqft · ${fmtAED(m.price_aed_month)}/mo · <span class="agent">${m.agent_name}</span> · ${m.listed_date}</span>
      </div>`).join("");
    return `
      <div class="cluster-card">
        <div class="cluster-card-head">
          <span class="mono" style="font-size:12px;color:var(--sand-dim)">${c.members.length} listings, same unit</span>
          <span class="cluster-score">match ${(topScore * 100).toFixed(0)}%</span>
        </div>
        ${rows}
      </div>`;
  }

  // ---------------- Fraud: naive vs model ----------------

  function renderFraudComparison(data) {
    const fr = data.metrics.fraud_detection;
    const card = (title, m, extraClass) => `
      <div class="comparison-card ${extraClass || ""}">
        <h4>${title}</h4>
        <div class="comparison-stat"><span>precision</span><span class="v">${fmtPct(m.precision)}</span></div>
        <div class="comparison-stat"><span>recall</span><span class="v">${fmtPct(m.recall)}</span></div>
        <div class="comparison-stat"><span>F1</span><span class="v">${fmtPct(m.f1)}</span></div>
        <div class="comparison-stat"><span>false positives</span><span class="v">${m.false_positives}</span></div>
      </div>`;
    document.getElementById("comparisonGrid").innerHTML =
      card("Naive: below 65% of area+type median", fr.naive_baseline) +
      card("Model: >30% below predicted fair value", fr.model_based, "highlight");
    const pts = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(1)} pts`;
    const recallDelta = fr.model_based.recall - fr.naive_baseline.recall;
    document.getElementById("comparison-delta").textContent =
      `Model vs. naive rule: precision ${pts(fr.precision_improvement)}, recall ${pts(recallDelta)}, F1 ${pts(fr.f1_improvement)}. ` +
      `Every listing is scored by a model that never trained on it (5-fold, grouped by physical unit).`;

    const rows = data.flagged_listings.slice(0, 25);
    document.getElementById("flaggedBody").innerHTML = rows.map((r) => {
      const gapPct = Math.abs(r.residual_pct * 100);
      return `
        <tr>
          <td>${r.title}</td>
          <td>${r.area}</td>
          <td class="num">${fmtAED(r.price_aed_month)}</td>
          <td class="num">${fmtAED(Math.round(r.predicted_price_aed_month))}</td>
          <td>
            <span class="gap-bar-track"><span class="gap-bar-fill" style="width:${Math.min(100, gapPct).toFixed(0)}%"></span></span>
            <span class="mono" style="font-size:12px;color:var(--rust);margin-left:6px">-${gapPct.toFixed(0)}%</span>
          </td>
        </tr>`;
    }).join("");
  }

  // ---------------- Interactive estimator ----------------

  function renderEstimator(data) {
    const grid = data.fair_value_grid;
    const areas = [...new Set(grid.map((r) => r.area))].sort();
    const types = [...new Set(grid.map((r) => r.property_type))];

    const areaSel = document.getElementById("est-area");
    const typeSel = document.getElementById("est-type");
    areaSel.innerHTML = areas.map((a) => `<option value="${a}">${a}</option>`).join("");
    typeSel.innerHTML = types.map((t) => `<option value="${t}">${t}</option>`).join("");

    const sizeSlider = document.getElementById("est-size");
    const sizeLabel = document.getElementById("est-size-label");
    const furnBtns = document.querySelectorAll(".toggle-btn[data-furnished]");
    let furnished = 0;

    function sizeRangeFor(type) {
      const rows = grid.filter((r) => r.property_type === type);
      const sizes = rows.map((r) => r.size_sqft);
      return [Math.min(...sizes), Math.max(...sizes)];
    }

    function updateSizeRange() {
      const [lo, hi] = sizeRangeFor(typeSel.value);
      sizeSlider.min = lo; sizeSlider.max = hi;
      sizeSlider.value = Math.round((lo + hi) / 2);
    }

    function estimate() {
      const area = areaSel.value, type = typeSel.value, size = Number(sizeSlider.value);
      sizeLabel.textContent = `${size} sqft`;

      const candidates = grid.filter((r) => r.area === area && r.property_type === type && r.furnished === furnished);
      if (!candidates.length) return;
      candidates.sort((a, b) => Math.abs(a.size_sqft - size) - Math.abs(b.size_sqft - size));
      const lower = candidates.filter((r) => r.size_sqft <= size).sort((a, b) => b.size_sqft - a.size_sqft)[0] || candidates[0];
      const upper = candidates.filter((r) => r.size_sqft >= size).sort((a, b) => a.size_sqft - b.size_sqft)[0] || candidates[0];

      let est;
      if (lower.size_sqft === upper.size_sqft) {
        est = lower.predicted_price_aed_month;
      } else {
        const t = (size - lower.size_sqft) / (upper.size_sqft - lower.size_sqft);
        est = lower.predicted_price_aed_month + t * (upper.predicted_price_aed_month - lower.predicted_price_aed_month);
      }
      document.getElementById("est-result-value").textContent = fmtAED(Math.round(est));
      document.getElementById("est-result-sub").textContent =
        `${area} · ${type} · ${size} sqft · ${furnished ? "furnished" : "unfurnished"} — model estimate, interpolated from precomputed grid`;
    }

    areaSel.addEventListener("change", estimate);
    typeSel.addEventListener("change", () => { updateSizeRange(); estimate(); });
    sizeSlider.addEventListener("input", estimate);
    furnBtns.forEach((b) => b.addEventListener("click", () => {
      furnBtns.forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      furnished = Number(b.dataset.furnished);
      estimate();
    }));

    updateSizeRange();
    estimate();
  }

  // ---------------- Pipeline health ----------------

  function renderHealth(data) {
    const dq = data.metrics.data_quality;
    const wh = data.metrics.warehouse;

    document.getElementById("qualityRows").innerHTML = Object.entries(dq.field_completeness).map(([field, pct]) => `
      <div class="quality-row">
        <div class="quality-row-head"><span class="field">${field}</span><span class="mono">${fmtPct(pct)}</span></div>
        <div class="quality-bar-track"><span class="quality-bar-fill" style="width:${(pct * 100).toFixed(0)}%"></span></div>
      </div>`).join("") + `
      <div class="quality-row">
        <div class="quality-row-head"><span class="field">exact-match duplicate rows (naive check)</span><span class="mono">${dq.exact_duplicate_rows}</span></div>
      </div>`;

    const run = data.metrics.etl_last_run;
    document.getElementById("warehouseList").innerHTML = Object.entries(wh).map(([table, n]) => `
      <div><span>${table}</span><span class="n">${fmtCompact(n)}</span></div>`).join("") + (run ? `
      <p class="etl-run">last load: ${fmtCompact(run.rows_inserted)} inserted · ${fmtCompact(run.rows_updated)} updated · ${fmtCompact(run.rows_unchanged)} unchanged · ${fmtCompact(run.rows_deactivated)} deactivated</p>` : "");
  }

  function init(data) {
    // Each panel renders independently: one failing panel logs and shows an
    // inline notice instead of silently taking down every panel after it.
    const panels = [
      [renderHero, ".hero"],
      [renderTicker, "#ticker"],
      [renderModelPerformance, "#model-performance"],
      [renderLeaderboard, "#market-index"],
      [renderWarehouse, "#warehouse"],
      [renderClusters, "#trust-engine"],
      [renderFraudComparison, "#fraud"],
      [renderEstimator, "#estimator"],
      [renderHealth, "#pipeline-health"],
    ];
    for (const [render, selector] of panels) {
      try {
        render(data);
      } catch (err) {
        console.error(`SouqPulse: ${render.name} failed`, err);
        document.querySelector(selector)?.insertAdjacentHTML("beforeend",
          `<p class="panel-error">This panel failed to render (${err.message}). See the console for details.</p>`);
      }
    }
  }

  return { init };
})();
