(() => {
  "use strict";

  /* ---------------- theme ---------------- */
  const root = document.documentElement;
  const themeBtn = document.getElementById("themeToggle");
  const storedTheme = localStorage.getItem("tcve-theme");
  if (storedTheme) root.setAttribute("data-theme", storedTheme);
  function currentTheme() {
    if (root.getAttribute("data-theme")) return root.getAttribute("data-theme");
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  themeBtn.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("tcve-theme", next);
  });

  /* ---------------- mobile menu ---------------- */
  const menuBtn = document.getElementById("menuToggle");
  const mobileNav = document.getElementById("mobileNav");
  menuBtn.addEventListener("click", () => mobileNav.classList.toggle("open"));
  mobileNav.querySelectorAll("a").forEach(a => a.addEventListener("click", () => mobileNav.classList.remove("open")));

  /* ---------------- scrollspy ---------------- */
  const navLinks = Array.from(document.querySelectorAll("nav.links a"));
  const sections = navLinks.map(a => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  function onScroll() {
    let activeIdx = 0;
    const pos = window.scrollY + 120;
    sections.forEach((sec, i) => { if (sec.offsetTop <= pos) activeIdx = i; });
    navLinks.forEach(a => a.classList.remove("active"));
    const activeHref = sections[activeIdx] ? "#" + sections[activeIdx].id : null;
    navLinks.filter(a => a.getAttribute("href") === activeHref).forEach(a => a.classList.add("active"));
  }
  document.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------------- reveal on scroll ---------------- */
  const revealEls = document.querySelectorAll(".reveal");
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { threshold: 0.12 });
  revealEls.forEach(el => io.observe(el));

  /* ---------------- animated counters ---------------- */
  const counters = document.querySelectorAll("[data-count]");
  function animateCounter(el) {
    const target = parseFloat(el.getAttribute("data-count"));
    const decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
    const suffix = el.getAttribute("data-suffix") || "";
    const dur = 1100;
    const start = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = target * eased;
      el.textContent = val.toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = target.toFixed(decimals) + suffix;
    }
    requestAnimationFrame(tick);
  }
  const counterIo = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) { animateCounter(e.target); counterIo.unobserve(e.target); } });
  }, { threshold: 0.4 });
  counters.forEach(c => counterIo.observe(c));

  /* ---------------- pipeline accordion ---------------- */
  const pipelineData = [
    { title: "Data collection", file: "count_models.py · fetch_models.py", body: "Query the HF Hub API for models tagged medical, clinical, healthcare, biomedical, radiology, or pathology (8,209 unique candidates). Randomly sample and fetch each candidate's raw README + metadata (author, downloads, likes, creation date, pipeline tag); cards <50 chars discarded and replaced. Author namespace checked against the HF Organizations API. Result: 500 valid healthcare models." },
    { title: "Coding scheme design", file: "study_design.md", body: "3 dimensions per card — trustworthiness/safety, bias/fairness, interpretability — each coded as a binary claim field (explicit positive assertion) + a 3-valued evidence field (named method, metric, test, benchmark, or citation). Plus clinical_use_disclaimer and interpretability_method_named. Derived gap_score (0–3) = dimensions where claim=yes AND evidence=no." },
    { title: "Classification & reliability", file: "aggregate_majority.py", body: "500 cards split into 10 batches of 50, classified using Claude (Sonnet 5) via Claude Code's agentic orchestration — zero-shot, rubric-based prompting. Every batch classified independently 3× (3,000 classifications total) and majority-voted per field; evidence ties resolve to 'partial'. Fleiss' κ per field pooled across healthcare + control: 0.525–0.920, overall average κ = 0.678 (substantial agreement)." },
    { title: "Control group", file: "fetch_control.py", body: "A numeric comparison against prior work's published statistics would be invalid (different coding/sampling). Self-collected control group instead: 20,000 most-recently-created HF models, excluding health-tagged ones, sampled and processed identically → 500 valid general-purpose models." },
    { title: "Analysis", file: "analyze_final.py · analyze_combined.py", body: "RQ1: chi-square test on documentation rates pre/post EU AI Act entry into force (2024-08-01). RQ2: Mann-Whitney U on gap_score + chi-square on any_claim/disclaimer, by authorship. RQ3: multivariate logistic regression (any_claim ~ domain + log(1+downloads) + is_organization) and analogous OLS on gap_score, on the combined N=1,000 dataset." },
  ];
  const pipelineEl = document.getElementById("pipeline");
  pipelineData.forEach((step, i) => {
    const div = document.createElement("div");
    div.className = "step";
    div.innerHTML = `
      <div class="num-badge">${i + 1}</div>
      <div>
        <h3>${step.title} <svg class="chev" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg></h3>
        <div class="body"><p style="margin:0 0 8px;">${step.body}</p><code class="file">${step.file}</code></div>
      </div>`;
    div.addEventListener("click", () => div.classList.toggle("open"));
    pipelineEl.appendChild(div);
  });
  pipelineEl.firstElementChild.classList.add("open");

  /* ---------------- tabs ---------------- */
  const tabBtns = document.querySelectorAll("#rqTabs button");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".tabpanel").forEach(p => p.classList.remove("active"));
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    });
  });

  /* ---------------- SVG chart helpers ---------------- */
  const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const NS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs) {
    const el = document.createElementNS(NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  function groupedBarChart(container, { labels, series, colors, height = 260, valueFmt = v => (v * 100).toFixed(1) + "%", yMax = null }) {
    const width = 560;
    const padL = 40, padR = 12, padT = 18, padB = 34;
    const chartW = width - padL - padR, chartH = height - padT - padB;
    const maxVal = yMax || Math.max(0.001, ...series.flatMap(s => s.data)) * 1.25;
    const groupW = chartW / labels.length;
    const barGap = 6;
    const barW = (groupW - barGap * (series.length + 1)) / series.length;

    const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, width: "100%", height: height });

    // gridlines
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const y = padT + chartH - (chartH * i) / ticks;
      const val = (maxVal * i) / ticks;
      svg.appendChild(svgEl("line", { x1: padL, x2: width - padR, y1: y, y2: y, stroke: cssVar("--border"), "stroke-width": 1 }));
      const t = svgEl("text", { x: padL - 8, y: y + 4, "text-anchor": "end", "font-size": 10, class: "dim" });
      t.textContent = (val * 100).toFixed(0) + "%";
      svg.appendChild(t);
    }

    labels.forEach((label, gi) => {
      const gx = padL + gi * groupW;
      series.forEach((s, si) => {
        const val = s.data[gi];
        const h = (val / maxVal) * chartH;
        const x = gx + barGap + si * (barW + barGap);
        const y = padT + chartH - h;
        const rect = svgEl("rect", {
          class: "bar-rect", x, y: padT + chartH, width: barW, height: 0,
          fill: colors[si], rx: 4,
        });
        rect.appendChild(svgEl("title", {})).textContent = `${s.name} · ${label}: ${valueFmt(val)}`;
        svg.appendChild(rect);
        requestAnimationFrame(() => {
          rect.style.transition = "y 0.7s cubic-bezier(.2,.8,.2,1), height 0.7s cubic-bezier(.2,.8,.2,1)";
          rect.setAttribute("y", y);
          rect.setAttribute("height", Math.max(0, h));
        });
      });
      const lt = svgEl("text", { x: gx + groupW / 2, y: height - padB + 18, "text-anchor": "middle", "font-size": 11, class: "dim" });
      lt.textContent = label;
      svg.appendChild(lt);
    });

    container.innerHTML = "";
    container.appendChild(svg);

    // legend
    if (series.length > 1) {
      const leg = document.createElement("div");
      leg.style.cssText = "display:flex;gap:16px;margin-top:8px;flex-wrap:wrap;";
      series.forEach((s, i) => {
        const item = document.createElement("div");
        item.style.cssText = "display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-dim);";
        item.innerHTML = `<span style="width:10px;height:10px;border-radius:3px;background:${colors[i]};display:inline-block;"></span>${s.name}`;
        leg.appendChild(item);
      });
      container.appendChild(leg);
    }
  }

  function forestPlot(container, { labels, or_, lo, hi, height = 190 }) {
    const width = 560;
    const padL = 140, padR = 20, padT = 16, padB = 30;
    const chartW = width - padL - padR;
    const rowH = (height - padT - padB) / labels.length;
    const maxX = Math.max(2.5, ...hi) * 1.05;
    const xScale = v => padL + (v / maxX) * chartW;
    const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, width: "100%", height });

    // x axis ticks
    [0, 0.5, 1, 1.5, 2, 2.5].forEach(v => {
      if (v > maxX) return;
      const x = xScale(v);
      svg.appendChild(svgEl("line", { x1: x, x2: x, y1: padT, y2: height - padB, stroke: cssVar("--border"), "stroke-width": v === 1 ? 1.4 : 1, "stroke-dasharray": v === 1 ? "0" : "3,3" }));
      const t = svgEl("text", { x, y: height - padB + 16, "text-anchor": "middle", "font-size": 10, class: "dim" });
      t.textContent = v;
      svg.appendChild(t);
    });

    labels.forEach((label, i) => {
      const y = padT + i * rowH + rowH / 2;
      const lt = svgEl("text", { x: padL - 10, y: y + 4, "text-anchor": "end", "font-size": 11.5, class: "dim" });
      lt.textContent = label;
      svg.appendChild(lt);

      const line = svgEl("line", { x1: xScale(lo[i]), x2: xScale(lo[i]), y1: y, y2: y, stroke: cssVar("--accent"), "stroke-width": 2 });
      svg.appendChild(line);
      const dot = svgEl("circle", { cx: xScale(lo[i]), cy: y, r: 4.5, fill: cssVar("--accent") });
      dot.appendChild(svgEl("title", {})).textContent = `${label}: OR=${or_[i]} [${lo[i]}, ${hi[i]}]`;
      svg.appendChild(dot);
      requestAnimationFrame(() => {
        line.style.transition = "x1 0.8s cubic-bezier(.2,.8,.2,1), x2 0.8s cubic-bezier(.2,.8,.2,1)";
        line.setAttribute("x2", xScale(hi[i]));
        dot.style.transition = "cx 0.8s cubic-bezier(.2,.8,.2,1)";
        dot.setAttribute("cx", xScale(or_[i]));
      });
      const vt = svgEl("text", { x: xScale(hi[i]) + 8, y: y + 4, "font-size": 10.5, class: "dim" });
      vt.textContent = `OR ${or_[i]}`;
      svg.appendChild(vt);
    });

    container.innerHTML = "";
    container.appendChild(svg);
  }

  /* ---------------- RQ1: grouped bars over time ---------------- */
  groupedBarChart(document.getElementById("chartRQ1"), {
    labels: ["Pre (N=80)", "Post (N=420)"],
    series: [
      { name: "Any claim", data: [0.063, 0.102] },
      { name: "Clinical-use disclaimer", data: [0.088, 0.198] },
    ],
    colors: [cssVar("--accent"), cssVar("--accent-3")],
  });

  /* ---------------- RQ2: org vs individual ---------------- */
  groupedBarChart(document.getElementById("chartRQ2"), {
    labels: ["Organization (N=149)", "Individual (N=351)"],
    series: [
      { name: "Any claim", data: [0.128, 0.083] },
      { name: "Clinical-use disclaimer", data: [0.067, 0.228] },
    ],
    colors: [cssVar("--accent"), cssVar("--accent-3")],
  });

  /* ---------------- RQ3: forest plot ---------------- */
  forestPlot(document.getElementById("chartRQ3"), {
    labels: ["Healthcare domain", "log(1+downloads)", "Organization-authored"],
    or_: [0.70, 0.99, 1.26],
    lo: [0.47, 0.90, 0.81],
    hi: [1.05, 1.09, 1.96],
  });

  /* ---------------- reliability kappa bars ---------------- */
  const kappaData = [
    { field: "trust_claim", value: 0.759 },
    { field: "bias_claim", value: 0.832 },
    { field: "interp_claim", value: 0.848 },
    { field: "clinical_use_disclaimer", value: 0.960 },
    { field: "trust_evidence", value: 0.700 },
    { field: "bias_evidence", value: 0.832 },
    { field: "interp_evidence", value: 0.822 },
    { field: "overall average", value: 0.678 },
  ];
  const kappaContainer = document.getElementById("kappaChart");
  kappaData.forEach(({ field, value }) => {
    const row = document.createElement("div");
    row.className = "kappa-row";
    row.innerHTML = `
      <div class="field">${field}</div>
      <div class="kappa-bar-track"><div class="kappa-bar-fill"></div></div>
      <div class="kappa-val">${value.toFixed(3)}</div>`;
    kappaContainer.appendChild(row);
    const fillIo = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          row.querySelector(".kappa-bar-fill").style.width = (value * 100) + "%";
          fillIo.unobserve(e.target);
        }
      });
    }, { threshold: 0.3 });
    fillIo.observe(row);
  });

  /* ---------------- copy bibtex ---------------- */
  const toast = document.getElementById("toast");
  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add("show");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => toast.classList.remove("show"), 1800);
  }
  document.getElementById("copyBib").addEventListener("click", async () => {
    const text = document.getElementById("bibtex").textContent;
    try {
      await navigator.clipboard.writeText(text);
      showToast("BibTeX copied to clipboard");
    } catch {
      showToast("Copy failed — select manually");
    }
  });
})();
