/* مصدر — chat page.
 *
 * Everything that comes from a source (titles, notes, publisher names, cell
 * values) is inserted as text, never as markup, and only http(s) links are
 * rendered as links. No inline script or style: the page's content policy
 * forbids both.
 */
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var TONE = {
    available: "ok", partial: "notice", not_available: "notice",
    unverified: "notice", no_source: "info", source_unreachable: "bad"
  };
  var VERDICT_ICON = {
    available: "check", partial: "half", not_available: "calendar",
    unverified: "alert", no_source: "search", source_unreachable: "cloud"
  };
  var EVIDENCE = {
    observed_data: "من داخل الملف",
    metadata_claim: "من وصف المصدر",
    inferred_title: "من العنوان فقط"
  };
  var KIND_TAG = { "رسمي": "tag-official", "لحظي": "tag-live", "دولي": "tag-intl" };
  var ICONS = {
    check: "M20 6 9 17l-5-5",
    half: "M12 3a9 9 0 1 0 0 18V3Z M12 3a9 9 0 0 1 0 18",
    calendar: "M8 3v3M16 3v3M4 9h16M5 5h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Z M10 13l4 4M14 13l-4 4",
    alert: "M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z",
    search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM21 21l-4.3-4.3",
    cloud: "M17.5 19H7a5 5 0 1 1 1.1-9.9A6 6 0 0 1 19.5 11a4 4 0 0 1-2 8Z M3 3l18 18",
    download: "M12 3v12M7 10l5 5 5-5M5 21h14",
    external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
    file: "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z M14 3v5h5",
    database: "M12 8c4.4 0 8-1.3 8-3s-3.6-3-8-3-8 1.3-8 3 3.6 3 8 3Z M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5 M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
    globe: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M3 12h18 M12 3a14 14 0 0 1 0 18 M12 3a14 14 0 0 0 0 18",
    bolt: "M13 2 4 14h7l-1 8 9-12h-7l1-8Z",
    info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M12 11v5M12 8h.01",
    archive: "M3 4h18v4H3z M5 8v11a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8 M10 12h4",
    clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M12 7v5l3 2"
  };

  var body = document.body;
  var scroller = document.getElementById("scroller");
  var thread = document.getElementById("thread");
  var form = document.getElementById("form");
  var input = document.getElementById("input");
  var send = document.getElementById("send");
  var heroSlot = document.getElementById("hero-slot");
  var dock = document.getElementById("dock");
  var session = null;
  var busy = false;
  var sourceKinds = {};

  // -- small helpers --------------------------------------------------
  function store(kind) {
    try { return window[kind]; } catch (e) { return null; }
  }
  function load(kind, key) {
    try { var s = store(kind); return s ? s.getItem(key) : null; } catch (e) { return null; }
  }
  function save(kind, key, value) {
    try {
      var s = store(kind);
      if (!s) return;
      if (value === null) s.removeItem(key); else s.setItem(key, value);
    } catch (e) { /* storage is a convenience only */ }
  }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }
  function icon(name) {
    var s = svgEl("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" });
    s.appendChild(svgEl("path", { d: ICONS[name] || ICONS.info }));
    return s;
  }
  function safeHref(url) {
    try {
      var parsed = new URL(url, window.location.href);
      return (parsed.protocol === "http:" || parsed.protocol === "https:") ? parsed.href : null;
    } catch (e) { return null; }
  }
  function linkButton(url, label, iconName, cls) {
    var href = safeHref(url);
    if (!href) return null;
    var a = el("a", "btn " + (cls || ""));
    a.href = href;
    if (iconName) a.appendChild(icon(iconName));
    a.appendChild(el("span", null, label));
    if (href.indexOf(window.location.origin) !== 0) { a.target = "_blank"; a.rel = "noopener noreferrer"; }
    return a;
  }

  var grouped = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
  function isYear(v) { return typeof v === "number" && v % 1 === 0 && v >= 1900 && v <= 2100; }
  function fmt(v) {
    if (typeof v !== "number" || !isFinite(v)) return v === null || v === undefined ? "" : String(v);
    if (Math.abs(v) >= 1000) return grouped.format(Math.round(v));
    return grouped.format(v);
  }
  function compact(v) {
    var a = Math.abs(v);
    function one(x) { return new Intl.NumberFormat("en-US", { maximumFractionDigits: x >= 100 ? 0 : 1 }).format(x); }
    if (a >= 1e9) return one(v / 1e9) + " مليار";
    if (a >= 1e6) return one(v / 1e6) + " مليون";
    if (a >= 1e4) return one(v / 1e3) + " ألف";
    return fmt(v);
  }

  // -- theme ----------------------------------------------------------
  function applyTheme(theme) {
    if (theme === "light" || theme === "dark") document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
  }
  function effectiveTheme() {
    var set = document.documentElement.getAttribute("data-theme");
    if (set) return set;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  applyTheme(load("localStorage", "masdar-theme"));
  document.getElementById("theme").addEventListener("click", function () {
    var next = effectiveTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    save("localStorage", "masdar-theme", next);
  });

  // -- landing --------------------------------------------------------
  function placeComposer(landing) {
    if (landing) { heroSlot.appendChild(form); body.classList.add("is-landing"); }
    else { dock.insertBefore(form, dock.firstChild); body.classList.remove("is-landing"); }
  }
  placeComposer(true);
  if (window.matchMedia && window.matchMedia("(max-width: 560px)").matches) {
    input.placeholder = "اسأل عن إحصاء… مثلاً: عدد السكان 2024";
  }

  function stat(list, value, text) {
    var li = el("li");
    var strong = el("strong");
    var number = el("bdi", null, value);
    number.setAttribute("dir", "ltr");
    strong.appendChild(number);
    li.appendChild(strong);
    li.appendChild(el("span", null, text));
    list.appendChild(li);
  }

  function renderOverview(data) {
    var stats = document.getElementById("stats");
    var s = data.stats || {};
    stat(stats, fmt(s.sources || 0), "مصادر بواجهات مؤكَّدة");
    stat(stats, fmt(s.publishers || 0), "جهة حكومية ومصدراً");
    stat(stats, fmt(Math.floor((s.datasets || 0) / 100) * 100) + "+", "مجموعة بيانات ومؤشر");
    stat(stats, fmt(s.topics || 0), "موضوعاً إحصائياً");

    var examples = document.getElementById("examples");
    (data.examples || []).forEach(function (e) {
      var b = el("button", "example");
      b.type = "button";
      b.appendChild(el("strong", null, e.q));
      if (e.shows) b.appendChild(el("span", null, e.shows));
      b.addEventListener("click", function () { ask(e.q); });
      examples.appendChild(b);
    });

    var sources = document.getElementById("sources");
    (data.sources || []).forEach(function (src) {
      sourceKinds[src.id] = src.kind;
      var card = el("div", "source" + (src.kind === "دولي" ? " kind-intl" : src.kind === "لحظي" ? " kind-live" : ""));
      var ic = el("span", "source-icon");
      ic.appendChild(icon(src.kind === "دولي" ? "globe" : src.kind === "لحظي" ? "bolt" : "database"));
      card.appendChild(ic);
      var b = el("div", "source-body");
      b.appendChild(el("strong", null, src.name));
      var meta = el("div", "source-meta");
      meta.appendChild(el("span", "tag " + (KIND_TAG[src.kind] || ""), src.kind));
      if (src.count) meta.appendChild(el("span", null, fmt(src.count) + " " + src.unit));
      if (src.operator && src.operator !== src.name) meta.appendChild(el("span", null, src.operator));
      b.appendChild(meta);
      card.appendChild(b);
      sources.appendChild(card);
    });

    var ai = document.getElementById("ai-badge");
    if (data.llm) {
      var on = data.llm.indexOf("مفعّل") === 0;
      ai.textContent = on ? "الفهم الذكي مفعّل" : "الفهم بالقواعد";
      ai.title = "الذكاء الاصطناعي: " + data.llm;
      ai.className = "badge" + (on ? "" : " badge-muted");
      ai.hidden = false;
    }
    if (data.demo) {
      var mode = document.getElementById("mode-badge");
      mode.textContent = "مصادر تجريبية مضافة";
      mode.className = "badge badge-gold";
      mode.hidden = false;
    }
    if (data.locked) showUnlock();
  }

  fetch("/api/overview", { credentials: "same-origin" })
    .then(function (r) { return r.json(); })
    .then(renderOverview)
    .catch(function () { /* the page still works without the overview */ });

  // -- access code ----------------------------------------------------
  var unlockDialog = document.getElementById("unlock");
  function showUnlock() {
    if (!unlockDialog || unlockDialog.open) return;
    document.getElementById("unlock-error").hidden = true;
    if (typeof unlockDialog.showModal === "function") unlockDialog.showModal();
    else unlockDialog.setAttribute("open", "");
    document.getElementById("unlock-code").focus();
  }
  document.getElementById("unlock-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var code = document.getElementById("unlock-code").value;
    fetch("/api/unlock", {
      method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: code })
    }).then(function (r) {
      if (r.ok) { unlockDialog.close(); input.focus(); }
      else document.getElementById("unlock-error").hidden = false;
    }).catch(function () { document.getElementById("unlock-error").hidden = false; });
  });
  unlockDialog.addEventListener("cancel", function (e) { e.preventDefault(); });

  // -- charts ---------------------------------------------------------
  function niceStep(span) {
    var raw = span / 4;
    var mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    var n = raw / mag;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag;
  }

  // Text in the charts is Arabic or carries Arabic units ("1.3 مليون"), so
  // every label is laid out right-to-left and anchored by its right edge
  // ("start" in RTL); the plot itself keeps a left-to-right geometry.
  function label(attrs, text, cls) {
    var t = svgEl("text", Object.assign({ direction: "rtl", "class": cls }, attrs));
    t.textContent = text;
    return t;
  }

  function categoryChart(chart, W) {
    var points = chart.points;
    var narrow = W < 480;
    var rowH = narrow ? 28 : 30, labelW = Math.round(Math.min(170, W * (narrow ? 0.34 : 0.26)));
    var valueW = narrow ? 70 : 84, top = 4;
    var H = top + points.length * rowH;
    var max = Math.max.apply(null, points.map(function (p) { return Math.abs(p.value); })) || 1;
    var right = W - labelW - 8;
    var barMax = right - valueW;
    var s = svgEl("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img",
      "aria-label": chart.measure + (chart.note ? " — " + chart.note : "") });
    var maxChars = Math.max(8, Math.floor(labelW / 7.5));
    points.forEach(function (p, i) {
      var y = top + i * rowH;
      var w = Math.max(2, Math.abs(p.value) / max * barMax);
      s.appendChild(svgEl("rect", { x: right - barMax, y: y + 6, width: barMax, height: rowH - 12, rx: 4, "class": "bar-bg" }));
      var bar = svgEl("rect", { x: right - w, y: y + 6, width: w, height: rowH - 12, rx: 4, "class": "bar" });
      var title = svgEl("title");
      title.textContent = p.label + ": " + fmt(p.value);
      bar.appendChild(title);
      s.appendChild(bar);
      var text = p.label.length > maxChars ? p.label.slice(0, maxChars - 1) + "…" : p.label;
      s.appendChild(label({ x: W - 2, y: y + rowH / 2 + 4, "text-anchor": "start" }, text, "bar-label"));
      s.appendChild(label({ x: right - barMax - 6, y: y + rowH / 2 + 4, "text-anchor": "start" },
        compact(p.value), "bar-value"));
    });
    return s;
  }

  function yearChart(chart, highlight, W) {
    var points = chart.points;
    var narrow = W < 480;
    var H = narrow ? 210 : 250, left = narrow ? 58 : 70, right = 8, top = 20, bottom = 30;
    var values = points.map(function (p) { return p.value; });
    var lo = Math.min(0, Math.min.apply(null, values));
    var hi = Math.max(0, Math.max.apply(null, values));
    var step = niceStep(hi - lo || 1);
    lo = Math.floor(lo / step) * step;
    hi = Math.ceil(hi / step) * step || step;
    var plotH = H - top - bottom, plotW = W - left - right;
    function y(v) { return top + (hi - v) / (hi - lo) * plotH; }
    var s = svgEl("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img",
      "aria-label": chart.measure + " حسب السنة" });
    for (var t = lo; t <= hi + step / 2; t += step) {
      s.appendChild(svgEl("line", { x1: left, x2: W - right, y1: y(t), y2: y(t), "class": t === 0 ? "axis" : "grid" }));
      s.appendChild(label({ x: left - 8, y: y(t) + 4, "text-anchor": "start" }, compact(t), "tick"));
    }
    var slot = plotW / points.length;
    var bw = Math.min(46, slot * 0.64);
    var every = Math.ceil(points.length / (narrow ? 6 : 12));
    var showValues = points.length <= (narrow ? 6 : 12);
    points.forEach(function (p, i) {
      var cx = left + slot * i + slot / 2;
      var y0 = y(0), y1 = y(p.value);
      var marked = highlight.indexOf(Number(p.label)) !== -1;
      var col = svgEl("rect", { x: cx - bw / 2, y: Math.min(y0, y1), width: bw,
        height: Math.max(1, Math.abs(y1 - y0)), rx: 3,
        "class": highlight.length && !marked ? "col col-dim" : "col" });
      var title = svgEl("title");
      title.textContent = p.label + ": " + fmt(p.value);
      col.appendChild(title);
      s.appendChild(col);
      // Counted from the latest year, so it is always labelled and the
      // labels stay evenly spaced.
      if ((points.length - 1 - i) % every === 0) {
        s.appendChild(label({ x: cx, y: H - 8, "text-anchor": "middle" }, p.label, "tick"));
      }
      if (showValues) {
        s.appendChild(label({ x: cx, y: Math.min(y0, y1) - 5, "text-anchor": "middle" },
          compact(p.value), "bar-value"));
      }
    });
    return s;
  }

  function renderCharts(charts, highlight) {
    var wrap = el("div", "chart");
    var head = el("div", "chart-head");
    var title = el("strong");
    var note = el("span");
    head.appendChild(title);
    head.appendChild(note);
    var canvas = el("div", "chart-canvas");
    var current = 0, drawnWidth = 0;
    // Drawn at the width it is shown at, so text stays legible on a phone.
    function draw() {
      var W = Math.round(canvas.clientWidth) || 640;
      if (Math.abs(W - drawnWidth) < 4 && canvas.firstChild) return;
      drawnWidth = W;
      var c = charts[current];
      canvas.textContent = "";
      canvas.appendChild(c.kind === "years" ? yearChart(c, highlight, W) : categoryChart(c, W));
    }
    function show(i) {
      current = i;
      var c = charts[i];
      title.textContent = c.measure;
      note.textContent = c.kind === "years" ? "حسب السنة" : (c.note || "");
      drawnWidth = 0;
      draw();
      Array.prototype.forEach.call(measures.children, function (b, j) {
        b.setAttribute("aria-pressed", String(i === j));
      });
    }
    var measures = el("div", "measures");
    if (charts.length > 1) {
      charts.forEach(function (c, i) {
        var b = el("button", "measure", c.measure);
        b.type = "button";
        b.addEventListener("click", function () { show(i); });
        measures.appendChild(b);
      });
      wrap.appendChild(measures);
    }
    wrap.appendChild(head);
    wrap.appendChild(canvas);
    show(0);
    if (typeof ResizeObserver === "function") new ResizeObserver(draw).observe(canvas);
    return wrap;
  }

  // -- table preview --------------------------------------------------
  function renderTable(preview) {
    var box = el("div");
    var scroll = el("div", "table-wrap");
    var table = el("table");
    var thead = el("thead");
    var hr = el("tr");
    preview.columns.forEach(function (c) { hr.appendChild(el("th", null, c)); });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = el("tbody");
    preview.rows.forEach(function (row) {
      var tr = el("tr");
      preview.columns.forEach(function (_, i) {
        var v = row[i];
        var numeric = typeof v === "number";
        tr.appendChild(el("td", numeric ? "n" : null, numeric && !isYear(v) ? fmt(v) : v));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    scroll.appendChild(table);
    box.appendChild(scroll);
    var shown = preview.rows.length;
    box.appendChild(el("p", "table-note", shown < preview.total_rows
      ? "أول " + fmt(shown) + " صفوف من أصل " + fmt(preview.total_rows) + " — الملف الكامل في الإكسل."
      : fmt(preview.total_rows) + " صفوف — كلها في ملف الإكسل."));
    return box;
  }

  function tabs(items) {
    var wrap = el("div");
    var bar = el("div", "tabs");
    bar.setAttribute("role", "tablist");
    var panel = el("div", "panel");
    panel.setAttribute("role", "tabpanel");
    var buttons = items.map(function (item, i) {
      var b = el("button", "tab", item.label);
      b.type = "button";
      b.setAttribute("role", "tab");
      b.addEventListener("click", function () { select(i); });
      bar.appendChild(b);
      return b;
    });
    function select(i) {
      buttons.forEach(function (b, j) { b.setAttribute("aria-selected", String(i === j)); });
      panel.textContent = "";
      panel.appendChild(items[i].render());
    }
    wrap.appendChild(bar);
    wrap.appendChild(panel);
    select(0);
    return wrap;
  }

  // -- answers --------------------------------------------------------
  function fact(dl, label, value, plain) {
    var box = el("div", "fact");
    box.appendChild(el("dt", null, label));
    box.appendChild(el("dd", plain ? "plain" : null, value === null || value === undefined || value === "" ? "—" : value));
    dl.appendChild(box);
  }
  // 2014، 2015، 2016، 2019 -> "2014–2016، 2019"
  function yearsText(years) {
    var sorted = years.slice().sort(function (a, b) { return a - b; });
    var parts = [];
    for (var i = 0; i < sorted.length; i++) {
      var start = sorted[i];
      while (i + 1 < sorted.length && sorted[i + 1] === sorted[i] + 1) i++;
      parts.push(sorted[i] - start >= 2 ? start + "–" + sorted[i]
        : sorted[i] === start ? String(start) : start + "، " + sorted[i]);
    }
    return parts.join("، ");
  }
  function banner(tone, iconName, text) {
    var b = el("div", "banner tone-" + tone);
    b.appendChild(icon(iconName));
    b.appendChild(el("div", null, text));
    return b;
  }
  function kindOf(f) {
    if (f.international) return "دولي";
    return sourceKinds[f.source_id] || null;
  }

  function renderResult(f, highlight) {
    var sec = el("section", "result");
    var head = el("div", "result-title");
    head.appendChild(el("h3", null, f.title));
    var pub = el("div", "result-publisher");
    pub.appendChild(el("span", null, f.publisher || "—"));
    var kind = kindOf(f);
    if (kind) pub.appendChild(el("span", "tag " + (KIND_TAG[kind] || ""), kind));
    head.appendChild(pub);
    sec.appendChild(head);

    var dl = el("dl", "facts");
    fact(dl, "آخر تحديث معلن", f.last_updated || "غير معلن");
    fact(dl, "أحدث سنة في البيانات", f.latest_year);
    fact(dl, "السنوات المطابقة لطلبك", f.matched_years.length ? yearsText(f.matched_years) : "لا شيء");
    fact(dl, "التحقق من السنوات", EVIDENCE[f.evidence] || f.evidence, true);
    sec.appendChild(dl);

    if (f.stale) sec.appendChild(banner("notice", "archive",
      "تعذّر الوصول إلى المصدر الآن، فهذه آخر نسخة محفوظة منه. قد تكون نُشرت بيانات أحدث."));
    if (f.international) sec.appendChild(banner("info", "globe",
      "مصدر دولي: يُعرض لأن الجهات السعودية لم تُرجع هذه السنة أو هذا المؤشر، وقد يختلف عن رقمها الرسمي."));

    var actions = el("div", "actions");
    if (f.download) {
      var d = linkButton(f.download, "تنزيل ملف الإكسل", "download", "btn-primary");
      if (d) { d.setAttribute("download", ""); actions.appendChild(d); }
    }
    var page = linkButton(f.landing_url, "صفحة المصدر", "external");
    if (page) actions.appendChild(page);
    var raw = linkButton(f.resource_url, "البيانات الأصلية", "file");
    if (raw) actions.appendChild(raw);
    if (actions.childNodes.length) sec.appendChild(actions);

    var items = [];
    var p = f.preview;
    if (p && p.charts && p.charts.length) items.push({ label: "رسم بياني", render: function () { return renderCharts(p.charts, highlight); } });
    if (p && p.columns && p.columns.length) items.push({ label: "معاينة البيانات", render: function () { return renderTable(p); } });
    if (f.notes && f.notes.length) {
      items.push({ label: "ملاحظات (" + f.notes.length + ")", render: function () {
        var ul = el("ul", "notes");
        f.notes.forEach(function (n) { ul.appendChild(el("li", null, n)); });
        return ul;
      } });
    }
    if (items.length) sec.appendChild(tabs(items));
    return sec;
  }

  function renderOther(f) {
    var box = el("div", "other");
    box.appendChild(el("strong", null, f.title));
    var meta = el("div", "other-meta");
    meta.appendChild(el("span", null, f.publisher || ""));
    var kind = kindOf(f);
    if (kind) meta.appendChild(el("span", "tag " + (KIND_TAG[kind] || ""), kind));
    meta.appendChild(el("span", null, f.verdict_label));
    if (f.latest_year) meta.appendChild(el("span", null, "أحدث سنة: " + f.latest_year));
    box.appendChild(meta);
    var actions = el("div", "actions");
    if (f.download) {
      var d = linkButton(f.download, "الإكسل", "download");
      if (d) { d.setAttribute("download", ""); actions.appendChild(d); }
    }
    var page = linkButton(f.landing_url, "المصدر", "external");
    if (page) actions.appendChild(page);
    if (actions.childNodes.length) box.appendChild(actions);
    return box;
  }

  function understoodChips(u) {
    var row = el("div", "understood");
    row.appendChild(el("span", "label", u.follow_up ? "متابعة للسؤال السابق:" : "فهمت طلبك:"));
    function chip(label, value) {
      var c = el("span", "chip");
      c.appendChild(document.createTextNode(label + " "));
      c.appendChild(el("b", null, value));
      row.appendChild(c);
    }
    if (u.topic) chip("الموضوع", u.topic);
    chip("الفترة", u.period);
    if (u.dimensions && u.dimensions.length) chip("التفصيل", u.dimensions.join("، "));
    if (u.method === "llm") row.appendChild(el("span", "chip chip-ai", "بمساعدة الذكاء الاصطناعي"));
    return row;
  }

  function renderAnswer(data) {
    var box = el("article", "answer");
    var head = el("div", "answer-head");
    var v = el("span", "verdict tone-" + (TONE[data.verdict] || "info"));
    v.appendChild(icon(VERDICT_ICON[data.verdict] || "info"));
    v.appendChild(el("span", null, data.verdict_label));
    head.appendChild(v);
    head.appendChild(el("p", "headline", data.headline.replace(/^[^؀-ۿA-Za-z0-9«]+/, "")));
    head.appendChild(understoodChips(data.understood));
    if (data.understood.method === "llm" && data.understood.note) {
      head.appendChild(el("p", "restatement", "«" + data.understood.note + "»"));
    }
    box.appendChild(head);

    var primary = data.findings[0];
    if (primary) {
      box.appendChild(renderResult(primary, primary.matched_years || []));
    } else if (data.message) {
      var rest = data.message.split("\n").slice(1).join("\n").trim();
      if (rest) box.appendChild(el("div", "message-body", rest));
    }

    if (data.suggestions.length) {
      var sug = el("div", "suggest");
      sug.appendChild(el("div", "label", "المتاح بدلاً من ذلك — اضغط لتجهيز الملف:"));
      var row = el("div", "suggest-row");
      data.suggestions.forEach(function (s) {
        var b = el("button", "suggestion");
        b.type = "button";
        b.appendChild(el("strong", null, "سنة " + s.year));
        b.appendChild(el("span", null, s.reason + (s.source ? " — " + s.source : "")));
        b.addEventListener("click", function () { ask(s.ask); });
        row.appendChild(b);
      });
      sug.appendChild(row);
      box.appendChild(sug);
    }

    var foot = el("div", "foot");
    if (data.consulted && data.consulted.length) {
      var answered = data.consulted.filter(function (c) { return c.found !== null; }).length;
      var det = el("details");
      det.appendChild(el("summary", null, "أين بحثت: " + data.consulted.length + " مصادر"
        + (answered < data.consulted.length ? " (" + (data.consulted.length - answered) + " لم تُجب)" : "")));
      var inner = el("div");
      var chips = el("div", "consulted");
      data.consulted.forEach(function (c) {
        var ch = el("span", "chip " + (c.found === null ? "fail" : "ok"));
        ch.appendChild(document.createTextNode(c.source + " "));
        ch.appendChild(el("b", null, c.found === null ? "تعذّر" : String(c.found)));
        chips.appendChild(ch);
      });
      inner.appendChild(chips);
      if (data.source_errors.length) {
        var ul = el("ul", "notes");
        data.source_errors.forEach(function (e) { ul.appendChild(el("li", null, e.source + ": " + e.reason)); });
        inner.appendChild(ul);
      }
      det.appendChild(inner);
      foot.appendChild(det);
    }
    if (data.findings.length > 1) {
      var others = el("details");
      others.appendChild(el("summary", null, "نتائج أخرى ذات صلة (" + (data.findings.length - 1) + ")"));
      var list = el("div");
      data.findings.slice(1).forEach(function (f) { list.appendChild(renderOther(f)); });
      others.appendChild(list);
      foot.appendChild(others);
    }
    if (typeof data.seconds === "number") {
      var meta = el("div", "meta-line");
      meta.appendChild(el("span", null, "استغرق " + data.seconds.toFixed(1) + " ث"));
      if (primary && primary.retrieved_at) meta.appendChild(el("span", null, "وقت الاستخراج: " + primary.retrieved_at.slice(0, 16).replace("T", " ") + " UTC"));
      foot.appendChild(meta);
    }
    if (foot.childNodes.length) box.appendChild(foot);
    return box;
  }

  function renderError(text) {
    var box = el("article", "answer");
    var head = el("div", "answer-head");
    var v = el("span", "verdict tone-bad");
    v.appendChild(icon("alert"));
    v.appendChild(el("span", null, "تعذّرت الإجابة"));
    head.appendChild(v);
    head.appendChild(el("p", "headline", text));
    box.appendChild(head);
    return box;
  }

  function renderThinking() {
    var box = el("article", "answer");
    var t = el("div", "thinking");
    var line = el("div", "thinking-line");
    line.appendChild(el("span", "spinner"));
    var label = el("span", null, "أبحث في المصادر الرسمية وأتحقق من الملفات…");
    line.appendChild(label);
    t.appendChild(line);
    t.appendChild(el("div", "skeleton w70"));
    t.appendChild(el("div", "skeleton w85"));
    t.appendChild(el("div", "skeleton w45"));
    box.appendChild(t);
    var started = Date.now();
    var timer = window.setInterval(function () {
      var s = Math.round((Date.now() - started) / 1000);
      label.textContent = "أبحث في المصادر الرسمية وأتحقق من الملفات… " + s + " ث";
    }, 1000);
    box.stop = function () { window.clearInterval(timer); };
    return box;
  }

  // -- conversation ---------------------------------------------------
  session = load("sessionStorage", "masdar-session");

  function setBusy(value) {
    busy = value;
    send.disabled = value;
  }

  function ask(message) {
    message = (message || "").trim();
    if (!message || busy) return;
    placeComposer(false);
    thread.appendChild(el("div", "msg-user", message));
    var waiting = renderThinking();
    thread.appendChild(waiting);
    waiting.scrollIntoView({ block: "end" });
    setBusy(true);

    fetch("/api/ask", {
      method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: session, message: message })
    })
      .then(function (r) { return r.json().then(function (b) { return { status: r.status, ok: r.ok, body: b }; }); })
      .then(function (res) {
        waiting.stop();
        waiting.remove();
        var node;
        if (res.status === 401) {
          node = renderError("أدخل رمز الدخول ثم أعد السؤال.");
          showUnlock();
        } else if (!res.ok || res.body.error) {
          node = renderError(res.body.error || "تعذّر الحصول على إجابة.");
        } else {
          session = res.body.session;
          save("sessionStorage", "masdar-session", session);
          node = renderAnswer(res.body);
        }
        thread.appendChild(node);
        node.scrollIntoView({ block: "start" });
      })
      .catch(function () {
        waiting.stop();
        waiting.remove();
        thread.appendChild(renderError("تعذّر الاتصال بالخادم. تأكد أن نافذة «مصدر» ما زالت تعمل."));
      })
      .then(function () { setBusy(false); input.focus({ preventScroll: true }); });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = input.value;
    input.value = "";
    input.style.height = "";
    ask(text);
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (typeof form.requestSubmit === "function") form.requestSubmit();
      else form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });
  input.addEventListener("input", function () {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 180) + "px";
  });
  document.getElementById("reset").addEventListener("click", function () {
    fetch("/api/reset", {
      method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: session })
    }).catch(function () {});
    save("sessionStorage", "masdar-session", null);
    session = null;
    thread.textContent = "";
    placeComposer(true);
    scroller.scrollTop = 0;
    input.focus();
  });
  input.focus();
})();
