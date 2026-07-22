"use strict";
// Infinite DJ web player. Consumes /timeline.json (clips + track metadata) and
// drives a live-looking dashboard synced to <audio> playback. The PlayerState(t)
// computed here is the same shape the real-time engine can emit later.

const $ = (id) => document.getElementById(id);
const audio = $("audio");
let TL = null;              // timeline data
let TRACKS = {};           // id -> track meta
const wheelWedges = {};    // camelot code -> <path>
let lastKey = null, lastInc = null;

const fmt = (s) => {
  s = Math.max(0, s | 0);
  return `${(s / 60) | 0}:${String(s % 60).padStart(2, "0")}`;
};

// ── Camelot helpers ──────────────────────────────────────────────────────────
const parseKey = (k) => k && /^\d{1,2}[AB]$/.test(k)
  ? { num: parseInt(k), letter: k.slice(-1) } : null;
function camelotRel(a, b) {          // relationship class of b to current a
  const pa = parseKey(a), pb = parseKey(b);
  if (!pa || !pb) return "";
  if (pa.num === pb.num && pa.letter === pb.letter) return "cur";
  if (pa.num === pb.num) return "rel";                       // relative maj/min
  const d = Math.min(Math.abs(pa.num - pb.num), 12 - Math.abs(pa.num - pb.num));
  if (pa.letter === pb.letter && d === 1) return "adj";      // ±1 on the wheel
  return "";
}

// ── Build the Camelot wheel (24 wedges) ──────────────────────────────────────
function wedgePath(rIn, rOut, a0, a1) {
  const p = (r, a) => [r * Math.cos(a), r * Math.sin(a)];
  const [x0, y0] = p(rOut, a0), [x1, y1] = p(rOut, a1);
  const [x2, y2] = p(rIn, a1), [x3, y3] = p(rIn, a0);
  const large = a1 - a0 > Math.PI ? 1 : 0;
  return `M${x0} ${y0}A${rOut} ${rOut} 0 ${large} 1 ${x1} ${y1}`
       + `L${x2} ${y2}A${rIn} ${rIn} 0 ${large} 0 ${x3} ${y3}Z`;
}
function buildWheel() {
  const svg = $("wheel");
  const SVGNS = "http://www.w3.org/2000/svg";
  for (let n = 1; n <= 12; n++) {
    const a0 = (n - 1) / 12 * 2 * Math.PI - Math.PI / 2 - Math.PI / 12;
    const a1 = a0 + Math.PI / 6;
    const mid = (a0 + a1) / 2;
    const hue = (n - 1) / 12 * 360;
    for (const [letter, rIn, rOut] of [["A", 44, 74], ["B", 74, 100]]) {
      const code = `${n}${letter}`;
      const path = document.createElementNS(SVGNS, "path");
      path.setAttribute("d", wedgePath(rIn, rOut, a0, a1));
      path.setAttribute("class", "wedge");
      path.dataset.hue = hue;
      path.dataset.code = code;
      path.style.fill = `hsl(${hue} 55% ${letter === "A" ? 34 : 46}%)`;
      path.style.opacity = 0.35;
      svg.appendChild(path);
      wheelWedges[code] = path;
      const lr = (rIn + rOut) / 2;
      const label = document.createElementNS(SVGNS, "text");
      label.setAttribute("x", lr * Math.cos(mid));
      label.setAttribute("y", lr * Math.sin(mid));
      label.setAttribute("class", "wlabel");
      label.textContent = code;
      svg.appendChild(label);
    }
  }
}
function paintWheel(curKey, incKey) {
  if (curKey === lastKey && incKey === lastInc) return;
  lastKey = curKey; lastInc = incKey;
  for (const [code, el] of Object.entries(wheelWedges)) {
    const rel = camelotRel(curKey, code);
    let op = 0.28, stroke = "var(--panel)", sw = 1.2;
    if (code === incKey) { op = 1; stroke = "var(--accent)"; sw = 2.4; }
    else if (rel === "cur") { op = 1; stroke = "#fff"; sw = 2.4; }
    else if (rel === "rel") op = 0.7;
    else if (rel === "adj") op = 0.55;
    el.style.opacity = op;
    el.style.stroke = stroke;
    el.style.strokeWidth = sw;
  }
}

// ── PlayerState from clips ───────────────────────────────────────────────────
function clipGain(c, t) {
  if (t < c.start || t >= c.end) return 0;
  let g = 1;
  if (c.fade_in > 0 && t < c.start + c.fade_in) g = (t - c.start) / c.fade_in;
  if (c.fade_out > 0 && t > c.end - c.fade_out)
    g = Math.min(g, (c.end - t) / c.fade_out);
  return Math.max(0, Math.min(1, g));
}
function playerState(t) {
  const act = [];
  for (const c of TL.clips) {
    if (c.start <= t && t < c.end) act.push({ c, g: clipGain(c, t) });
  }
  // Primary = loudest; tie-break toward the most recently started (fading in).
  act.sort((a, b) => (b.g - a.g) || (b.c.start - a.c.start));
  const byStart = [...act].sort((a, b) => a.c.start - b.c.start);
  const primary = act[0] || null;
  // Incoming = newest clip still within its fade-in, with something else present.
  let transition = null;
  const newest = byStart[byStart.length - 1];
  if (newest && newest.c.fade_in > 0 && t < newest.c.start + newest.c.fade_in
      && byStart.length > 1) {
    transition = {
      to: newest.c,
      progress: (t - newest.c.start) / newest.c.fade_in,
      mode: newest.c.mode,
    };
  }
  let upcoming = null, best = Infinity;
  for (const c of TL.clips)
    if (c.start > t && c.start < best) { best = c.start; upcoming = c; }
  return { active: act, primary, transition, upcoming, t };
}

// ── Render ───────────────────────────────────────────────────────────────────
function render(t) {
  if (!TL) return;
  const st = playerState(t);
  const p = st.primary ? st.primary.c : null;

  if (p) {
    const tr = TRACKS[p.track] || {};
    document.documentElement.style.setProperty("--now", tr.color || "#6ea8fe");
    $("now-title").textContent = tr.title || "—";
    $("now-meta").textContent = `${(p.key || tr.key || "")}  ·  ${Math.round(p.bpm || tr.bpm || 0)} BPM`;
    $("now-section").textContent = p.section ? p.section : "";
    $("now-mode").textContent = st.transition ? st.transition.mode : (p.mode || "playing");
    // energy meter from the primary track
    $("energy-fill").style.width = `${Math.round((tr.energy ?? 0.5) * 100)}%`;
  }

  // Other active layers (collage) — only ones actually audible
  const others = st.active.slice(1).filter((x) => x.g > 0.1).slice(0, 4);
  $("layers").innerHTML = others.map(({ c }) => {
    const tr = TRACKS[c.track] || {};
    return `<span class="chip"><span class="dot" style="background:${tr.color}"></span>${tr.title || ""}</span>`;
  }).join("");

  // Crossfade ring
  const xf = $("xfade");
  if (st.transition) {
    xf.classList.remove("hidden");
    const pct = Math.round(st.transition.progress * 100);
    $("ring-fg").style.strokeDashoffset = 327 * (1 - st.transition.progress);
    $("xfade-pct").textContent = `${pct}%`;
    $("xfade-mode").textContent = st.transition.mode;
  } else xf.classList.add("hidden");

  // Camelot wheel
  paintWheel(p ? (p.key) : null, st.transition ? st.transition.to.key : null);

  // Up next
  if (st.upcoming) {
    const tr = TRACKS[st.upcoming.track] || {};
    $("upnext").innerHTML = `Up next · <b>${tr.title || ""}</b> in ${Math.max(0, st.upcoming.start - t) | 0}s`;
  } else $("upnext").textContent = "";

  drawTimeline(t);
}

// ── Mini arrangement timeline (canvas) ───────────────────────────────────────
function drawTimeline(t) {
  const cv = $("tlcanvas"), ctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const dur = TL.duration || 1;
  // lane packing: greedily assign clips to rows that are free
  const lanes = [];
  const laneOf = new Map();
  for (const c of TL.clips) {
    let i = 0;
    for (; i < lanes.length; i++) if (lanes[i] <= c.start) break;
    lanes[i] = c.end; laneOf.set(c, i);
  }
  const rows = Math.max(1, lanes.length);
  const rh = Math.min(14, (H - 6) / rows);
  for (const c of TL.clips) {
    const x = (c.start / dur) * W, w = Math.max(2, ((c.end - c.start) / dur) * W);
    const y = 3 + laneOf.get(c) * rh;
    const tr = TRACKS[c.track] || {};
    const on = t >= c.start && t < c.end;
    ctx.globalAlpha = on ? 0.95 : 0.5;
    ctx.fillStyle = tr.color || "#6ea8fe";
    roundRect(ctx, x, y, w, rh - 2, 3); ctx.fill();
  }
  ctx.globalAlpha = 1;
  const px = (t / dur) * W;
  ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
}
function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// ── Transport ────────────────────────────────────────────────────────────────
function loop() {
  const t = audio.currentTime;
  $("pos").textContent = fmt(t);
  $("ttime").textContent = fmt(t);
  const frac = TL ? Math.min(1, t / (TL.duration || audio.duration || 1)) : 0;
  $("scrub-fill").style.width = `${frac * 100}%`;
  $("scrub-head").style.left = `${frac * 100}%`;
  render(t);
  requestAnimationFrame(loop);
}
$("playbtn").onclick = () => audio.paused ? audio.play() : audio.pause();
audio.addEventListener("play", () => $("playbtn").textContent = "❚❚");
audio.addEventListener("pause", () => $("playbtn").textContent = "▶");
$("scrub").onclick = (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const frac = (e.clientX - r.left) / r.width;
  audio.currentTime = frac * (TL.duration || audio.duration || 0);
};

// ── Boot ─────────────────────────────────────────────────────────────────────
buildWheel();
fetch("/timeline.json").then((r) => r.json()).then((data) => {
  TL = data; TRACKS = data.tracks || {};
  $("dur").textContent = fmt(data.duration);
  requestAnimationFrame(loop);
}).catch((e) => { $("now-title").textContent = "Failed to load timeline"; console.error(e); });
