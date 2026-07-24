"use strict";
// Infinite DJ web player. Consumes /timeline.json (clips + track metadata) and
// drives a live-looking dashboard synced to <audio> playback. The PlayerState(t)
// computed here is the same shape the real-time engine can emit later.

const $ = (id) => document.getElementById(id);
const audio = $("audio");
let TL = null;              // timeline data
let TRACKS = {};           // id -> track meta

const fmt = (s) => {
  s = Math.max(0, s | 0);
  return `${(s / 60) | 0}:${String(s % 60).padStart(2, "0")}`;
};

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
  // Previous: the clip that started most recently before the primary clip.
  let prev = null;
  if (primary) {
    let bestStart = -Infinity;
    for (const c of TL.clips)
      if (c.start < primary.c.start && c.start > bestStart && c.track !== primary.c.track) {
        bestStart = c.start; prev = c;
      }
  }
  return { active: act, primary, transition, upcoming, prev, t };
}

// ── Render ───────────────────────────────────────────────────────────────────
function render(t) {
  if (!TL) return;
  const st = playerState(t);
  const p = st.primary ? st.primary.c : null;

  if (p) {
    const tr = TRACKS[p.track] || {};
    $("cur-title").textContent = tr.title || "—";
    $("cur-bpm").textContent = `${Math.round(p.bpm || tr.bpm || 0)} BPM`;
  }

  const prevTr = st.prev ? (TRACKS[st.prev.track] || {}) : null;
  $("prev-title").textContent = prevTr ? (prevTr.title || "") : "—";
  const nextTr = st.upcoming ? (TRACKS[st.upcoming.track] || {}) : null;
  $("next-title").textContent = nextTr ? (nextTr.title || "") : "—";

  // Crossfade fill, inline between current and next
  const xf = $("xfade");
  if (st.transition) {
    xf.classList.remove("hidden");
    const pct = Math.round(st.transition.progress * 100);
    $("xfade-fill").style.width = `${pct}%`;
    $("xfade-pct").textContent = `MIXING ${pct}%`;
  } else xf.classList.add("hidden");
}

// ── Stereo power meter (Web Audio API) ───────────────────────────────────────
let actx = null, analyserL = null, analyserR = null, meterBufL = null, meterBufR = null;
let peakL = 0, peakR = 0;

function ensureMeterGraph() {
  if (actx) return;
  try {
    actx = new (window.AudioContext || window.webkitAudioContext)();
    const src = actx.createMediaElementSource(audio);
    src.connect(actx.destination);           // keep audio audible
    const splitter = actx.createChannelSplitter(2);
    src.connect(splitter);
    analyserL = actx.createAnalyser(); analyserL.fftSize = 512;
    analyserR = actx.createAnalyser(); analyserR.fftSize = 512;
    splitter.connect(analyserL, 0);
    splitter.connect(analyserR, 1);
    meterBufL = new Uint8Array(analyserL.fftSize);
    meterBufR = new Uint8Array(analyserR.fftSize);
  } catch (e) {
    console.warn("Stereo meter unavailable:", e);
    actx = null;
  }
}

function rms(analyser, buf) {
  analyser.getByteTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) {
    const v = (buf[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / buf.length);
}

const METER_FLOOR_DB = -60;
function dbMeterLevel(value) {
  const db = 20 * Math.log10(Math.max(value, 0.000001));
  return Math.max(0, Math.min(1, (db - METER_FLOOR_DB) / -METER_FLOOR_DB));
}

let smoothL = 0, smoothR = 0;
function updateMeters() {
  let lvL = 0, lvR = 0;
  if (analyserL && analyserR && !audio.paused) {
    lvL = dbMeterLevel(rms(analyserL, meterBufL));
    lvR = dbMeterLevel(rms(analyserR, meterBufR));
  }
  // Fast attack, slower release, so it reads like a real meter.
  smoothL = lvL > smoothL ? lvL : smoothL * 0.85;
  smoothR = lvR > smoothR ? lvR : smoothR * 0.85;
  $("meter-l").style.transform = `scaleY(${smoothL})`;
  $("meter-r").style.transform = `scaleY(${smoothR})`;
  peakL = Math.max(smoothL, peakL - 0.012);
  peakR = Math.max(smoothR, peakR - 0.012);
  $("peak-l").style.bottom = `${peakL * 100}%`;
  $("peak-r").style.bottom = `${peakR * 100}%`;
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
  updateMeters();
}
// Drive updates on a timer rather than requestAnimationFrame: rAF fully pauses
// on hidden/background tabs (freezing the meters); a 30fps interval keeps the
// live meters and clock running whenever the player is playing.
setInterval(loop, 33);
$("playbtn").onclick = () => {
  ensureMeterGraph();
  if (actx && actx.state === "suspended") actx.resume();
  audio.paused ? audio.play() : audio.pause();
};
audio.addEventListener("play", () => $("playbtn").textContent = "❚❚");
audio.addEventListener("pause", () => $("playbtn").textContent = "▶");
$("scrub").onclick = (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const frac = (e.clientX - r.left) / r.width;
  audio.currentTime = frac * (TL.duration || audio.duration || 0);
};

// ── Boot ─────────────────────────────────────────────────────────────────────
fetch("/timeline.json").then((r) => r.json()).then((data) => {
  TL = data; TRACKS = data.tracks || {};
  $("dur").textContent = fmt(data.duration);
  loop();  // render once immediately; setInterval keeps it live
}).catch((e) => { $("cur-title").textContent = "Failed to load timeline"; console.error(e); });
