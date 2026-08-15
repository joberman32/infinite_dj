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

function readableTrackTitle(raw) {
  let title = (raw || "Unknown track").replace(/\.[^.]+$/, "").replaceAll("_", " ");
  title = title.replace(/\s*-\s*/g, " - ");
  title = title.replace(/^[a-z][a-z0-9-]*\d+\s+\d+\s+/i, "");
  title = title.replace(/^\d{1,3}\s+/, "");
  const parts = title.split(" - ");
  return parts[parts.length - 1].replace(/\s+/g, " ").trim();
}

function clipLabel(c) {
  if (!c) return "—";
  const title = readableTrackTitle((TRACKS[c.track] || {}).title);
  const section = c.section || "segment";
  const trackClips = TL.clips.filter((other) => other.track === c.track);
  const ordinal = trackClips.indexOf(c) + 1;
  const sectionCode = section.charAt(0).toUpperCase();
  return `${title} · ${ordinal}${sectionCode}`;
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
    const outgoing = act.find((item) => item.c !== newest.c);
    const overlapEnd = outgoing
      ? Math.min(newest.c.start + newest.c.fade_in, outgoing.c.end)
      : newest.c.start + newest.c.fade_in;
    const overlapDuration = Math.max(0.001, overlapEnd - newest.c.start);
    transition = {
      from: outgoing ? outgoing.c : null,
      to: newest.c,
      progress: (t - newest.c.start) / overlapDuration,
      mode: newest.c.mode,
    };
  }
  let upcoming = null, best = Infinity;
  for (const c of TL.clips)
    if (c.start > t && c.start < best) { best = c.start; upcoming = c; }
  // Previous: the clip that started most recently before the primary clip.
  // This can be another segment from the same track.
  let prev = null;
  if (primary) {
    let bestStart = -Infinity;
    for (const c of TL.clips)
      if (c.start < primary.c.start && c.start > bestStart) {
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
  const currentTitle = $("cur-title");
  const incomingTitle = $("incoming-title");

  if (st.transition && st.transition.from) {
    const progress = Math.max(0, Math.min(1, st.transition.progress));
    const from = st.transition.from;
    const to = st.transition.to;
    currentTitle.textContent = clipLabel(from);
    incomingTitle.textContent = clipLabel(to);
    currentTitle.style.opacity = `${1 - progress}`;
    incomingTitle.style.opacity = `${progress}`;
    currentTitle.style.transform = `translateY(${-5 * progress}px)`;
    incomingTitle.style.transform = `translateY(${5 * (1 - progress)}px)`;
    const toTrack = TRACKS[to.track] || {};
    $("cur-bpm").textContent = `${Math.round(to.bpm || toTrack.bpm || 0)} BPM`;
  } else if (p) {
    const tr = TRACKS[p.track] || {};
    currentTitle.textContent = clipLabel(p);
    currentTitle.style.opacity = "1";
    currentTitle.style.transform = "translateY(0)";
    incomingTitle.textContent = "";
    incomingTitle.style.opacity = "0";
    incomingTitle.style.transform = "translateY(5px)";
    $("cur-bpm").textContent = `${Math.round(p.bpm || tr.bpm || 0)} BPM`;
  }

  $("prev-title").textContent = clipLabel(st.prev);
  $("next-title").textContent = clipLabel(st.upcoming);
}

// ── Overlap timeline ─────────────────────────────────────────────────────────
// A scrolling multi-lane view of which clips are sounding around `t`, so a
// layered collage's overlap is visible instead of only audible. Lanes are
// assigned once per clip (greedy interval colouring, cached by clip identity)
// so a bar never jumps rows on a later radio poll even though `TL.clips` is
// replaced wholesale each time.
const OVERLAP_WINDOW_SEC = 30;
const OVERLAP_PLAYHEAD_FRAC = 0.28;   // where "now" sits across the window
const OVERLAP_MAX_LANES = 5;          // matches the engine's highest layer count (insane)
const OVERLAP_LANE_H = 20;            // px; keep in sync with #lanes height in player.css

const laneOf = new Map();             // clip key -> lane index (or -1: overflow)
const laneFreeAt = new Array(OVERLAP_MAX_LANES).fill(-Infinity);
const overlapBars = new Map();        // clip key -> DOM el, reused across frames

function clipKey(c) { return `${c.track}:${c.start}`; }

function assignLanes() {
  if (!TL) return;
  const sorted = [...TL.clips].sort((a, b) => a.start - b.start);
  for (const c of sorted) {
    const key = clipKey(c);
    if (laneOf.has(key)) continue;
    let lane = -1;
    for (let i = 0; i < OVERLAP_MAX_LANES; i++) {
      if (laneFreeAt[i] <= c.start) { lane = i; break; }
    }
    if (lane !== -1) laneFreeAt[lane] = c.end;
    laneOf.set(key, lane);
  }
}

function renderOverlap(t) {
  if (!TL) return;
  assignLanes();
  const lanesEl = $("lanes");
  const windowStart = t - OVERLAP_WINDOW_SEC * OVERLAP_PLAYHEAD_FRAC;

  $("playhead").style.left = `${OVERLAP_PLAYHEAD_FRAC * 100}%`;

  const visible = new Set();
  let activeCount = 0;
  for (const c of TL.clips) {
    if (c.start <= t && t < c.end) activeCount++;
    if (c.end < windowStart || c.start > windowStart + OVERLAP_WINDOW_SEC) continue;
    const lane = laneOf.get(clipKey(c));
    if (lane === undefined || lane === -1) continue;   // more than MAX_LANES overlapping — drop, not drawn
    const key = clipKey(c);
    visible.add(key);
    let el = overlapBars.get(key);
    if (!el) {
      el = document.createElement("div");
      el.className = "clip-bar";
      lanesEl.appendChild(el);
      overlapBars.set(key, el);
    }
    const track = TRACKS[c.track] || {};
    const left = ((c.start - windowStart) / OVERLAP_WINDOW_SEC) * 100;
    const width = ((c.end - c.start) / OVERLAP_WINDOW_SEC) * 100;
    el.style.left = `${left}%`;
    el.style.width = `${Math.max(0.3, width)}%`;
    el.style.top = `${lane * OVERLAP_LANE_H}px`;
    el.style.background = track.color || "var(--dim)";
    const g = clipGain(c, Math.max(c.start, Math.min(c.end - 1e-3, t)));
    el.style.opacity = `${0.35 + 0.65 * g}`;
    const span = Math.max(0.01, c.end - c.start);
    const fadeInPct = c.fade_in > 0 ? Math.min(50, (c.fade_in / span) * 100) : 0;
    const fadeOutPct = c.fade_out > 0 ? Math.min(50, (c.fade_out / span) * 100) : 0;
    const mask = `linear-gradient(to right, transparent 0%, black ${fadeInPct}%, ` +
      `black ${100 - fadeOutPct}%, transparent 100%)`;
    el.style.maskImage = mask;
    el.style.webkitMaskImage = mask;
    el.title = `${readableTrackTitle(track.title)} · ${c.mode}${c.eq ? " · " + c.eq : ""}`;
  }
  for (const [key, el] of overlapBars) {
    if (!visible.has(key)) { el.remove(); overlapBars.delete(key); }
  }
  $("layer-count").textContent = `${activeCount} layer${activeCount === 1 ? "" : "s"}`;
}

// ── Audio graph + stereo power meter (Web Audio API) ─────────────────────────
// Everything audible runs through one `bus`, so the meters read the same signal
// whether it comes from the <audio> element (a finished mix) or from scheduled
// buffers (radio). A growing stream can't be fed through <audio>/MSE, hence the
// two sources.
let actx = null, bus = null, analyserL = null, analyserR = null;
let meterBufL = null, meterBufR = null;
let peakL = 0, peakR = 0;

function ensureAudioGraph(connectElement) {
  if (actx) return true;
  try {
    actx = new (window.AudioContext || window.webkitAudioContext)();
    bus = actx.createGain();
    bus.connect(actx.destination);
    const splitter = actx.createChannelSplitter(2);
    bus.connect(splitter);
    analyserL = actx.createAnalyser(); analyserL.fftSize = 512;
    analyserR = actx.createAnalyser(); analyserR.fftSize = 512;
    splitter.connect(analyserL, 0);
    splitter.connect(analyserR, 1);
    meterBufL = new Uint8Array(analyserL.fftSize);
    meterBufR = new Uint8Array(analyserR.fftSize);
    if (connectElement) actx.createMediaElementSource(audio).connect(bus);
    return true;
  } catch (e) {
    console.warn("Audio graph unavailable:", e);
    actx = null;
    return false;
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

// ── Radio: schedule rendered chunks back-to-back on the AudioContext clock ───
const RADIO = new URLSearchParams(location.search).get("radio") === "1";
const radio = { t0: null, nextIdx: 0, nextAt: 0, chunks: [], playing: false };

function radioTime() {
  if (!actx || radio.t0 === null) return 0;
  return Math.max(0, actx.currentTime - radio.t0);
}

async function radioPoll() {
  if (!RADIO || !jobParam) return;
  try {
    const played = radioTime().toFixed(3);
    const s = await (await fetch(
      `/api/radio/state?job=${encodeURIComponent(jobParam)}&played=${played}`
    )).json();
    if (s.error) { $("cur-title").textContent = s.error; return; }
    TL = { clips: s.clips || [], tracks: s.tracks || {}, duration: s.generated_sec };
    TRACKS = TL.tracks;
    radio.chunks = s.chunks || [];
    $("dur").textContent = "LIVE";
    scheduleAhead();
  } catch (e) { /* transient — the next poll retries */ }
}

async function scheduleAhead() {
  if (!RADIO || !radio.playing || !actx) return;
  // Keep roughly 30s queued ahead of the clock; each chunk is decoded and
  // started exactly where the previous one ends, so seams are sample-accurate.
  while (radio.nextIdx < radio.chunks.length &&
         radio.nextAt - actx.currentTime < 30) {
    const i = radio.nextIdx;
    try {
      const res = await fetch(`/api/radio/chunk?job=${jobParam}&i=${i}`);
      const buf = await actx.decodeAudioData(await res.arrayBuffer());
      const src = actx.createBufferSource();
      src.buffer = buf;
      src.connect(bus);
      const when = Math.max(actx.currentTime + 0.08, radio.nextAt);
      src.start(when);
      if (radio.t0 === null) radio.t0 = when;
      radio.nextAt = when + buf.duration;
      radio.nextIdx = i + 1;
    } catch (e) {
      console.warn("chunk", i, e);
      break;
    }
  }
}

// Reveal the fixed dBFS gradient up to `v` rather than scaling the fill, so a
// colour always means the same level.
function setMeter(el, v) {
  const f = Math.max(0, Math.min(1, v));
  el.style.clipPath = `inset(${(1 - f) * 100}% 0 0 0)`;
}

let smoothL = 0, smoothR = 0;
function updateMeters() {
  let lvL = 0, lvR = 0;
  const live = RADIO ? (radio.playing && actx && actx.state === "running")
                     : !audio.paused;
  // Same signal that gates the meters — so the badge and the meters can
  // never disagree about whether audio is actually flowing.
  if (RADIO) {
    $("livedot").textContent = live ? "◉" : "❚❚";
    $("livelabel").textContent = live ? "LIVE" : "PAUSED";
    $("livetag").classList.toggle("paused", !live);
  }
  if (analyserL && analyserR && live) {
    lvL = dbMeterLevel(rms(analyserL, meterBufL));
    lvR = dbMeterLevel(rms(analyserR, meterBufR));
  }
  // Fast attack, slower release, so it reads like a real meter.
  smoothL = lvL > smoothL ? lvL : smoothL * 0.85;
  smoothR = lvR > smoothR ? lvR : smoothR * 0.85;
  setMeter($("meter-l"), smoothL);
  setMeter($("meter-r"), smoothR);
  peakL = Math.max(smoothL, peakL - 0.012);
  peakR = Math.max(smoothR, peakR - 0.012);
  $("peak-l").style.bottom = `${peakL * 100}%`;
  $("peak-r").style.bottom = `${peakR * 100}%`;
}

// ── Transport ────────────────────────────────────────────────────────────────
function loop() {
  const t = RADIO ? radioTime() : audio.currentTime;
  $("pos").textContent = fmt(t);
  $("ttime").textContent = fmt(t);
  if (!RADIO) {
    const frac = TL ? Math.min(1, t / (TL.duration || audio.duration || 1)) : 0;
    $("scrub-fill").style.width = `${frac * 100}%`;
    $("scrub-head").style.left = `${frac * 100}%`;
  }
  render(t);
  renderOverlap(t);
  updateMeters();
}
// Drive updates on a timer rather than requestAnimationFrame: rAF fully pauses
// on hidden/background tabs (freezing the meters); a 30fps interval keeps the
// live meters and clock running whenever the player is playing.
setInterval(loop, 33);
$("playbtn").onclick = async () => {
  if (RADIO) {
    if (!ensureAudioGraph(false)) return;
    if (!radio.playing) {
      radio.playing = true;
      $("playbtn").textContent = "❚❚";
      await actx.resume();
      scheduleAhead();
    } else if (actx.state === "running") {
      await actx.suspend();                 // pause: the clock stops with it
      $("playbtn").textContent = "▶";
    } else {
      await actx.resume();
      $("playbtn").textContent = "❚❚";
    }
    return;
  }
  ensureAudioGraph(true);
  if (actx && actx.state === "suspended") actx.resume();
  audio.paused ? audio.play() : audio.pause();
};
$("exitbtn").onclick = async () => {
  try { await fetch(`/api/radio/stop?job=${jobParam}`, { method: "POST" }); }
  catch (e) { /* leaving anyway */ }
  window.location = "/";
};
audio.addEventListener("play", () => $("playbtn").textContent = "❚❚");
audio.addEventListener("pause", () => $("playbtn").textContent = "▶");
$("scrub").onclick = (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const frac = (e.clientX - r.left) / r.width;
  audio.currentTime = frac * (TL.duration || audio.duration || 0);
};

// ── Boot ─────────────────────────────────────────────────────────────────────
// A studio-rendered mix is addressed by ?job=<id>; without one the server
// serves the single pre-rendered mix (the `dj.py serve` path).
const jobParam = new URLSearchParams(location.search).get("job");
const jobQuery = jobParam ? `?job=${encodeURIComponent(jobParam)}` : "";

if (RADIO) {
  // Endless mix: no duration and nothing to seek, so the scrubber gives way to
  // a live badge and an exit.
  document.body.classList.add("radio");
  radioPoll();
  setInterval(radioPoll, 2000);
} else {
  audio.src = `/audio${jobQuery}`;
  fetch(`/timeline.json${jobQuery}`).then((r) => r.json()).then((data) => {
    TL = data; TRACKS = data.tracks || {};
    $("dur").textContent = fmt(data.duration);
    loop();  // render once immediately; setInterval keeps it live
  }).catch((e) => { $("cur-title").textContent = "Failed to load timeline"; console.error(e); });
}
