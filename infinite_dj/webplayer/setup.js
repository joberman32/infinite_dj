"use strict";
// Infinite DJ studio: pick a track pool, dial in Serendipity + Pace, and POST a
// spec to /api/render. Polls the job, then hands off to the player.

const $ = (id) => document.getElementById(id);

let LIB = [];                       // [{artist, albums:[{album, tracks:[]}]}]
const selected = new Set();         // track ids
let serendipity = "medium";
let pace = "flowing";
let polling = null;

const SER_HELP = {
  low: "Whole tracks, long solos, classic DJ crossfades.",
  medium: "Sequential splices — each track plays a bounded segment.",
  high: "Structured collage: layered segments, timbre-contrasting cuts.",
  insane: "Anything goes — heavy overlays, tiny sub-segments, fresh seed. Pace is ignored.",
};

// ── Library tree ─────────────────────────────────────────────────────────────
function albumIds(al) { return al.tracks.map((t) => t.id); }
function artistIds(a) { return a.albums.flatMap(albumIds); }
function stateOf(ids) {
  const n = ids.filter((id) => selected.has(id)).length;
  return n === 0 ? "off" : (n === ids.length ? "on" : "some");
}
function boxHTML(state) {
  const cls = state === "on" ? "box on" : (state === "some" ? "box some" : "box");
  return `<span class="${cls}">${state === "on" ? "✕" : ""}</span>`;
}

function renderLibrary() {
  const host = $("library");
  if (!LIB.length) { host.innerHTML = `<div class="dim" style="padding:12px">No tracks in library.</div>`; return; }
  host.innerHTML = LIB.map((a, ai) => {
    const albums = a.albums.map((al, li) => {
      const tracks = al.tracks.map((t) => `
        <div class="row track-row" data-kind="track" data-id="${t.id}">
          ${boxHTML(selected.has(t.id) ? "on" : "off")}
          <span>${t.title}</span>
          <span class="meta">${Math.round(t.bpm)} BPM · ${t.key}</span>
        </div>`).join("");
      return `<div class="album">
        <div class="row album-row" data-kind="album" data-ai="${ai}" data-li="${li}">
          ${boxHTML(stateOf(albumIds(al)))}<span>${al.album}</span>
          <span class="meta">${al.tracks.length}</span>
        </div>${tracks}</div>`;
    }).join("");
    return `<div class="artist" data-ai="${ai}">
      <div class="row artist-row" data-kind="artist" data-ai="${ai}">
        <span class="caret">▾</span>${boxHTML(stateOf(artistIds(a)))}
        <span>${a.artist}</span><span class="meta">${a.count}</span>
      </div>${albums}</div>`;
  }).join("");
  $("poolcount").textContent = `${selected.size} / ${LIB.reduce((n, a) => n + a.count, 0)} tracks`;
  $("generate").disabled = selected.size < 2;
}

function toggleIds(ids) {
  const allOn = ids.every((id) => selected.has(id));
  ids.forEach((id) => allOn ? selected.delete(id) : selected.add(id));
}

$("library").addEventListener("click", (e) => {
  const row = e.target.closest(".row");
  if (!row) return;
  const kind = row.dataset.kind;
  if (kind === "track") {
    const id = row.dataset.id;
    selected.has(id) ? selected.delete(id) : selected.add(id);
  } else if (kind === "album") {
    toggleIds(albumIds(LIB[+row.dataset.ai].albums[+row.dataset.li]));
  } else if (kind === "artist") {
    // Clicking the caret collapses; clicking elsewhere toggles selection.
    if (e.target.classList.contains("caret")) {
      row.parentElement.classList.toggle("collapsed");
      e.target.textContent = row.parentElement.classList.contains("collapsed") ? "▸" : "▾";
      return;
    }
    toggleIds(artistIds(LIB[+row.dataset.ai]));
  }
  renderLibrary();
});
$("select-all").onclick = () => { LIB.forEach((a) => artistIds(a).forEach((id) => selected.add(id))); renderLibrary(); };
$("select-none").onclick = () => { selected.clear(); renderLibrary(); };

// ── Controls ─────────────────────────────────────────────────────────────────
function applySerendipity(v) {
  serendipity = v;
  [...$("serendipity").children].forEach((b) => b.classList.toggle("on", b.dataset.v === v));
  $("ser-help").textContent = SER_HELP[v] || "";
  const insane = v === "insane";
  $("pace").classList.toggle("disabled", insane);
  $("advanced").classList.toggle("disabled", insane);
  $("adv-toggle").classList.toggle("hidden", insane);
  $("pace-note").classList.toggle("hidden", !insane);
  if (insane) $("advanced").classList.add("hidden");
  ["min-sec", "max-sec"].forEach((id) => { $(id).disabled = insane; });
}
$("serendipity").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (b) applySerendipity(b.dataset.v);
});

function buildPace(presets) {
  $("pace").innerHTML = presets.map((p) =>
    `<button data-v="${p}"${p === pace ? ' class="on"' : ""}>${p.toUpperCase()}</button>`).join("");
}
$("pace").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  pace = b.dataset.v;
  [...$("pace").children].forEach((x) => x.classList.toggle("on", x.dataset.v === pace));
});
$("adv-toggle").onclick = () => $("advanced").classList.toggle("hidden");

// ── Generate ─────────────────────────────────────────────────────────────────
function spec() {
  const s = {
    track_ids: [...selected],
    serendipity,
    pace,
    length_min: Math.max(1, Number($("length").value) || 8),
  };
  if (serendipity !== "insane" && !$("advanced").classList.contains("hidden")) {
    const lo = Number($("min-sec").value), hi = Number($("max-sec").value);
    if (lo > 0 && hi > lo) { s.min_sec = lo; s.max_sec = hi; }
  }
  return s;
}

$("generate").onclick = async () => {
  if (selected.size < 2) return;
  $("generate").disabled = true;
  $("status").textContent = "generating…";
  try {
    const r = await fetch("/api/render", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec()),
    });
    const { job, error } = await r.json();
    if (error) throw new Error(error);
    poll(job);
  } catch (e) {
    $("status").textContent = `error: ${e.message}`;
    $("generate").disabled = false;
  }
};

function poll(job) {
  clearInterval(polling);
  const started = Date.now();
  polling = setInterval(async () => {
    try {
      const r = await fetch(`/api/render?job=${job}`);
      const j = await r.json();
      const secs = ((Date.now() - started) / 1000) | 0;
      if (j.status === "done") {
        clearInterval(polling);
        $("status").textContent = "ready — opening player…";
        window.location = `/player?job=${job}`;
      } else if (j.status === "error") {
        clearInterval(polling);
        $("status").textContent = `error: ${j.error}`;
        $("generate").disabled = false;
      } else {
        $("status").textContent = `${j.status}… ${secs}s`;
      }
    } catch (e) { /* transient — keep polling */ }
  }, 1000);
}

// ── Boot ─────────────────────────────────────────────────────────────────────
fetch("/api/library").then((r) => r.json()).then((data) => {
  LIB = data.artists || [];
  LIB.forEach((a) => artistIds(a).forEach((id) => selected.add(id)));   // start with all
  buildPace(data.pace_presets || ["flowing"]);
  applySerendipity(serendipity);
  renderLibrary();
}).catch((e) => {
  $("library").innerHTML = `<div class="dim" style="padding:12px">Failed to load library: ${e.message}</div>`;
});
