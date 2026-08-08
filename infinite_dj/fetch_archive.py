"""
Fetch Creative Commons electronic music from the Internet Archive netlabels
collection, screened against what the library actually needs.

The `gaps` report says a library is gated on *composition*, not size: 24 hours
of tempo-scattered tracks still renders mostly hard cuts. So this module screens
before it downloads, using metadata the Archive already publishes.

**The screening trick.** Around 2014-15 the Archive ran Essentia over most of
its audio and left the results beside each file as `<stem>_esshigh.json.gz`
(genre + danceability classifiers, ~2 KB) and `<stem>_esslow.json.gz` (BPM, key,
duration, ~24 KB). Reading those is ~26 KB per track against ~6 MB for the MP3,
so a BPM/genre target costs a couple of hundred KB to evaluate instead of a
couple of hundred megabytes. Screens run cheapest-first — duration, then the
2 KB genre file, then the 24 KB rhythm file — so most rejects cost 2 KB.

Nothing here writes to the track database. The flow is deliberately two-step:
this fetches and organizes files on disk, then `analyze` and `triage` decide
what the engine actually thinks of them. The Archive's BPM is a screening hint
only — `analyzer` re-derives everything from the audio.

**Licensing.** Every downloaded file's license, source URL and creator are
appended to `PROVENANCE.jsonl` in the destination root. Items without a
declared license are skipped by default. Note that much of the collection is
NoDerivatives (`by-nc-nd`): fine for private listening, not for publishing a
mix built from it. `--license` filters on the short code if that matters.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

SCRAPE_URL   = "https://archive.org/services/search/v1/scrape"
METADATA_URL = "https://archive.org/metadata/"
DOWNLOAD_URL = "https://archive.org/download/"

# The Archive asks that automated clients identify themselves.
USER_AGENT = "infinite-dj/0.1 (personal DJ library builder)"

# Ready-made searches. All are scoped to `netlabels`, the Archive's curated
# collection of net-label releases, which is Creative Commons by editorial
# policy — that is what makes bulk fetching legitimate here.
PRESETS: Dict[str, str] = {
    "house": 'collection:netlabels AND mediatype:audio AND '
             '(subject:house OR subject:"deep house" OR subject:"tech house")',
    "techno": 'collection:netlabels AND mediatype:audio AND '
              '(subject:techno OR subject:"minimal techno" OR subject:minimal)',
    "experimental": 'collection:netlabels AND mediatype:audio AND '
                    '(subject:experimental OR subject:ambient OR subject:idm)',
    "electronic": 'collection:netlabels AND mediatype:audio AND subject:electronic',
}

# Format preference: originals before derivatives, lossless before lossy. The
# Archive's `format` strings are free text, so this is a lookup, not a parse.
FORMAT_RANK: Dict[str, Tuple[int, str]] = {
    "Flac":          (0, ".flac"),
    "24bit Flac":    (0, ".flac"),
    "AIFF":          (1, ".aiff"),
    "WAVE":          (1, ".wav"),
    "VBR MP3":       (2, ".mp3"),
    "320Kbps MP3":   (2, ".mp3"),
    "256Kbps MP3":   (3, ".mp3"),
    "192Kbps MP3":   (3, ".mp3"),
    "MP3":           (4, ".mp3"),
    "Ogg Vorbis":    (5, ".ogg"),
    "128Kbps MP3":   (6, ".mp3"),
    "64Kbps MP3":    (9, ".mp3"),
    "32Kbps MP3":    (9, ".mp3"),
}

# The engine folds tempo into this range (`analyzer.BPM_MIN` / `BPM_MAX`), so a
# BPM target has to be compared at the same octave the engine will see.
BPM_MIN, BPM_MAX = 90.0, 180.0

# Essentia's `genre_electronic` classifier vocabulary. "experimental" isn't one
# of its classes; ambient is the closest thing it can express.
ELECTRONIC_GENRES = ("ambient", "dnb", "house", "techno", "trance")


# ── Screening spec ───────────────────────────────────────────────────────────

@dataclass
class Screen:
    """
    What counts as a track worth downloading.

    Defaults target the thing the gap report flags first: danceable
    four-to-the-floor material long enough to hold a solo plus a blend.
    """
    bpm_center: Optional[float] = None
    bpm_tol: float = 8.0
    min_duration: float = 150.0      # shorter can't hold a 32-bar dwell + blend
    max_duration: float = 900.0      # longer is usually a DJ mix, not a track
    genres: Tuple[str, ...] = ("house", "techno")
    min_genre_prob: float = 0.25     # summed over `genres`
    min_electronic: float = 0.50     # genre_dortmund "electronic" probability
    min_danceability: float = 0.0    # off by default; BPM + genre carry it
    allow_unscreened: bool = False   # keep tracks with no Essentia sidecars
    licenses: Optional[Tuple[str, ...]] = None   # e.g. ("by", "by-sa", "by-nc")
    require_license: bool = True


@dataclass
class Candidate:
    """One audio file on the Archive, with whatever we know about it so far."""
    identifier: str
    filename: str
    format: str
    size: int = 0
    creator: str = ""            # item-level: the label, or "AA.VV." on comps
    artist: str = ""             # per-file: the actual track artist
    album: str = ""
    track_title: str = ""
    license: str = ""
    duration: float = 0.0
    bpm: float = 0.0
    key: str = ""
    genre: str = ""
    genre_prob: float = 0.0          # probability of `genre`, the top class
    target_prob: float = 0.0         # summed probability of the genres we asked for
    danceability: float = 0.0
    screened: bool = False           # did Essentia data actually load?
    reject: str = ""                 # why it was dropped, "" if kept

    @property
    def stem(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def url(self) -> str:
        return (DOWNLOAD_URL + urllib.parse.quote(self.identifier) + "/"
                + urllib.parse.quote(self.filename))

    def dest_path(self, root: str) -> str:
        """
        `<root>/<creator>/<artist> - <title>.<ext>`.

        The folder uses the item-level creator so a release stays together; the
        filename uses the per-file artist, which on a compilation is the only
        place the real name appears (item creator reads "AA.VV."). The stem is
        UI: `analyzer` titles a track from its filename.
        """
        ext = os.path.splitext(self.filename)[1] or ".mp3"
        folder = _safe(self.creator or self.identifier) or self.identifier
        title = _safe(self.track_title) or _safe(self.stem)
        artist = _safe(self.artist or self.creator)
        name = f"{artist} - {title}" if artist and artist != title else title
        return os.path.join(root, folder, f"{name[:120]}{ext}")


# ── HTTP ─────────────────────────────────────────────────────────────────────

class FetchError(RuntimeError):
    pass


def _get(url: str, params: Optional[dict] = None, *, timeout: float = 30.0,
         retries: int = 3) -> bytes:
    """GET with backoff. Raises `FetchError` rather than leaking urllib types."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (404, 403):        # not transient — don't retry
                raise FetchError(f"{last} for {url}") from e
        except Exception as e:              # timeouts, DNS, connection resets
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"{last} for {url}")


def _get_json(url: str, params: Optional[dict] = None,
              timeout: float = 60.0) -> dict:
    # 60s, not the 30s default: `/metadata/` responses for a release with a few
    # hundred files are large and slow, and a timeout there discards the whole
    # release. Measured at 30s, roughly a third of releases were being lost this
    # way and counted as "item unreadable".
    return json.loads(_get(url, params, timeout=timeout).decode("utf-8", "replace"))


def _get_gzip_json(url: str) -> dict:
    return json.loads(gzip.decompress(_get(url)).decode("utf-8", "replace"))


# ── Search ───────────────────────────────────────────────────────────────────

def search_items(query: str, limit: int = 500) -> List[dict]:
    """
    Archive items matching `query`, via the scrape API (cursor-paged, unlike
    `advancedsearch.php`, which caps out on deep result sets).
    """
    out: List[dict] = []
    cursor = None
    while len(out) < limit:
        params = {
            "q": query,
            "fields": "identifier,title,creator,licenseurl,downloads",
            "count": min(1000, max(100, limit - len(out))),
        }
        if cursor:
            params["cursor"] = cursor
        page = _get_json(SCRAPE_URL, params)
        items = page.get("items") or []
        if not items:
            break
        out.extend(items)
        cursor = page.get("cursor")
        if not cursor:
            break
    return out[:limit]


def item_candidates(identifier: str, item_meta: Optional[dict] = None) -> List[Candidate]:
    """
    The best audio file per track in one Archive item.

    Derivatives share a stem with their original (`x.flac` -> `x.mp3`), so files
    are grouped by stem and the highest-ranked format in each group wins.
    """
    data = _get_json(METADATA_URL + urllib.parse.quote(identifier))
    md = data.get("metadata") or {}
    item_meta = item_meta or {}
    creator = _first(md.get("creator") or item_meta.get("creator") or "")
    album = _first(md.get("title") or item_meta.get("title") or "")
    lic = _first(md.get("licenseurl") or item_meta.get("licenseurl") or "")

    best: Dict[str, Tuple[int, Candidate]] = {}
    for f in data.get("files") or []:
        rank_ext = FORMAT_RANK.get(f.get("format") or "")
        if not rank_ext:
            continue
        rank, _ext = rank_ext
        stem = os.path.splitext(f.get("name") or "")[0]
        if not stem:
            continue
        cand = Candidate(
            identifier=identifier, filename=f["name"], format=f["format"],
            size=int(f.get("size") or 0), creator=creator,
            artist=_first(f.get("artist") or "") or creator, album=album,
            track_title=_first(f.get("title") or "") or stem,
            license=lic, duration=_parse_length(f.get("length")),
        )
        if stem not in best or rank < best[stem][0]:
            best[stem] = (rank, cand)
    return [c for _r, c in best.values()]


# ── Essentia screening ───────────────────────────────────────────────────────

def load_highlevel(cand: Candidate) -> bool:
    """Genre + danceability from the ~2 KB sidecar. False if it isn't there."""
    url = (DOWNLOAD_URL + urllib.parse.quote(cand.identifier) + "/"
           + urllib.parse.quote(cand.stem) + "_esshigh.json.gz")
    try:
        hl = (_get_gzip_json(url).get("highlevel") or {})
    except (FetchError, OSError, ValueError):
        return False
    elec = ((hl.get("genre_electronic") or {}).get("all") or {})
    if elec:
        cand.genre = max(elec, key=elec.get)
        cand.genre_prob = float(elec[cand.genre])
    dance = ((hl.get("danceability") or {}).get("all") or {})
    cand.danceability = float(dance.get("danceable", 0.0))
    cand._elec_all = elec                                    # type: ignore[attr-defined]
    cand._dortmund = ((hl.get("genre_dortmund") or {}).get("all") or {})  # type: ignore[attr-defined]
    return bool(elec)


def load_lowlevel(cand: Candidate) -> bool:
    """BPM, key and true duration from the ~24 KB sidecar."""
    url = (DOWNLOAD_URL + urllib.parse.quote(cand.identifier) + "/"
           + urllib.parse.quote(cand.stem) + "_esslow.json.gz")
    try:
        d = _get_gzip_json(url)
    except (FetchError, OSError, ValueError):
        return False
    cand.bpm = float((d.get("rhythm") or {}).get("bpm") or 0.0)
    tonal = d.get("tonal") or {}
    if tonal.get("key_key"):
        cand.key = f"{tonal['key_key']} {tonal.get('key_scale', '')}".strip()
    length = ((d.get("metadata") or {}).get("audio_properties") or {}).get("length")
    if length:
        cand.duration = float(length)
    return cand.bpm > 0


def fold_bpm(bpm: float) -> float:
    """Fold into the engine's [90, 180) window, as `analyzer` does."""
    if bpm <= 0:
        return 0.0
    while bpm < BPM_MIN:
        bpm *= 2.0
    while bpm >= BPM_MAX:
        bpm /= 2.0
    return bpm


def screen(cand: Candidate, spec: Screen,
           fetch: bool = True) -> str:
    """
    Decide on one candidate; returns "" to keep it or a short rejection reason.

    Checks run cheapest-first so a reject usually costs one 2 KB request or
    none at all. Mutates `cand` with whatever it learned along the way.
    """
    if spec.require_license and not cand.license:
        return "no declared license"
    if spec.licenses and _license_code(cand.license) not in spec.licenses:
        return f"license ({_license_code(cand.license) or 'unknown'})"

    # Free duration check: the file listing already carries a length for most
    # derivatives, so obvious interludes and hour-long mixes die before any
    # sidecar request.
    if cand.duration and not (spec.min_duration <= cand.duration <= spec.max_duration):
        return f"duration ({cand.duration:.0f}s)"

    if not fetch:
        return ""

    has_high = load_highlevel(cand)
    if has_high:
        elec_prob = float(getattr(cand, "_dortmund", {}).get("electronic", 0.0))
        if elec_prob and elec_prob < spec.min_electronic:
            return f"not electronic ({elec_prob:.2f})"
        # The gate is the *summed* probability of the genres asked for, not the
        # top class: a track that reads 0.4 trance / 0.3 house is still house
        # enough to mix. So `genre` (top class) and `target_prob` (the number
        # that decided it) can disagree, and both are worth reporting.
        elec = getattr(cand, "_elec_all", {})
        cand.target_prob = sum(float(elec.get(g, 0.0)) for g in spec.genres)
        if cand.target_prob < spec.min_genre_prob:
            return f"wrong subgenre ({cand.genre} {cand.target_prob:.2f})"
        if cand.danceability < spec.min_danceability:
            return f"danceability ({cand.danceability:.2f})"
    elif not spec.allow_unscreened:
        return "no Essentia data"

    # Only pay for the 24 KB rhythm file when a tempo target makes it matter.
    if spec.bpm_center is not None:
        if not load_lowlevel(cand):
            return "" if spec.allow_unscreened else "no BPM data"
        if not (spec.min_duration <= cand.duration <= spec.max_duration):
            return f"duration ({cand.duration:.0f}s)"
        if abs(fold_bpm(cand.bpm) - fold_bpm(spec.bpm_center)) > spec.bpm_tol:
            return (f"off-tempo ({fold_bpm(cand.bpm):.0f} vs "
                    f"{fold_bpm(spec.bpm_center):.0f}±{spec.bpm_tol:.0f})")

    cand.screened = has_high
    return ""


# ── Planning ─────────────────────────────────────────────────────────────────

@dataclass
class FetchPlan:
    keep: List[Candidate] = field(default_factory=list)
    rejected: Dict[str, int] = field(default_factory=dict)
    items_seen: int = 0
    tracks_seen: int = 0

    @property
    def hours(self) -> float:
        return sum(c.duration or 300.0 for c in self.keep) / 3600.0

    @property
    def bytes(self) -> int:
        return sum(c.size for c in self.keep)


def _screen_item(ident: str, item: dict, spec: Screen,
                 have: set) -> Tuple[List[Candidate], List[str], int]:
    """Screen every track in one item. Runs in a worker thread; touches no
    shared state, so the caller can merge results under a single lock."""
    try:
        cands = item_candidates(ident, item)
    except FetchError:
        return [], ["item unreadable"], 0
    keep: List[Candidate] = []
    rejects: List[str] = []
    for cand in cands:
        if f"{cand.identifier}/{cand.filename}" in have:
            rejects.append("already have")
            continue
        reason = screen(cand, spec)
        if reason:
            rejects.append(reason)
        else:
            keep.append(cand)
    return keep, rejects, len(cands)


def build_plan(queries: Iterable[str], spec: Screen, *, target_hours: float,
               max_items: int = 400, max_bytes: int = 20 * 1024 ** 3,
               have: Optional[set] = None, workers: int = 6,
               progress: Optional[Callable[[FetchPlan, str], None]] = None) -> FetchPlan:
    """
    Walk search results item by item until `target_hours` of surviving tracks
    have accumulated, or the item budget runs out.

    Stops early on purpose: screening is the expensive part, so there's no
    reason to evaluate the whole collection once the target is met. Items are
    screened concurrently — the work is almost entirely network latency — but
    `workers` stays small to keep the load on the Archive polite.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    plan = FetchPlan()
    have = have or set()
    seen_items: set = set()

    pending: List[Tuple[str, dict]] = []
    for query in queries:
        for item in search_items(query, limit=max_items):
            ident = item.get("identifier")
            if ident and ident not in seen_items:
                seen_items.add(ident)
                pending.append((ident, item))

    if workers <= 1:
        results = ((i, _screen_item(i, m, spec, have)) for i, m in pending)
    else:
        def _run(pair):
            return pair[0], _screen_item(pair[0], pair[1], spec, have)
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = [pool.submit(_run, p) for p in pending]
        results = (f.result() for f in as_completed(futures))

    try:
        for ident, (keep, rejects, n_tracks) in results:
            plan.items_seen += 1
            plan.tracks_seen += n_tracks
            for reason in rejects:
                key = _reason_key(reason)
                plan.rejected[key] = plan.rejected.get(key, 0) + 1
            for cand in keep:
                if plan.hours >= target_hours or plan.bytes >= max_bytes:
                    return plan
                plan.keep.append(cand)
            if progress:
                progress(plan, ident)
            if plan.hours >= target_hours or plan.bytes >= max_bytes:
                return plan
    finally:
        if workers > 1:
            for f in futures:
                f.cancel()
            pool.shutdown(wait=False)
    return plan


def _reason_key(reason: str) -> str:
    """Collapse a rejection message to a countable category. Reasons are
    written as `category (detail)`, so the split is all it takes."""
    return reason.split(" (")[0].strip() or reason


# ── Download ─────────────────────────────────────────────────────────────────

def download(cand: Candidate, root: str, *, timeout: float = 120.0) -> Tuple[str, bool]:
    """
    Fetch one file to its destination. Returns `(path, downloaded)`; a file
    already present at the expected size is left alone so runs are resumable.

    Writes to `.part` and renames, so an interrupted run never leaves a
    truncated file for `analyze` to choke on.
    """
    dest = cand.dest_path(root)
    if os.path.exists(dest) and (not cand.size or os.path.getsize(dest) == cand.size):
        return dest, False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(cand.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dest)
    return dest, True


def record_provenance(root: str, cand: Candidate, dest: str) -> None:
    """Append the attribution record. CC-BY and friends require this survive."""
    os.makedirs(root, exist_ok=True)
    row = {
        "path": os.path.relpath(dest, root),
        "source": cand.url,
        "identifier": cand.identifier,
        "creator": cand.creator,
        "artist": cand.artist,
        "album": cand.album,
        "title": cand.track_title,
        "license": cand.license,
        "archive_bpm": round(cand.bpm, 2) or None,
        "archive_key": cand.key or None,
        "archive_genre": cand.genre or None,
    }
    with open(os.path.join(root, "PROVENANCE.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")


def existing_sources(root: str) -> set:
    """`identifier/filename` keys already fetched, so re-runs skip them."""
    path = os.path.join(root, "PROVENANCE.jsonl")
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            src = row.get("source", "")
            if "/download/" in src:
                out.add(urllib.parse.unquote(src.split("/download/", 1)[1]))
    return out


# ── Helpers ──────────────────────────────────────────────────────────────────

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe(name: str) -> str:
    """A filesystem-safe fragment. The stem becomes the library title."""
    name = _UNSAFE.sub(" ", str(name or ""))
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name


def _first(value) -> str:
    """Archive metadata fields are sometimes a list, sometimes a string."""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value or "")


def _parse_length(value) -> float:
    """File lengths come as seconds ("272.12") or "MM:SS" / "H:MM:SS"."""
    if value is None:
        return 0.0
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    try:
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60.0 + float(part)
        return seconds
    except ValueError:
        return 0.0


def _license_code(url: str) -> str:
    """`http://creativecommons.org/licenses/by-nc-sa/3.0/` -> `by-nc-sa`."""
    m = re.search(r"/licenses/([a-z-]+)/", url or "")
    if m:
        return m.group(1)
    if "publicdomain" in (url or ""):
        return "cc0"
    return ""


def candidate_dict(cand: Candidate) -> dict:
    """JSON-safe view, dropping the private Essentia scratch attributes."""
    return {k: v for k, v in asdict(cand).items() if not k.startswith("_")}
