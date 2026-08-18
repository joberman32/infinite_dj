"""A/B: CLAP's effect on collage splice placement. Writes results incrementally."""
import copy, os, re, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np
from infinite_dj.db import TrackDB
from infinite_dj.mixer import render_collage, _clap_cos

START_RE = re.compile(r"@(\d+)s\]$")

def strip(tracks):
    out=[]
    for t in tracks:
        t2=copy.copy(t)
        t2.sections=[copy.copy(s) for s in t.sections]
        for s in t2.sections: s.embedding=None
        t2.cue_points=[copy.copy(c) for c in t.cue_points]
        for c in t2.cue_points: c.embedding=None
        out.append(t2)
    return out

def resolve(markers, clips, by_path):
    """markers and clips are appended 1:1 in place(); the marker carries the
    section start, so each clip maps to an exact original Section."""
    out=[]
    for m, c in zip(markers, clips):
        t = by_path.get(c["track"])
        mm = START_RE.search(m.label)
        sec = None
        if t and mm:
            want = int(mm.group(1))
            sec = min(t.sections, key=lambda s: abs(s.start - want)) if t.sections else None
        out.append((c, sec))
    return out

def overlap_sims(pairs):
    sims=[]
    for i,(a,sa) in enumerate(pairs):
        if not sa or not sa.embedding: continue
        for b,sb in pairs[i+1:]:
            if b["out_start"] >= a["out_end"]: break
            if not sb or not sb.embedding: continue
            sims.append(_clap_cos(sa.embedding, sb.embedding))
    return sims

def main():
    db=TrackDB("combined.db"); orig=db.load_all(); db.close()
    by={t.file_path:t for t in orig}
    pool=orig[:8]; pool_off=strip(pool)
    configs={
        "calm  chaos=0.0 ": dict(layers=2, min_seg_bars=12, max_seg_bars=28, chaos=0.0),
        "mid   chaos=0.35": dict(layers=3, min_seg_bars=6,  max_seg_bars=18, chaos=0.35),
        "wild  chaos=0.8 ": dict(layers=4, min_seg_bars=3,  max_seg_bars=12, chaos=0.8),
    }
    seeds=[3,17,42,101]
    print(f"{'config':<18} {'cond':<4} {'clips':>6} {'ovl':>5} {'overlapCLAP':>12} "
          f"{'identical':>10} {'posMatch':>9}", flush=True)
    for name, prof in configs.items():
        acc={"ON":[], "OFF":[]}; ident=0; posm=[]
        for seed in seeds:
            _,_,mon,con = render_collage(pool,     target_length_sec=120, seed=seed, **prof)
            _,_,mof,cof = render_collage(pool_off, target_length_sec=120, seed=seed, **prof)
            son=[(c["track"], round(c["out_start"],1)) for c in con]
            sof=[(c["track"], round(c["out_start"],1)) for c in cof]
            if son==sof: ident+=1
            n=min(len(son),len(sof))
            posm.append(sum(1 for x,y in zip(son,sof) if x==y)/max(1,n))
            acc["ON"].append((len(con),  overlap_sims(resolve(mon,con,by))))
            acc["OFF"].append((len(cof), overlap_sims(resolve(mof,cof,by))))
        for cond in ("ON","OFF"):
            nc=np.mean([r[0] for r in acc[cond]])
            s=[x for r in acc[cond] for x in r[1]]
            sim=f"{np.mean(s):.3f}" if s else "—"
            extra=f"{ident}/{len(seeds)}" if cond=="ON" else ""
            pm=f"{np.mean(posm):.0%}" if cond=="ON" else ""
            print(f"{name:<18} {cond:<4} {nc:>6.1f} {len(s):>5} {sim:>12} "
                  f"{extra:>10} {pm:>9}", flush=True)

main()
