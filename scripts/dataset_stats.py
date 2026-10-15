"""Compute MIT-BIH statistics and render presentation/index.html from presentation/template.html."""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
MIT = ROOT / "data" / "mitdb"
PTB = ROOT / "data" / "ptbdb"
PRES = ROOT / "presentation"

AAMI = {**dict.fromkeys("NLRej", "N"), **dict.fromkeys("AaJS", "S"),
        **dict.fromkeys("VE", "V"), "F": "F", **dict.fromkeys("/fQ", "Q")}
PACED = {"102", "104", "107", "217"}
DS1 = "101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230".split()
DS2 = "100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234".split()
CLASSES = ["N", "S", "V", "F", "Q"]
FS = 360
HALF = 180
SOS = butter(4, [0.5, 40], btype="band", fs=FS, output="sos")
SEED = 42


def record_stats(rec):
    hdr = wfdb.rdheader(str(MIT / rec))
    ann = wfdb.rdann(str(MIT / rec), "atr")
    raw = Counter(s for s in ann.symbol if s in AAMI)
    cls = Counter(AAMI[s] for s in ann.symbol if s in AAMI)
    m = re.match(r"\s*(\d+)\s+([MF])", hdr.comments[0]) if hdr.comments else None
    return {
        "id": rec,
        "minutes": round(hdr.sig_len / FS / 60, 1),
        "leads": hdr.sig_name,
        "age": int(m.group(1)) if m else None,
        "sex": m.group(2) if m else None,
        "paced": rec in PACED,
        "ds": "DS1" if rec in DS1 else "DS2" if rec in DS2 else "exclu",
        "classes": {c: cls.get(c, 0) for c in CLASSES},
        "raw": dict(raw),
        "_ann": (ann.sample, ann.symbol),
    }


def assign_splits(stats):
    # Split strict_v1 figé dans configs/splits.json (patient-aware, De Chazal DS1/DS2 adapté).
    sp = json.loads((ROOT / "configs" / "splits.json").read_text(encoding="utf-8"))
    where = {r: s for s in ("train", "val", "test") for r in sp[s]}
    for r in stats:
        r["split"] = "exclu" if r["paced"] else where[r["id"]]


def read_mlii(rec, sampfrom=0, sampto=None):
    r = wfdb.rdrecord(str(MIT / rec), sampfrom=sampfrom, sampto=sampto)
    ch = r.sig_name.index("MLII") if "MLII" in r.sig_name else 0
    return r.p_signal[:, ch]


def r2(a, k=2):
    return [round(float(v), k) for v in a]


def sample_beats(stats):
    out = {}
    for c in CLASSES:
        cand = [r for r in stats if not r["paced"] and r["classes"][c] >= 3]
        if not cand:
            continue
        r = max(cand, key=lambda r: r["classes"][c])
        samples, symbols = r["_ann"]
        idx = [i for i, s in enumerate(symbols) if AAMI.get(s) == c]
        i = idx[len(idx) // 2]
        x = read_mlii(r["id"], max(0, int(samples[i]) - HALF), int(samples[i]) + HALF)
        x = (x - x.mean()) / (x.std() + 1e-8)
        out[c] = {"record": r["id"], "symbol": symbols[i], "y": r2(x[::2], 3)}
    return out


def filtering_demo(stats):
    """Raw vs band-pass filtered signal on the record with the strongest baseline drift."""
    best, best_drift = None, -1
    for r in stats:
        if r["paced"]:
            continue
        x = read_mlii(r["id"], 0, FS * 8)
        d = float(np.std(x - sosfiltfilt(SOS, x)))
        if d > best_drift:
            best, best_drift = r["id"], d
    x = read_mlii(best, 0, FS * 8)
    return {"record": best, "fs": FS, "raw": r2(x[::2]), "filt": r2(sosfiltfilt(SOS, x)[::2])}


def normalization_demo(stats):
    """One normal beat from the lowest- and highest-amplitude patients, before/after z-score."""
    amp = []
    for r in stats:
        if r["paced"]:
            continue
        s, sym = r["_ann"]
        i = [k for k, v in enumerate(sym) if v == "N" and 2 * FS < s[k] < FS * 110]
        if len(i) < 20:
            continue
        x = sosfiltfilt(SOS, read_mlii(r["id"], 0, FS * 120))
        peaks = [int(s[k]) for k in i[:60]]
        amp.append((float(np.median([abs(x[p]) for p in peaks])), r["id"], x, peaks[len(peaks) // 2]))
    amp.sort(key=lambda t: t[0])
    out = []
    for _, rec, x, p in (amp[0], amp[-1]):
        w = x[p - HALF:p + HALF]
        out.append({"record": rec, "raw": r2(w[::2]), "z": r2((w - w.mean()) / (w.std() + 1e-8))})
    return out


def rr_demo(stats):
    """Window around an isolated premature ventricular beat, with its RR intervals."""
    for r in sorted(stats, key=lambda r: -r["classes"]["V"]):
        if r["paced"]:
            continue
        s, sym = r["_ann"]
        for i in range(3, len(sym) - 3):
            if sym[i] != "V" or any(sym[j] != "N" for j in range(i - 3, i + 3) if j != i):
                continue
            pre, post = (s[i] - s[i - 1]) / FS, (s[i + 1] - s[i]) / FS
            if pre < 0.75 * (s[i - 1] - s[i - 2]) / FS and post > 1.2 * pre:
                a, b = int(s[i - 3]) - 40, int(s[i + 2]) + 40
                x = sosfiltfilt(SOS, read_mlii(r["id"], 0, b + FS))[a:b]
                return {"record": r["id"], "y": r2(x[::2]), "pre": round(float(pre), 3), "post": round(float(post), 3),
                        "ratio": round(float(pre / post), 3),
                        "peaks": [{"x": int(s[j] - a) // 2, "sym": sym[j]} for j in range(i - 3, i + 3)]}
    return None


def nva_demo(stats, seconds=6):
    """6 s sans anomalie vs 6 s avec exactement 2 extrasystoles ventriculaires prématurées (pour illustrer 0 vs 1)."""
    def window(rec, a, inside):
        x = sosfiltfilt(SOS, read_mlii(rec, 0, a + FS * (seconds + 1)))[a:a + FS * seconds]
        return {"record": rec, "y": r2(x[::2]), "peaks": [{"x": int(s - a) // 2, "sym": sy} for s, sy in inside]}

    normal = abnormal = None
    for r in sorted(stats, key=lambda r: -r["classes"]["V"]):
        if r["paced"]:
            continue
        samples, symbols = np.asarray(r["_ann"][0]), list(r["_ann"][1])
        for a in range(FS, int(samples[-1]) - FS * (seconds + 1), FS):
            lo, hi = np.searchsorted(samples, [a, a + FS * seconds])
            inside = [(int(s), sy) for s, sy in zip(samples[lo:hi], symbols[lo:hi]) if sy in AAMI]
            syms = [sy for _, sy in inside]
            if len(inside) < 5:
                continue
            if normal is None and r["id"] == "100" and set(syms) == {"N"}:  # enregistrement 100 : signal propre
                normal = window(r["id"], a, inside)
            if abnormal is None and syms.count("V") == 2 and set(syms) <= {"N", "V"}:
                v = [i for i, sy in enumerate(syms) if sy == "V"]
                if all(i > 1 and inside[i][0] - inside[i - 1][0] < 0.85 * (inside[i - 1][0] - inside[i - 2][0]) for i in v):
                    abnormal = window(r["id"], a, inside)
        if normal and abnormal:
            return {"normal": normal, "abnormal": abnormal}
    return None


def split_beats():
    """Comptes réels du cache strict_v1 (battements retenus après découpe en fenêtres)."""
    m = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text(encoding="utf-8"))["splits"]
    return {s: {"n": v["n_beats"], "normal": v["n_normal"], "abnormal": v["n_abnormal"], "patients": len(v["records"])}
            for s, v in m.items()}


def strip(rec="100", seconds=8):
    x = read_mlii(rec, 0, FS * seconds)
    ann = wfdb.rdann(str(MIT / rec), "atr", sampto=FS * seconds)
    return {"record": rec, "fs": FS, "y": r2(x[::2]),
            "peaks": [int(s // 2) for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI]}


def ptb_status():
    heas = list(PTB.rglob("*.hea")) if PTB.exists() else []
    return {"records_local": len(heas), "patients_local": len({p.parent.name for p in heas}), "records_total": 549}


def main():
    recs = sorted(p.stem for p in MIT.glob("*.hea"))
    stats = [record_stats(r) for r in recs]
    assign_splits(stats)
    totals = {c: sum(r["classes"][c] for r in stats) for c in CLASSES}
    raw_total = Counter()
    for r in stats:
        raw_total.update(r["raw"])
    split = {s: {c: sum(r["classes"][c] for r in stats if r["split"] == s) for c in CLASSES}
             for s in ("train", "val", "test", "exclu")}
    data = {
        "n_records": len(stats), "fs": FS, "half": HALF,
        "total_minutes": round(sum(r["minutes"] for r in stats), 1),
        "totals": totals, "raw": dict(raw_total), "aami_map": dict(AAMI),
        "split": split,
        "split_patients": {s: sum(r["split"] == s for r in stats) for s in split},
        "records": [{k: v for k, v in r.items() if k != "_ann"} for r in stats],
        "beats": sample_beats(stats), "strip": strip(),
        "filter": filtering_demo(stats), "norm": normalization_demo(stats), "rr": rr_demo(stats),
        "nva": nva_demo(stats), "split_beats": split_beats(),
        "ptb": ptb_status(),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_stats.json").write_text(json.dumps(data), encoding="utf-8")
    html = (PRES / "template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data))
    (PRES / "index.html").write_text(html, encoding="utf-8")
    tot = lambda s: sum(split[s].values())
    print("totals", totals)
    print("train/val/test/exclu beats:", tot("train"), tot("val"), tot("test"), tot("exclu"))
    print("patients:", data["split_patients"], "| rr demo:", bool(data["rr"]), "| ptb:", data["ptb"])


if __name__ == "__main__":
    main()
