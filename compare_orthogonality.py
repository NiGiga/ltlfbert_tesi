#!/usr/bin/env python3
"""
Genera i dataset randltl e gp (se mancanti) e confronta quale è più ortogonale.

Uso:
  python compare_orthogonality.py --quick
  python compare_orthogonality.py --formulas 5000
  python compare_orthogonality.py --skip-generate
  python compare_orthogonality.py --ks 4 5 --max-formulas 1000
"""
import argparse
import importlib
import inspect
import json
import os
from functools import reduce

import numpy as np
import spot

AP = ["p0", "p1", "p2", "p3"]
A = len(AP)


# ═════════════ Valutatore LTLf vettorizzato (X plain = weak, X[!] = strong) ═════════════
def ev(n, S, V):
    """S: [N,T,A] bool, V: [N,T] maschera (t < len). Ritorna [N,T] con il valore in ogni t."""
    k = n[0]
    if k == "ap": return S[:, :, n[1]] & V
    if k == "tt": return V.copy()
    if k == "ff": return np.zeros_like(V)
    a = ev(n[1], S, V)
    if k == "!": return ~a & V
    if k == "Xs":
        o = np.zeros_like(a); o[:, :-1] = a[:, 1:]; return o & V
    if k == "Xw":
        o = np.ones_like(a); o[:, :-1] = a[:, 1:] | ~V[:, 1:]; return o & V
    if k == "F": return np.logical_or.accumulate(a[:, ::-1], axis=1)[:, ::-1] & V
    if k == "G": return np.logical_and.accumulate((a | ~V)[:, ::-1], axis=1)[:, ::-1] & V
    b = ev(n[2], S, V)
    if k == "&": return a & b
    if k == "|": return a | b
    if k == "->": return (~a | b) & V
    if k == "U":
        u = np.zeros_like(a); T = a.shape[1]; u[:, T-1] = b[:, T-1]
        for t in range(T-2, -1, -1):
            u[:, t] = b[:, t] | (a[:, t] & u[:, t+1])
        return u & V
    raise ValueError(f"nodo AST sconosciuto: {k}")


def universe(a_n, kmax):
    """Tutte le tracce di lunghezza 1..kmax su a_n proposizioni (kmax=4, a_n=4: 69904 tracce)."""
    Ss, Ls = [], []
    for l in range(1, kmax + 1):
        n = (2 ** a_n) ** l
        idx = np.arange(n)
        bits = ((idx[:, None] >> np.arange(a_n * l)) & 1).astype(bool).reshape(n, l, a_n)
        S = np.zeros((n, kmax, a_n), bool); S[:, :l] = bits
        Ss.append(S); Ls.append(np.full(n, l))
    S = np.concatenate(Ss); L = np.concatenate(Ls)
    return S, np.arange(kmax)[None, :] < L[:, None]


# ═════════════ Formula Spot -> AST (da verificare sulla tua versione di Spot) ═════════════
def spot_to_ast(f, ap=AP):
    k = f.kind()
    if k == spot.op_ap: return ("ap", ap.index(f.ap_name()))
    if k == spot.op_tt: return ("tt",)
    if k == spot.op_ff: return ("ff",)
    ch = [spot_to_ast(c, ap) for c in f]
    if k == spot.op_Not: return ("!", ch[0])
    if k == spot.op_X: return ("Xw", ch[0])
    if hasattr(spot, "op_strong_X") and k == spot.op_strong_X: return ("Xs", ch[0])
    if k == spot.op_F: return ("F", ch[0])
    if k == spot.op_G: return ("G", ch[0])
    if k == spot.op_U: return ("U", ch[0], ch[1])
    if k == spot.op_Implies: return ("->", ch[0], ch[1])
    if k == spot.op_Equiv:
        a, b = ch; return ("|", ("&", a, b), ("&", ("!", a), ("!", b)))
    if k == spot.op_Xor:
        a, b = ch; return ("|", ("&", a, ("!", b)), ("&", ("!", a), b))
    if k == spot.op_And: return reduce(lambda x, y: ("&", x, y), ch)
    if k == spot.op_Or: return reduce(lambda x, y: ("|", x, y), ch)
    raise ValueError(f"operatore non gestito: {f}")


def formula_to_ast(s):
    f = spot.unabbreviate(spot.formula(s), "WMR")   # W, M, R -> U / G
    return spot_to_ast(f)


# ═════════════ Metriche ═════════════
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def universe_metrics(asts, S, V, rng, n_pairs=20000):
    """Metriche formula-formula sull'universo esaustivo di tracce."""
    U, N = len(S), len(asts)
    cover = np.zeros(U, np.int32)
    packed = []
    dens = np.zeros(N, np.int64)
    for i, a in enumerate(asts):
        v = ev(a, S, V)[:, 0]
        cover += v; dens[i] = v.sum(); packed.append(np.packbits(v))
    uniq = len({p.tobytes() for p in packed}) / N

    privfrac = np.zeros(N)
    for i, p in enumerate(packed):
        v = np.unpackbits(p, count=U).astype(bool)
        privfrac[i] = (v & (cover == 1)).sum() / max(dens[i], 1)

    jac, cont, nz = [], [], []
    for _ in range(n_pairs):
        i, j = rng.choice(N, 2, replace=False)
        if dens[i] == 0 or dens[j] == 0:
            continue
        inter = int(POP[packed[i] & packed[j]].sum())
        jac.append(inter / (dens[i] + dens[j] - inter))
        cont.append(inter / min(dens[i], dens[j]))
        nz.append(inter > 0)

    cp = cover[cover > 0]
    return {
        "semantic_unique_frac": uniq,
        "density_mean": float(dens.mean() / U),
        "empty_frac": float((dens == 0).mean()),
        "taut_frac": float((dens == U).mean()),
        "private_mass_mean": float(privfrac.mean()),
        "has_private_frac": float((privfrac > 0).mean()),
        "jaccard_mean": float(np.mean(jac)),
        "jaccard_p95": float(np.percentile(jac, 95)),
        "containment_mean": float(np.mean(cont)),
        "overlap_pair_frac": float(np.mean(nz)),
        "multiplicity_mean": float(cp.mean()) if len(cp) else 0.0,
    }


def parse_traces(strs, a_n):
    """'0101 1100 ...' -> (S [R,T,A], V [R,T])."""
    L = np.array([len(s.split()) for s in strs])
    T = int(L.max())
    S = np.zeros((len(strs), T, a_n), bool)
    for r, s in enumerate(strs):
        for t, st in enumerate(s.split()):
            S[r, t, :] = [c == "1" for c in st]
    return S, np.arange(T)[None, :] < L[:, None]


def pair_metrics(asts, rows, batch_sizes=(32, 256)):
    """Metriche formula-traccia sulle coppie reali del dataset.
    rows: lista di (indice_formula, stringa_traccia, label)."""
    fidx = np.array([r[0] for r in rows])
    lab = np.array([r[2] for r in rows], bool)
    S, V = parse_traces([r[1] for r in rows], A)
    N, R = len(asts), len(rows)

    M = np.zeros((N, R), bool)                      # M[i, r] = formula i accetta la traccia r
    for i, a in enumerate(asts):
        M[i] = ev(a, S, V)[:, 0]

    own = M[fidx, np.arange(R)]                     # la formula della riga accetta la sua traccia?
    pos = np.flatnonzero(lab)
    others = M[:, pos].sum(0) - own[pos]            # quante ALTRE formule accettano ogni positivo
    fn = float(others.mean() / (N - 1))

    out = {
        "label_agreement": float((own == lab).mean()),
        "others_accepting_pos_mean": float(others.mean()),
        "private_pos_frac": float((others == 0).mean()),
        "offdiag_density": fn,
    }
    for B in batch_sizes:
        out[f"expected_false_neg_B{B}"] = fn * (B - 1)
    return out


# ═════════════ Ricerca e chiamata delle funzioni di generazione ═════════════
# per ogni modalità: moduli in cui cercare (il primo è quello "di casa")
CANDIDATES = {
    "randltl": ["data.generate_dataset_randltl", "data.generate_dataset"],
    "gp":      ["data.generate_dataset", "data.ltlf_gp"],
}


def resolve(gen, module=None, func=None):
    mods = [module] if module else CANDIDATES[gen]
    for i, m in enumerate(mods):
        try:
            mod = importlib.import_module(m)
        except ImportError as e:
            print(f"  (modulo {m} non importabile: {e})")
            continue
        names = [func] if func else [f"generate_dataset_{gen}"]
        if i == 0 or module:        # il nome generico solo nel file "di casa" (evita la funzione legacy)
            names.append("generate_dataset")
        for n in names:
            if hasattr(mod, n):
                return getattr(mod, n), f"{m}.{n}"
    raise ImportError(f"Nessuna funzione trovata per '{gen}' in {mods}")


def _call(fn, **kw):
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kw)
    dropped = [k for k in kw if k not in params]
    if dropped:
        print(f"  (parametri non accettati da {fn.__name__}, ignorati: {dropped})")
    return fn(**{k: v for k, v in kw.items() if k in params})


def generate(args, gen, data_dir):
    if os.path.exists(os.path.join(data_dir, "train.json")) and not args.force_data:
        print(f"✅ [{gen}] dataset già presente in {data_dir}")
        return
    os.makedirs(data_dir, exist_ok=True)

    fn, where = resolve(gen, getattr(args, f"{gen}_module"), getattr(args, f"{gen}_func"))
    print(f"⏳ [{gen}] chiamo {where}")

    kw = dict(n_formulas=args.formulas, seed=args.seed, output_dir=data_dir)
    if gen == "gp":
        kw.update(k=args.gp_k, n_sub=args.n_sub, pop=args.gp_pop, gens=args.gp_gens,
                  dmin_frac=args.dmin_frac, dmax_frac=args.dmax_frac,
                  thr=args.gp_thr, verify=not args.no_verify)
    _call(fn, **kw)


# ═════════════ Caricamento dataset ═════════════
def load_rows(data_dir):
    rows = []
    for split in ("train", "val", "test", "ood"):
        p = os.path.join(data_dir, f"{split}.json")
        if os.path.exists(p):
            with open(p) as f:
                rows += json.load(f)
    return rows


def prepare(rows, n_eval, rng):
    """Campiona n_eval formule distinte, le converte in AST e raccoglie le loro coppie."""
    forms = sorted({r["formula"] for r in rows})
    forms = [forms[i] for i in rng.permutation(len(forms))]
    asts, keep = [], []
    for f in forms:
        if len(asts) >= n_eval:
            break
        try:
            asts.append(formula_to_ast(f)); keep.append(f)
        except Exception as e:  # noqa: BLE001
            print(f"  formula scartata ({e}): {f}")
    idx = {f: i for i, f in enumerate(keep)}
    prs = [(idx[r["formula"]], r["trace"], r["label"]) for r in rows if r["formula"] in idx]
    return keep, asts, prs


# ═════════════ Tabella di confronto ═════════════
# (chiave, descrizione, direzione: +1 più alto è meglio, -1 più basso è meglio, 0 informativa)
TABLE = [
    ("jaccard_mean", "Jaccard medio", -1),
    ("jaccard_p95", "Jaccard p95", -1),
    ("containment_mean", "Contenimento medio", -1),
    ("overlap_pair_frac", "Coppie con intersezione", -1),
    ("private_mass_mean", "Massa privata media", +1),
    ("has_private_frac", "Formule con traccia privata", +1),
    ("multiplicity_mean", "Molteplicità media", -1),
    ("semantic_unique_frac", "Formule semanticamente uniche", +1),
    ("empty_frac", "Formule vuote", -1),
    ("taut_frac", "Formule tautologiche", -1),
    ("density_mean", "Densità media", 0),
    ("offdiag_density", "Densità fuori diagonale (M)", -1),
    ("others_accepting_pos_mean", "Altre formule per positivo", -1),
    ("private_pos_frac", "Positivi privati", +1),
    ("expected_false_neg_B32", "Falsi negativi attesi B=32", -1),
    ("expected_false_neg_B256", "Falsi negativi attesi B=256", -1),
    ("label_agreement", "Accordo con etichette", 0),
    ("avg_len", "Lunghezza media formula (char)", 0),
]


def main():
    p = argparse.ArgumentParser(description="Confronto di ortogonalità randltl vs gp")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data-root", default="ltlfbert")
    p.add_argument("--formulas", type=int, default=5000)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--force-data", action="store_true", help="rigenera i dataset")
    p.add_argument("--skip-generate", action="store_true", help="solo analisi")
    p.add_argument("--max-formulas", type=int, default=2000,
                   help="formule valutate per dataset (stesso numero per entrambi)")
    p.add_argument("--ks", type=int, nargs="+", default=[4],
                   help="lunghezze max degli universi per le metriche (es. 4 5)")
    p.add_argument("--pairs", type=int, default=20000)
    # dove stanno le funzioni di generazione
    p.add_argument("--randltl-module", default=None)
    p.add_argument("--randltl-func", default=None)
    p.add_argument("--gp-module", default=None)
    p.add_argument("--gp-func", default=None)
    # parametri GP
    p.add_argument("--gp-k", type=int, default=4)
    p.add_argument("--n-sub", type=int, default=20000)
    p.add_argument("--gp-pop", type=int, default=1000)
    p.add_argument("--gp-gens", type=int, default=200)
    p.add_argument("--gp-thr", type=float, default=0.8)
    p.add_argument("--dmin-frac", type=float, default=0.002)
    p.add_argument("--dmax-frac", type=float, default=0.05)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.formulas, args.gp_pop, args.gp_gens = 1000, 200, 30
        args.n_sub, args.max_formulas = 5000, 500

    mode = "quick" if args.quick else "comp"
    gens = ["randltl", "gp"]
    dirs = {g: os.path.join(args.data_root, f"data_{mode}_{g}") for g in gens}

    # ── 1. Generazione ──
    if not args.skip_generate:
        for g in gens:
            generate(args, g, dirs[g])

    # ── 2. Caricamento e stesso numero di formule per entrambi ──
    all_rows = {g: load_rows(dirs[g]) for g in gens}
    n_forms = {g: len({r["formula"] for r in all_rows[g]}) for g in gens}
    n_eval = min(min(n_forms.values()), args.max_formulas)
    print(f"\nFormule distinte: {n_forms} → valuto {n_eval} formule per dataset")

    # ── 3. Metriche ──
    results = {}
    for g in gens:
        rng = np.random.default_rng(args.seed)
        keep, asts, prs = prepare(all_rows[g], n_eval, rng)
        res = {"n_formulas_eval": len(asts), "avg_len": float(np.mean([len(f) for f in keep]))}
        res.update(pair_metrics(asts, prs))
        for k in args.ks:
            S, V = universe(A, k)
            um = universe_metrics(asts, S, V, np.random.default_rng(args.seed), args.pairs)
            suffix = f"@k{k}" if len(args.ks) > 1 else ""
            res.update({f"{key}{suffix}": v for key, v in um.items()})
        results[g] = res

    # ── 4. Tabella ──
    print("\n" + "=" * 78 + "\nORTOGONALITÀ: randltl vs gp\n" + "=" * 78)
    print(f"{'metrica':<40}{'randltl':>12}{'gp':>12}   migliore")

    lines = []
    for key, label, d in TABLE:
        if key in results["gp"]:
            lines.append((key, label, d))
        else:
            for k in args.ks:
                kk = f"{key}@k{k}"
                if kk in results["gp"]:
                    lines.append((kk, f"{label} (k={k})", d))

    for key, label, d in lines:
        a, b = results["randltl"][key], results["gp"][key]
        if d == 0 or abs(a - b) < 1e-12:
            best = ""
        else:
            best = "gp" if (b > a) == (d > 0) else "randltl"
        print(f"{label:<40}{a:>12.4f}{b:>12.4f}   {best}")

    out = os.path.join(args.data_root, f"ortho_comparison_{mode}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()