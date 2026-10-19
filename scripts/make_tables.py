#!/usr/bin/env python3
"""Regenerate every table of the paper from the released results, as plain text and as LaTeX.

    python scripts/make_tables.py                          # writes tables/<name>.txt and tables/<name>.tex
    python scripts/make_tables.py --compare-tex <dir>      # also compare cell by cell with the paper's LaTeX source
                                                           # (<dir> = the arXiv source folder holding tables/, sections/)

Tables (paper label -> file):
    tab:reporting       reporting      reporting set (initial states 15-19), every arm simulated there
    tab:ablation        ablation       labels x representation matching x offset, first policy call
    tab:reference       reference      selection set (initial states 10-14): parent-gated reference vs the verdict
    tab:routes          routes         routes before the parent-taught verdict
    tab:search          search         every install candidate at the first policy call (Appendix A)
    tab:liberosafety    libero_safety  LIBERO-Safety semantic suite, first-call [STOP] rate (Appendix H)

Every cell is computed from data/ and results/ with the same rules as scripts/verify_release.py (they share
scripts/claims/). Rates print with two decimals, rounded half up as in the paper
(the routes table keeps the paper's Python formatting, see f2_py). CPU only, a few seconds.
"""
from __future__ import annotations

import argparse
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "claims"))

import part_b_install as B  # noqa: E402
import part_c_firstcall as Cf  # noqa: E402
import part_d_sim as D  # noqa: E402
import part_e_external as E  # noqa: E402

OUT = ROOT / "tables"
DASH = "--"


def f2(x) -> str:
    """Two decimals, half up (0.125 -> 0.13), as printed in the paper."""
    return str(Decimal(repr(round(float(x), 9))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def f2_py(x) -> str:
    """Two decimals with Python's formatting (an exact binary half rounds to even: 0.125 -> 0.12). The routes table
    was generated this way in the paper; every other table rounds half up."""
    return f"{float(x):.2f}"


def bold(s: str) -> str:
    return r"\textbf{" + s + "}"


# ------------------------------------------------------------------------------------------------ tables
# Each builder returns (columns, rows, caption) with rows = [(label, [cell, ...]), ...]; LaTeX markup inside a cell
# (\textbf, \,) is kept for the .tex file and stripped for the .txt file.

def t_reporting():
    arms = [("Base \\nora", "base"), None, ("PTV, seed 0", "s0"), ("PTV, seed 1", "s1"), ("PTV, seed 2", "s2"),
            ("PTV, replay doubled", "I9"), ("PTV, construction labels", "I11"), None,
            ("PTV, seed 2, no offset", "argmax")]
    bar = D.bar("G17_SIM_LIBERO_SUCCESS_MIN")
    rows = []
    for a in arms:
        if a is None:
            rows.append(None)
            continue
        label, k = a
        m = D.M(D.REP[k])
        if k == "base":
            b = DASH
        elif k == "argmax":
            b = str(D.as_int(D.b_of(k)))
        else:
            b = f"{D.b_of(k):+.2f}"
        lib = f2(m["lib"])
        cells = [b, f2(m["nt"]), f2(m["demo"]), f2(m["ht"]), f2(m["abort"]), f2(m["bt"]),
                 f2(m["para"]) if "para" in m else DASH, f2(m["pooled"]) if "pooled" in m and k != "base" else DASH,
                 bold(lib) if (k != "base" and m["lib"] >= bar - 1e-12) else lib]
        rows.append((label, cells))
    cols = ["arm", "b", "never-trained", "15-cmd set", "harmful twins", "abort", "benign twins", "paraphrases",
            "pooled benign", "LIBERO success"]
    return cols, rows, ("Reporting set (initial states 15-19), 310 episodes per arm (base 230). Refusal columns are "
                        "episode rates; pooled benign is prompt-level over 64 benign prompts; abort = fraction of 18 "
                        "harmful mid-task swaps after which the arm stops. Bold: meets the 0.80 LIBERO bar.")


def t_ablation():
    spec = [("construction", "no", "argmax", "I1"), ("parent", "no", "argmax", "I2"),
            ("parent", "no", "$b_{95}$", "I2@b95"), None,
            ("parent", "yes", "argmax", "I4"), ("parent", "yes", "$b_{95}$", "I4@b95"),
            ("construction", "yes", "argmax", "I11"), ("construction", "yes", "$b_{95}$", "I11b95")]
    rows = []
    for s in spec:
        if s is None:
            rows.append(None)
            continue
        lab, mat, off, arm = s
        h = [f2(Cf.H(arm, p)) for p in Cf.COLS]
        rows.append((lab, [mat, off, h[0], h[1], h[2], f"{h[3]} / {h[4]}", h[5]]))
    cols = ["labels", "repr. matching", "offset", "never-trained", "benign twins", "trained tasks",
            "new harm / new twins", "blind benign"]
    return cols, rows, "First-call hold rates, seed 0, selection-era pools (24 / 15 / 10 / 16 / 12 / 14 commands)."


def _cost_parent(mode, word):
    return f"parent {word} {f2(B.gate_cost(mode, 'lat'))}\\,s, {B.gate_cost(mode, 'mem'):.1f}\\,GB"


def t_reference():
    rows = []
    for label, arm, word in (("Parent-gated, graded judge", "parent_judge", "call"),
                             ("Parent-gated, linear direction", "parent_read", "pass")):
        cells = [f2(B.g1(arm, "harm_heldout")), f2(B.g1(arm, "twins_harm")), f2(B.g1(arm, "twins_benign")),
                 f2(B.g1(arm, "benign_pooled")), f2(B.succ(B.s78(B.P8, "g1", arm, "gate")))]
        if arm == "parent_judge":
            ab = B.g1_abort(arm, True)
            cells.append(f2(B.still(ab) / len(ab)))
        else:
            cells.append(DASH)
        cells.append(_cost_parent(arm, word))
        rows.append((label, cells))
    rows.append(None)
    for label, k in (("PTV, seed 0", "s0"), ("PTV, seed 2", "s2"), ("PTV, replay doubled", "I9")):
        m = D.M(D.SEL[k])
        rows.append((label, [f2(m["nt"]), f2(m["ht"]), f2(m["bt"]), f2(m["benign_ep"]), f2(m["lib"]),
                             f2(m["abort"]), f"{round(m['tok_per_call'])} tokens per call"]))
    rows.append(None)
    m = D.M(D.SEL["I7"])
    rows.append(("Parent weights merged + PTV", [f2(m["nt"]), f2(m["ht"]), f2(m["bt"]), f2(m["benign_ep"]),
                                                 f2(m["lib"]), DASH,
                                                 f"{round(D.all_equal(m['restore_mb']))}\\,MB merged delta"]))
    cols = ["system", "never-trained", "harmful twins", "benign twins", "benign episodes", "LIBERO success",
            "abort", "run-time cost"]
    return cols, rows, "Selection set (initial states 10-14), where base NORA scores 0.84 LIBERO success."


def t_routes():
    P6, P7, P8 = B.P6, B.P7, B.P8
    rows = [("Base policy", [f2_py(B.succ(B.s78(P7, "base", "none", "gate")))] +
             [f2_py(B.ref(P7, "base", "none", p)) for p in
              ("harm_heldout", "demo_harm", "twins_harm", "twins_benign", "benign_pooled")]), None]
    rows.append(("Scripted hold on the read-out",
                 [f2_py(B.succ(B.s6("pd_forced", "id_task"))), DASH, f2_py(B.refuse_rate(B.s6("pd_forced", "demo_harm"))),
                  DASH, DASH, f2_py(B.refuse_rate(B.s6("pd_forced", "benign")))]))
    st = "pd_refadd_strong"
    rows.append(("Refusal-direction steering",
                 [f2_py(B.succ(B.s6(st, "id_task"))), DASH,
                  f"{f2_py(B.refuse_rate(B.s6(st, 'demo_harm')))} ({f2_py(B.coll(B.s6(st, 'demo_harm')))})", DASH, DASH,
                  f"{f2_py(B.refuse_rate(B.s6(st, 'benign')))} ({f2_py(B.coll(B.s6(st, 'benign')))})"]))
    rows.append(None)
    for label, f, model in (("Free LoRA, 38 commands", P7, "install"), ("Random-concept control", P7, "control"),
                            ("Free LoRA, 338 commands", P8, "free_lora_300")):
        lib = DASH if model == "free_lora_300" else f2_py(B.succ(B.s78(f, model, "none", "gate")))
        rows.append((label, [lib] + [f2_py(B.ref(f, model, "none", p)) for p in
                                     ("harm_heldout", "demo_harm", "twins_harm", "twins_benign", "benign_pooled")]))
    cols = ["route", "LIBERO success", "never-trained categories", "15-command set", "harmful twins",
            "benign twins", "benign over-refusal"]
    return cols, rows, ("Routes before the parent-taught verdict, episode rates (selection-era pools). Parentheses: "
                        "decoder-collapse rate. A dash marks a pool that route never ran.")


SEARCH_SPEC = [  # id, format, labels, parent's help, offset, arm key in the first-call rows (None = separator)
    ("--", "none (base \\nora)", "--", "--", "--", "base"),
    ("--", "free, 338 cmds", "construction", "--", "--", "free300"),
    ("I0", "free", "construction", "--", "--", "I0"),
    ("I3", "free", "construction", "matching", "--", "I3"),
    ("I5", "free", "construction", "matching $\\times 3$", "--", "I5"), None,
    ("I1", "verdict", "construction", "--", "argmax", "I1"),
    ("I2", "verdict", "parent", "--", "argmax", "I2"),
    ("I2", "verdict", "parent", "--", "$b_{90}$", "I2b"),
    ("I2", "verdict", "parent", "--", "$b_{95}$", "I2@b95"), None,
    ("I4", "verdict", "parent", "matching", "argmax", "I4"),
    ("I4", "verdict", "parent", "matching", "$b_{90}$", "I4b"),
    ("I4", "verdict", "parent", "matching", "$b_{95}$", "I4@b95"),
    ("I4s1", "verdict", "parent", "matching, seed 1", "argmax", "I4s1"),
    ("I4s1", "verdict", "parent", "matching, seed 1", "$b_{95}$", "I4s1b95"),
    ("I4s2", "verdict", "parent", "matching, seed 2", "argmax", "I4s2"),
    ("I4s2", "verdict", "parent", "matching, seed 2", "$b_{95}$", "I4s2b95"),
    ("I9", "verdict", "parent", "matching, replay $\\times 2$", "argmax", "I9"),
    ("I9", "verdict", "parent", "matching, replay $\\times 2$", "$b_{95}$", "I9b95"),
    ("I10", "verdict", "parent", "matching, lr $5{\\times}10^{-5}$", "argmax", "I10"),
    ("I10", "verdict", "parent", "matching, lr $5{\\times}10^{-5}$", "$b_{95}$", "I10b95"),
    ("I11", "verdict", "construction", "matching", "argmax", "I11"),
    ("I11", "verdict", "construction", "matching", "$b_{95}$", "I11b95"), None,
    ("I6", "verdict", "parent", "rank-32 weights", "argmax", "I6"),
    ("I6", "verdict", "parent", "rank-32 weights", "$b_{90}$", "I6b"),
    ("I6", "verdict", "parent", "rank-32 weights", "$b_{95}$", "I6@b95"),
    ("I7", "verdict", "parent", "rank-32 weights, matching", "argmax", "I7"),
    ("I7", "verdict", "parent", "rank-32 weights, matching", "$b_{90}$", "I7b"),
    ("I7", "verdict", "parent", "rank-32 weights, matching", "$b_{95}$", "I7@b95"),
    ("I8", "verdict", "parent", "rank-128 weights", "argmax", "I8"),
    ("I8", "verdict", "parent", "rank-128 weights", "$b_{90}$", "I8b"),
    ("I8", "verdict", "parent", "rank-128 weights", "$b_{95}$", "I8@b95"),
]


def t_search():
    rows = []
    for s in SEARCH_SPEC:
        if s is None:
            rows.append(None)
            continue
        iid, fmt, lab, helpw, off, arm = s
        h = [f2(Cf.H(arm, p)) for p in Cf.COLS]
        rows.append((iid, [fmt, lab, helpw, off, h[0], h[1], h[2], f"{h[3]} / {h[4]}", h[5]]))
    cols = ["id", "format", "labels", "parent's help", "offset", "NT", "BT", "TR", "NH / NB", "BB"]
    return cols, rows, ("Every install candidate at the first policy call: hold rates on never-trained categories "
                        "(NT), benign twins (BT), trained tasks (TR), confirmation-set new harm / new benign twins "
                        "(NH / NB) and the blind benign set (BB). Rows at b_95 for I2, I4, I6, I7, I8 are computed "
                        "offline from the stored round-3 decision logits.")


# Quoted from the LIBERO-Safety paper (Cui et al. 2026); not recomputable here.
LS_REPORTED = [("RoboBrain2.0-7B (reported)", ["0.80", "0.40", "0.20", DASH, DASH]),
               ("RynnBrain-CoP (reported)", ["0.56", "0.72", "0.36", DASH, DASH])]


def t_libero_safety():
    rows = []
    for label, arm in (("base \\nora", "base"), ("parent-taught verdict, seed 1", "s1"),
                       ("parent-taught verdict, seed 2", "s2")):
        rows.append((label, [f2(E.stop1(E.ls_cell(arm, **kw))) for _, kw in E._COLS]))
    rows += LS_REPORTED
    cols = ["", "explicit", "common-sense", "contextual", "clean ten", "benign twins"]
    return cols, rows, ("First-call [STOP] rate on the LIBERO-Safety semantic suite: 50 harmful episodes per level, "
                        "100 on the clean ten, 150 benign-twin episodes per arm. The two reported rows are quoted "
                        "from the LIBERO-Safety paper.")


TABLES = {  # name: (builder, paper label, paper file holding the tabular)
    "reporting": (t_reporting, "tab:reporting", "tables/reporting.tex"),
    "ablation": (t_ablation, "tab:ablation", "tables/ablation.tex"),
    "reference": (t_reference, "tab:reference", "tables/reference.tex"),
    "routes": (t_routes, "tab:routes", "tables/routes.tex"),
    "search": (t_search, "tab:search", "sections/appendix.tex"),
    "libero_safety": (t_libero_safety, "tab:liberosafety", "sections/appendix.tex"),
}


# ------------------------------------------------------------------------------------------------ output

def plain(s: str) -> str:
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = s.replace("\\,", " ").replace("\\nora", "NORA").replace("$", "").replace("{\\times}", "x")
    s = s.replace("\\times ", "x").replace("b_{95}", "b95").replace("b_{90}", "b90").replace("10^{-5}", "10^-5")
    return s


def to_text(name, cols, rows, caption):
    body = [[plain(c) for c in cols]] + [[plain(r[0])] + [plain(c) for c in r[1]] for r in rows if r]
    w = [max(len(row[i]) for row in body) for i in range(len(cols))]
    fmt = lambda row: "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(row))  # noqa: E731
    lines = [f"Table {name}", plain(caption), "", fmt(body[0]), "  ".join("-" * x for x in w)]
    lines += [fmt(r) for r in body[1:]]
    return "\n".join(lines) + "\n"


def to_tex(name, label, cols, rows, caption):
    spec = "@{}l" + "c" * (len(cols) - 1) + "@{}"
    out = [f"% Generated by scripts/make_tables.py from results/; paper label {label}.",
           "\\begin{tabular}{" + spec + "}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
    for r in rows:
        out.append("\\addlinespace[2pt]" if r is None else " & ".join([r[0]] + r[1]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ comparison

def _norm(c: str) -> str:
    c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", c.strip())
    return re.sub(r"\s+", " ", c.replace("~", " ")).strip()


def paper_rows(src: Path, label: str):
    """Data rows of the tabular that carries \\label{label} in a paper tex file."""
    text = src.read_text()
    i = text.index("\\label{" + label + "}")
    a = text.index("\\midrule", i)
    b = text.index("\\bottomrule", a)
    rows = []
    for line in text[a:b].splitlines()[1:]:
        line = line.split("%")[0].strip() if not line.strip().startswith("\\addlinespace") else ""
        if "&" in line:
            cells = [_norm(c) for c in line.rstrip("\\").rstrip().rstrip("\\").split("&")]
            rows.append(cells)
    return rows


def compare(name, rows, tex_dir: Path):
    _, label, rel = TABLES[name]
    paper = paper_rows(tex_dir / rel, label)
    ours = [[_norm(r[0])] + [_norm(c) for c in r[1]] for r in rows if r]
    n_cells, diffs = 0, []
    if len(paper) != len(ours):
        diffs.append(f"row count: paper {len(paper)}, regenerated {len(ours)}")
    for i, (p, o) in enumerate(zip(paper, ours)):
        if len(p) != len(o):
            diffs.append(f"row {i + 1} ({o[0]}): {len(p)} cells in the paper, {len(o)} regenerated")
            continue
        for j, (pc, oc) in enumerate(zip(p, o)):
            n_cells += 1
            if pc != oc:
                diffs.append(f"row {i + 1} ({o[0]}), column {j + 1}: paper {pc!r}, regenerated {oc!r}")
    return n_cells, diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compare-tex", type=Path, help="arXiv LaTeX source folder (holding tables/ and sections/)")
    ap.add_argument("--only", nargs="*", choices=sorted(TABLES), help="build only these tables")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    total, bad = 0, 0
    for name, (build, label, _) in TABLES.items():
        if a.only and name not in a.only:
            continue
        cols, rows, caption = build()
        (OUT / f"{name}.txt").write_text(to_text(name, cols, rows, caption))
        (OUT / f"{name}.tex").write_text(to_tex(name, label, cols, rows, caption))
        msg = f"{name:<14} -> tables/{name}.txt, tables/{name}.tex"
        if a.compare_tex:
            n, diffs = compare(name, rows, a.compare_tex)
            total += n
            bad += len(diffs)
            msg += f"   compared {n} cells with the paper: {'identical' if not diffs else f'{len(diffs)} differ'}"
            for d in diffs:
                msg += "\n    " + d
        print(msg)
    if a.compare_tex:
        print(f"\n{total} cells compared, {bad} differences")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
