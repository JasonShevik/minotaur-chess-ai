"""
Builds minotaur_jepa_experiments.xlsx: every training run under runs/step*/ with its settings, its final z probes,
its own untrained start, its evaluation metrics, head scores, left/right checks and token probes, plus paired
comparisons (as formulas), probe trajectories and a step-by-step summary.
Needs the local run folders (runs/ is not in git). Run from anywhere: python reports/build_workbook.py
openpyxl writes the formulas without values: open the file in Excel or LibreOffice (or recalculate it) to see them.
"""
import glob
import json
import os
import re

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")
ANALYSIS = os.path.join(RUNS, "analysis")
OUT = os.path.join(HERE, "minotaur_jepa_experiments.xlsx")

FONT = "Arial"
HEAD_FILL = PatternFill("solid", fgColor="1F3A4D")
GROUP_FILL = PatternFill("solid", fgColor="D9E4EA")
THIN = Side(style="thin", color="B0BEC5")

PROBES = [("Material", "val/material_balance"), ("Eval", "labeled/eval_win_prob"),
          ("Themes", "puzzles/theme/auc"), ("Best move", "labeled/best_move/auc")]
KINDS = [("lin", "linear"), ("kNN", "knn"), ("white", "knn_white")]


def step_of(folder):
    m = re.match(r"step(\d+)", folder)
    return int(m.group(1)) if m else None


def family(j):
    tv = j.get("token_variant", "squares")
    tl = j.get("transformer_layers", 4) or 0
    ro = j.get("readout") or "attention"
    gs = j.get("gat_style") or "split"
    if tv == "unpool":
        name = "F (unpool, " + ("A layout)" if tl > 0 else "C layout)")
    elif gs == "concat":
        name = "G1r (concat GAT + residual)" if j.get("gat_residual") else "G1 (concat GAT)"
        if ro != "mean":
            name += f" + {ro} readout"
    elif tl > 0:
        mixer = j.get("token_mixer") or "attention"
        if ro == "attention":
            name = "A (transformer + attention readout)"
        else:
            base = "E" if (j.get("pred_transformer_layers") or 0) > 0 else "B"
            name = base + {"attention": " (transformer + mean)", "mlp": "_mlp (per-token MLP + mean)",
                           "mixer": "_mixer (MLP-Mixer + mean)"}[mixer]
    else:
        if ro == "mean_max" or (j.get("readout_hidden") or 0) > 0:
            name = "G2 (split GAT + mean/max MLP readout)"
        elif (j.get("pred_transformer_layers") or 0) > 0:
            name = "D (C + predictor transformer)"
        else:
            name = "C (split GAT, mean readout)"
    if j.get("gat_sum"):
        name += " + sum branch"
    return name


def objectives(j):
    out = []
    if j.get("mask_weight", 1.0) > 0:
        out.append("masked squares")
    out.append("next position")
    if j.get("zpred_weight", 0) > 0:
        out.append("whole-board z from masked z")
    if j.get("inverse_weight", 0) > 0:
        out.append(f"move guess x{j['inverse_weight']:g}")
    if j.get("count_weight", 0) > 0:
        out.append("piece counts")
    if j.get("legal_weight", 0) > 0:
        out.append("legal moves (token head)")
    if j.get("decode_weight", 0) > 0:
        out.append("decode own z")
    if j.get("decode_next_weight", 0) > 0:
        out.append("decode predicted next z")
    if j.get("decode_masked_weight", 0) > 0:
        out.append("decode predicted whole z")
    if j.get("zlegal_weight", 0) > 0:
        out.append("legal moves on own z")
    if j.get("zlegal_next_weight", 0) > 0:
        out.append("legal moves on predicted next z")
    if j.get("zlegal_masked_weight", 0) > 0:
        out.append("legal moves on predicted whole z")
    judges = any(j.get(k, 0) > 0 for k in ("decode_next_weight", "decode_masked_weight",
                                           "zlegal_next_weight", "zlegal_masked_weight"))
    if judges:
        out.append("head fit on " + ("ONLINE z" if j.get("zhead_fit") == "online" else "target z"))
    if (j.get("rollout_siblings") or 1) > 1:
        out.append(f"{j['rollout_siblings']} sibling moves")
    return "; ".join(out)


def predictor(j):
    out = [{"tokens": "moves as square tokens", "squares": "moves as square names",
            "rank_file": "moves as files/ranks"}[j.get("move_encoding") or "tokens"]]
    if j.get("move_piece"):
        out.append("+ piece type")
    if j.get("next_residual") is False:
        out.append("non-residual next predictor")
    if (j.get("next_layers") or 2) != 2 or (j.get("next_hidden") or 1024) != 1024:
        out.append(f"next {j.get('next_layers')}x{j.get('next_hidden')}")
    if (j.get("pred_transformer_layers") or 0) > 0:
        out.append(f"mask predictor + transformer x{j['pred_transformer_layers']}")
    return "; ".join(out)


def orientation(j):
    out = []
    if j.get("orientation_anchor"):
        out.append("a1 anchor edge")
    if j.get("orientation_flag"):
        out.append("a1 node flag")
    if j.get("orientation_files"):
        out.append("file marks on nodes")
    if j.get("orientation_edges"):
        scale = j.get("mark_init_scale", 1.0) or 1.0
        out.append("left/right edge marks" + (f" x{scale:g} start" if scale != 1.0 else ""))
    return "; ".join(out) or "none (mirror-blind)"


def anticollapse(j):
    s = f"{j.get('anticollapse', 'sigreg')} {j.get('sigreg_weight', 0.05):g}"
    if (j.get("projector_layers") or 0) > 0:
        s += f"; projector {j['projector_layers']} layers"
    if (j.get("anticollapse_views") or "both") != "both":
        s += f"; {j['anticollapse_views']} view only"
    if j.get("anticollapse_square_mean"):
        s += "; + square mean"
    if (j.get("decorrelation_weight") or 0) > 0:
        s += f"; decorrelation {j['decorrelation_weight']:g}"
    return s


def rollout(j):
    start, top = j.get("rollout_start_steps", 1), j.get("rollout_max_steps", 4)
    stage = j.get("rollout_stage_positions", 250000)
    if stage >= 1e8 or start >= top:
        return f"{start} plies"
    return f"{start} -> {top} plies, +1 per {stage / 1000:g}k positions"


def read_history(path):
    rows = [json.loads(line) for line in open(path)]
    base = next((r["probe_baselines"] for r in rows if "probe_baselines" in r), None)
    probes = [(r["step"], r["probe"]["z"]) for r in rows if "probe" in r and "z" in r["probe"]]
    evals = {r["step"]: r["eval"] for r in rows if "eval" in r}
    train = [r for r in rows if "loss" in r]
    return base, probes, evals, train


def val(block, key, kind):
    if block is None or key not in block:
        return None
    v = block[key].get(kind)
    return None if v is None else round(float(v), 4)


def parse_tables(pattern, skip=()):
    """Rows of the markdown tables in matching analysis outputs, oldest file first."""
    files = sorted((f for f in glob.glob(os.path.join(ANALYSIS, pattern)) if os.path.basename(f) not in skip),
                   key=os.path.getmtime)
    for f in files:
        section_online = False
        for line in open(f, encoding="utf-8"):
            if line.startswith("###"):
                section_online = "--online" in line
            if not line.startswith("| step") or section_online:
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            yield os.path.basename(f), cells


def main():
    runs = []
    for hist in sorted(glob.glob(os.path.join(RUNS, "step*", "*", "history.jsonl"))):
        folder = os.path.dirname(hist)
        step_dir, name = os.path.basename(os.path.dirname(folder)), os.path.basename(folder)
        if step_of(step_dir) is None or not os.path.exists(os.path.join(folder, "run_config.json")):
            continue
        cfg = json.load(open(os.path.join(folder, "run_config.json")))
        j, rc = cfg["jepa_config"], cfg["run_config"]
        base, probes, evals, train = read_history(hist)
        if not probes:
            continue
        runs.append(dict(id=f"{step_dir}/{name}", step=step_of(step_dir), name=name, j=j, rc=rc,
                         base=base, probes=probes, evals=evals, train=train))
    runs.sort(key=lambda r: (r["step"], r["name"]))

    orient = {}
    for _, cells in parse_tables("orientation*_out.md", skip=("target_vs_online_out.md",)):
        if len(cells) >= 5 and cells[1].isdigit():
            lin, _, knn = cells[4].partition("/")
            orient[(cells[0], int(cells[1]))] = (float(cells[2]), float(cells[3]), float(lin), float(knn))
    zhead = {}
    for _, cells in parse_tables("zhead_check*_out.md"):
        if len(cells) >= 10:
            zhead[cells[0]] = cells[1:9]
    tokens = {}
    for fname, cells in parse_tables("token_probe*_out.md"):
        if len(cells) >= 8:
            run, _, snap = cells[0].partition(" @")
            tokens[(run, int(snap) if snap else 4000)] = [float(c.split("/")[0]) for c in cells[1:8]]

    wb = Workbook()
    readme = wb.active
    readme.title = "Read me"
    ws = wb.create_sheet("Runs")
    comp = wb.create_sheet("Comparisons")
    traj = wb.create_sheet("Trajectories")
    ori = wb.create_sheet("Orientation")
    tok = wb.create_sheet("Tokens")
    steps = wb.create_sheet("Steps")

    # ---------------- Runs ----------------
    head_keys = ["legal_precision", "legal_recall", "decode_pieces", "decode_fit_pieces", "decode_next_pieces",
                 "decode_masked_pieces", "zlegal_found", "zlegal_rejected", "zlegal_next_found",
                 "zlegal_next_rejected", "zlegal_masked_found", "zlegal_masked_rejected", "count_exact",
                 "inverse_accuracy"]
    head_names = ["Legal (token head) precision", "Legal (token head) recall", "Decode own z: pieces",
                  "Decoder fit: pieces", "Decode next z: pieces", "Decode whole z: pieces",
                  "Legal own z: found", "Legal own z: rejected", "Legal next z: found", "Legal next z: rejected",
                  "Legal whole z: found", "Legal whole z: rejected", "Piece counts all exact", "Move guess accuracy"]
    groups = [
        ("Run", ["Run ID", "Step", "Name", "Encoder", "Seed", "Steps trained", "Objectives", "Predictor and moves",
                 "Left/right signal", "Anti-collapse", "Rollout", "Batch"]),
        ("z probes, final (trainer's own probe set)",
         [f"{p} {k}" for p, _ in PROBES for k, _ in KINDS] + ["Puzzle move lin"]),
        ("Same encoder untrained (its step 0)",
         ["U material lin", "U material kNN", "U material white", "U eval lin", "U eval kNN", "U eval white",
          "U best move lin"]),
        ("Trained minus untrained (formulas)",
         ["d material lin", "d material kNN", "d material white", "d eval lin", "d eval kNN", "d eval white",
          "d best move lin"]),
        ("Evaluation (trainer's val subset)",
         ["RankMe", "RankMe centred", "LiDAR", "Online LiDAR", "Self-consistency", "Val mask loss",
          "Val next loss"]),
        ("Heads, last training batches", head_names),
        ("Heads on 2,048 random val positions (zhead_check)",
         ["Decoder real", "Decoder own z", "Decoder next z", "Decoder whole z", "Legal real", "Legal own z",
          "Legal next z", "Legal whole z"]),
        ("Left/right at the final step (target encoder)",
         ["Mirror gap z", "King side lin", "King side kNN", "King side lin at step 0"]),
        ("Token probes at the final step (linear)",
         ["Tok attacked", "Tok defended", "Tok en prise", "Tok pinned", "Tok legal-to", "Tok balance R2",
          "Tok mobility R2"]),
    ]
    headers = [h for _, hs in groups for h in hs]
    col = {h: i + 1 for i, h in enumerate(headers)}
    c0 = 1
    for title, hs in groups:
        ws.cell(row=1, column=c0, value=title)
        if len(hs) > 1:
            ws.merge_cells(start_row=1, start_column=c0, end_row=1, end_column=c0 + len(hs) - 1)
        c0 += len(hs)
    for i, h in enumerate(headers, start=1):
        ws.cell(row=2, column=i, value=h)

    for r_i, run in enumerate(runs, start=3):
        j, rc = run["j"], run["rc"]
        final_step, final = run["probes"][-1]
        untrained = run["base"]["untrained"]["z"] if run["base"] else None
        last_eval = run["evals"][max(run["evals"])] if run["evals"] else {}
        last_train = run["train"][-5:]
        row = {
            "Run ID": run["id"], "Step": run["step"], "Name": run["name"], "Encoder": family(j),
            "Seed": rc.get("seed"), "Steps trained": final_step, "Objectives": objectives(j),
            "Predictor and moves": predictor(j), "Left/right signal": orientation(j),
            "Anti-collapse": anticollapse(j), "Rollout": rollout(j), "Batch": rc.get("batch_size"),
            "Puzzle move lin": val(final, "puzzles/puzzle_move/auc", "linear"),
            "U material lin": val(untrained, "val/material_balance", "linear"),
            "U material kNN": val(untrained, "val/material_balance", "knn"),
            "U material white": val(untrained, "val/material_balance", "knn_white"),
            "U eval lin": val(untrained, "labeled/eval_win_prob", "linear"),
            "U eval kNN": val(untrained, "labeled/eval_win_prob", "knn"),
            "U eval white": val(untrained, "labeled/eval_win_prob", "knn_white"),
            "U best move lin": val(untrained, "labeled/best_move/auc", "linear"),
            "RankMe": last_eval.get("rankme"), "RankMe centred": last_eval.get("rankme_centered"),
            "LiDAR": last_eval.get("lidar"), "Online LiDAR": last_eval.get("online_lidar"),
            "Self-consistency": last_eval.get("self_consistency"),
            "Val mask loss": last_eval.get("val_loss_mask"), "Val next loss": last_eval.get("val_loss_next"),
        }
        for p, key in PROBES:
            for k, kind in KINDS:
                row[f"{p} {k}"] = val(final, key, kind)
        for key, name in zip(head_keys, head_names):
            vs = [t[key] for t in last_train if key in t]
            row[name] = round(sum(vs) / len(vs), 4) if vs else None
        if run["id"] in zhead:
            for name, v in zip(groups[6][1], zhead[run["id"]]):
                row[name] = v or None
        o = orient.get((run["id"], final_step))
        if o:
            row["Mirror gap z"], row["King side lin"], row["King side kNN"] = o[0], o[2], o[3]
        o0 = orient.get((run["id"], 0))
        if o0:
            row["King side lin at step 0"] = o0[2]
        t = tokens.get((run["id"], final_step))
        if t:
            for name, v in zip(groups[8][1], t):
                row[name] = v
        for h, v in row.items():
            if v is not None:
                ws.cell(row=r_i, column=col[h], value=v)
        for d, a, b in [("d material lin", "Material lin", "U material lin"),
                        ("d material kNN", "Material kNN", "U material kNN"),
                        ("d material white", "Material white", "U material white"),
                        ("d eval lin", "Eval lin", "U eval lin"), ("d eval kNN", "Eval kNN", "U eval kNN"),
                        ("d eval white", "Eval white", "U eval white"),
                        ("d best move lin", "Best move lin", "U best move lin")]:
            ca, cb = get_column_letter(col[a]), get_column_letter(col[b])
            ws.cell(row=r_i, column=col[d], value=f'=IF(OR({ca}{r_i}="",{cb}{r_i}=""),"",{ca}{r_i}-{cb}{r_i})')
    last_row = 2 + len(runs)

    style_sheet(ws, header_rows=2, freeze="D3")
    for h in headers:
        c = col[h]
        width = {"Run ID": 30, "Name": 22, "Encoder": 30, "Objectives": 46, "Predictor and moves": 30,
                 "Left/right signal": 24, "Anti-collapse": 18, "Rollout": 26}.get(h, 10)
        ws.column_dimensions[get_column_letter(c)].width = width
        fmt = None
        if h in ("RankMe", "RankMe centred", "LiDAR", "Online LiDAR"):
            fmt = "0.0"
        elif h.startswith(("Material", "Eval", "Themes", "Best move", "Puzzle", "U ", "d ", "Tok", "King",
                           "Mirror", "Self")):
            fmt = "0.00"
        elif h.startswith(("Val ",)):
            fmt = "0.0000"
        elif h in head_names:
            fmt = "0%"
        if fmt:
            for r in range(3, last_row + 1):
                ws.cell(row=r, column=c).number_format = fmt
    for h in ("d material lin", "d material kNN", "d material white", "d eval lin", "d eval kNN", "d eval white",
              "d best move lin"):
        letter = get_column_letter(col[h])
        ws.conditional_formatting.add(f"{letter}3:{letter}{last_row}", ColorScaleRule(
            start_type="num", start_value=-0.3, start_color="E8A09A", mid_type="num", mid_value=0,
            mid_color="FFFFFF", end_type="num", end_value=0.15, end_color="8FC79A"))
    ws.auto_filter.ref = f"A2:{get_column_letter(len(headers))}{last_row}"
    ws.cell(row=2, column=col["Run ID"]).comment = Comment(
        "Folder under runs/. Seeds 1 and 2 are separate folders (_seed1, _seed2).", "Claude")
    ws.cell(row=2, column=col["Material lin"]).comment = Comment(
        "Material balance (R2) on 8,000 validation positions. lin = ridge probe, kNN = k nearest neighbours by "
        "cosine, white = kNN after whitening the top 64 directions. Whitened kNN did not exist before step 7; "
        "earlier runs leave it blank.", "Claude")

    # ---------------- Comparisons ----------------
    comp_headers = ["Step", "What changes", "Arm", "Baseline", "d material lin", "d material kNN",
                    "d material white", "d eval lin", "d eval kNN", "d eval white", "d best move lin"]
    for i, h in enumerate(comp_headers, start=1):
        comp.cell(row=1, column=i, value=h)
    ids = {r["id"] for r in runs}
    r_i = 2
    for stp, what, arm, baseline in comparison_list():
        if arm not in ids or baseline not in ids:
            continue
        for i, v in enumerate([stp, what, arm, baseline], start=1):
            comp.cell(row=r_i, column=i, value=v)
        for i, h in enumerate(["Material lin", "Material kNN", "Material white", "Eval lin", "Eval kNN",
                               "Eval white", "Best move lin"], start=5):
            letter = get_column_letter(col[h])
            rng = f"Runs!${letter}$3:${letter}${last_row}"
            ids_rng = f"Runs!$A$3:$A${last_row}"
            a = f"INDEX({rng},MATCH($C{r_i},{ids_rng},0))"
            b = f"INDEX({rng},MATCH($D{r_i},{ids_rng},0))"
            comp.cell(row=r_i, column=i, value=f'=IFERROR(IF(OR({a}="",{b}=""),"",{a}-{b}),"")')
            comp.cell(row=r_i, column=i).number_format = "+0.00;-0.00;0.00"
        r_i += 1
    style_sheet(comp, header_rows=1, freeze="E2")
    for c, w in zip(range(1, 12), [6, 44, 32, 32] + [11] * 7):
        comp.column_dimensions[get_column_letter(c)].width = w
    for c in range(5, 12):
        letter = get_column_letter(c)
        comp.conditional_formatting.add(f"{letter}2:{letter}{r_i - 1}", ColorScaleRule(
            start_type="num", start_value=-0.15, start_color="E8A09A", mid_type="num", mid_value=0,
            mid_color="FFFFFF", end_type="num", end_value=0.15, end_color="8FC79A"))
    comp.auto_filter.ref = f"A1:K{r_i - 1}"

    # ---------------- Trajectories ----------------
    t_headers = ["Run ID", "Step", "Training step"] + [f"{p} {k}" for p, _ in PROBES for k, _ in KINDS]
    for i, h in enumerate(t_headers, start=1):
        traj.cell(row=1, column=i, value=h)
    r_i = 2
    for run in runs:
        points = []
        if run["base"]:
            points.append((0, run["base"]["untrained"]["z"]))
        points += run["probes"]
        for s, block in points:
            vals = [val(block, key, kind) for _, key in PROBES for _, kind in KINDS]
            for i, v in enumerate([run["id"], run["step"], s] + vals, start=1):
                if v is not None:
                    traj.cell(row=r_i, column=i, value=v)
            for i in range(4, len(t_headers) + 1):
                traj.cell(row=r_i, column=i).number_format = "0.00"
            r_i += 1
    style_sheet(traj, header_rows=1, freeze="D2")
    traj.column_dimensions["A"].width = 30
    traj.auto_filter.ref = f"A1:{get_column_letter(len(t_headers))}{r_i - 1}"

    # ---------------- Orientation ----------------
    o_headers = ["Run ID", "Training step", "Mirror gap z", "Mirror gap tokens", "King side lin", "King side kNN"]
    for i, h in enumerate(o_headers, start=1):
        ori.cell(row=1, column=i, value=h)
    for r_i, ((run_id, s), o) in enumerate(sorted(orient.items(), key=lambda kv: (step_key(kv[0][0]), kv[0])),
                                           start=2):
        for i, v in enumerate([run_id, s, *o], start=1):
            ori.cell(row=r_i, column=i, value=v)
            if i > 2:
                ori.cell(row=r_i, column=i).number_format = "0.000"
    style_sheet(ori, header_rows=1, freeze="C2")
    ori.column_dimensions["A"].width = 32
    ori.auto_filter.ref = f"A1:F{len(orient) + 1}"

    # ---------------- Tokens ----------------
    k_headers = ["Run ID", "Training step", "Attacked", "Defended", "En prise", "Pinned", "Legal-to", "Balance R2",
                 "Mobility R2"]
    for i, h in enumerate(k_headers, start=1):
        tok.cell(row=1, column=i, value=h)
    for r_i, ((run_id, s), t) in enumerate(sorted(tokens.items(), key=lambda kv: (step_key(kv[0][0]), kv[0])),
                                           start=2):
        for i, v in enumerate([run_id, s, *t], start=1):
            tok.cell(row=r_i, column=i, value=v)
            if i > 2:
                tok.cell(row=r_i, column=i).number_format = "0.000"
    style_sheet(tok, header_rows=1, freeze="C2")
    tok.column_dimensions["A"].width = 32
    tok.auto_filter.ref = f"A1:I{len(tokens) + 1}"

    # ---------------- Steps ----------------
    s_headers = ["Step", "Date", "Question", "Runs", "What we learned"]
    for i, h in enumerate(s_headers, start=1):
        steps.cell(row=1, column=i, value=h)
    for r_i, row in enumerate(step_list(), start=2):
        for i, v in enumerate(row, start=1):
            c = steps.cell(row=r_i, column=i, value=v)
            c.alignment = Alignment(wrap_text=True, vertical="top")
    style_sheet(steps, header_rows=1, freeze="B2")
    for c, w in zip("ABCDE", [7, 12, 48, 34, 80]):
        steps.column_dimensions[c].width = w

    # ---------------- Read me ----------------
    seed0 = next(r for r in runs if r["id"] == "step15/C_base")["base"]
    write_readme(readme, len(runs), seed0)

    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            for c in row:
                if c.font is None or c.font.name != FONT:
                    c.font = Font(name=FONT, size=c.font.size if c.font and c.font.size else 10,
                                  bold=c.font.bold if c.font else False,
                                  color=c.font.color if c.font else None)
    wb.save(OUT)
    print(f"wrote {OUT}: {len(runs)} runs")


def step_key(run_id):
    s = step_of(run_id.split("/")[0])
    return s if s is not None else 0


def style_sheet(ws, header_rows, freeze):
    for r in range(1, header_rows + 1):
        for c in ws[r]:
            if r == header_rows:
                c.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
                c.fill = HEAD_FILL
                c.alignment = Alignment(wrap_text=True, vertical="bottom")
            else:
                c.font = Font(name=FONT, size=10, bold=True, color="1F3A4D")
                c.fill = GROUP_FILL
                c.alignment = Alignment(horizontal="center")
            c.border = Border(bottom=THIN)
    ws.row_dimensions[header_rows].height = 42
    ws.freeze_panes = freeze


def comparison_list():
    out = []
    add = lambda s, what, a, b: out.append((s, what, a, b))
    for arm, what in [("zpred", "whole-board z from masked z"), ("sig_0.01", "SIGReg 0.05 -> 0.01"),
                      ("sig_0.002", "SIGReg 0.05 -> 0.002"), ("proj", "projector head under SIGReg"),
                      ("long", "long rollouts (2 -> 20 plies)"), ("all", "zpred + projector + SIGReg 0.01 + long"),
                      ("sig_full", "SIGReg on the full view only")]:
        add(3, what, f"step3/{arm}", "step3/base")
    add(4, "attention readout -> mean readout", "step4/B", "step4/A")
    add(4, "remove the encoder transformer", "step4/C", "step4/B")
    add(4, "add a transformer to the mask predictor (C)", "step4/D", "step4/C")
    add(4, "add a transformer to the mask predictor (B)", "step4/E", "step4/B")
    add(4, "unpool (C layout, batch 48)", "step4/F", "step4/C")
    add(4, "unpool + square-mean SIGReg", "step4/F2", "step4/C")
    for s in ("", "_seed1", "_seed2"):
        c = "step4/C" + s
        add(4, "a1 anchor edge" + s, "step4/C_anchor" + s, c)
        add(5, "drop the masked-square loss" + s, "step5/C_nomask" + s, c)
        add(5, "masked squares -> whole-board z from masked z" + s, "step5/C_zpred_nomask" + s, c)
        b = "step4/B" if not s else "step5/B" + s
        e = "step4/E" if not s else "step5/E" + s
        add(5, "B vs C (encoder transformer)" + s, b, c)
        add(5, "transformer -> per-token MLP (B)" + s, "step5/B_mlp" + s, b)
        add(5, "transformer -> MLP-Mixer (B)" + s, "step5/B_mixer" + s, b)
        add(5, "transformer -> per-token MLP (E)" + s, "step5/E_mlp" + s, e)
        add(5, "transformer -> MLP-Mixer (E)" + s, "step5/E_mixer" + s, e)
        a = "step4/C_anchor" + s
        add(6, "P7: moves as square names, no anchor" + s, "step6/P7" + s, c)
        for p, what in [("P3", "square names"), ("P4", "files/ranks"), ("P5", "masked z + square names"),
                        ("P6", "masked z + files/ranks"), ("P8", "move guess (inverse dynamics)")]:
            add(6, f"{p}: {what} (vs anchored C)" + s, f"step6/{p}" + s, a)
        add(6, "P9: a1 node flag (vs P7)" + s, "step6/P9" + s, "step6/P7" + s)
        add(7, "non-residual next predictor" + s, "step7/C_short_nores" + s, "step7/C_short" + s)
        add(7, "2-ply rollouts instead of 2 -> 20" + s, "step7/C_short" + s, c)
        add(9, "legal-move head on square tokens" + s, "step9/C_legal" + s, "step7/C_short" + s)
        add(10, "left/right edge marks" + s, "step10/C_edges" + s, "step7/C_short" + s)
        add(11, "legal head on top of edge marks" + s, "step11/C_edges_legal" + s, "step10/C_edges" + s)
        add(13, "sum branch in each GAT layer" + s, "step13/C_edges_legal_sum" + s, "step11/C_edges_legal" + s)
    add(7, "concat GAT (G1) vs C", "step7/G1", "step4/C")
    add(7, "concat GAT + residual (G1r) vs C", "step7/G1r", "step4/C")
    add(7, "mean/max MLP readout (G2) vs C", "step7/G2", "step4/C")
    add(7, "2-ply rollouts (G1r)", "step7/G1r_short", "step7/G1r")
    add(7, "non-residual next predictor (G1r)", "step7/G1r_short_nores", "step7/G1r_short")
    add(7, "learning rate 1e-4 (G1r)", "step7/G1r_short_lr", "step7/G1r_short")
    add(7, "SIGReg 0.1 (C)", "step7/C_sig", "step4/C")
    add(7, "SIGReg 0.1 (G1r)", "step7/G1r_sig", "step7/G1r")
    add(8, "everything in z (Jason's Z bundle) vs G1r", "step8/Z", "step7/G1r")
    add(8, "small predictors", "step8/Z_weak", "step8/Z")
    add(9, "piece-count decoder", "step9/C_count", "step7/C_short")
    add(9, "piece counts + legal head", "step9/C_count_legal", "step7/C_short")
    add(10, "file marks on nodes", "step10/C_files", "step7/C_short")
    add(10, "file marks on nodes (named moves)", "step10/P7_files", "step6/P7")
    add(11, "Z bundle + edge marks + legal head", "step11/Z_weak_edges_legal", "step8/Z_weak")
    add(12, "G1r + edge marks + legal head", "step12/G1r_edges_legal", "step7/G1r")
    add(13, "sum branch (G1r)", "step13/G1r_edges_legal_sum", "step12/G1r_edges_legal")
    add(14, "board decoder on own z", "step14/C_edges_decode", "step10/C_edges")
    add(14, "board decoder on own z (with legal head)", "step14/C_edges_legal_decode", "step11/C_edges_legal")
    add(14, "edge marks + decoder (G1r)", "step14/G1r_edges_decode", "step7/G1r")
    add(14, "decoder (G1r with legal head)", "step14/G1r_edges_legal_decode", "step12/G1r_edges_legal")
    for e in ("C", "G"):
        for arm, what in [("zpred", "whole-board z predictor"), ("dec_next", "decode predicted next z"),
                          ("leg_next", "legal moves on predicted next z"), ("leg_now", "legal moves on own z"),
                          ("dec_now", "decode own z")]:
            add(15, f"{what} ({e})", f"step15/{e}_{arm}", f"step15/{e}_base")
        for arm, what in [("dec_mask", "decode predicted whole z"), ("leg_mask", "legal moves on predicted whole z")]:
            add(15, f"{what} ({e}, vs zpred)", f"step15/{e}_{arm}", f"step15/{e}_zpred")
    add(16, "edge-mark start x4 (G1r, decode own z)", "step16/G_dec_now_marks4", "step15/G_dec_now")
    add(16, "edge-mark start x16 (G1r, decode own z)", "step16/G_dec_now_marks16", "step15/G_dec_now")
    add(16, "edge-mark start x16 (G1r, no head)", "step16/G_base_marks16", "step15/G_base")
    add(16, "16k steps, 2 -> 20 plies (no head)", "step16/C_base_long", "step15/C_base")
    add(16, "legal moves on own z, 16k steps", "step16/C_leg_now_long", "step16/C_base_long")
    add(16, "decode own z, 16k steps", "step16/C_dec_now_long", "step16/C_base_long")
    add(17, "online decoder fit (C, next z)", "step17/C_dec_next_online", "step15/C_dec_next")
    add(17, "online decoder fit (C, whole z)", "step17/C_dec_mask_online", "step15/C_dec_mask")
    add(17, "online decoder fit (G1r x16, next z)", "step17/G_dec_next_online_m16", "step17/G_dec_next_m16")
    add(17, "online decoder fit (G1r x16, whole z)", "step17/G_dec_mask_online_m16", "step17/G_dec_mask_m16")
    add(17, "edge-mark start x16 (G1r, decode next z)", "step17/G_dec_next_m16", "step15/G_dec_next")
    add(17, "edge-mark start x16 (G1r, decode whole z)", "step17/G_dec_mask_m16", "step15/G_dec_mask")
    add(17, "edge-mark start x16 (G1r, zpred)", "step17/G_zpred_m16", "step15/G_zpred")
    return out


def step_list():
    return [
        ("Before 3", "2026-10-24/26", "Build the JEPA encoder (one encoder + moving-average target, GAT square stage, "
         "transformer, attention readout) and check it trains.",
         "smoke runs, squares_sigreg 20k",
         "The untrained encoder already maps every position to nearly the same z; SIGReg fixed the spread. First "
         "probes: every trained encoder scored below the untrained one. LiDAR ranked the encoders opposite to the probes."),
        (3, "2026-10-27", "Make the objectives harder and the regularizer gentler: whole-board z predictor, SIGReg "
         "weight, projector, long rollouts.", "base, zpred, sig_*, proj, long, all",
         "Lower SIGReg helps linear probes up to 0.01. Long rollouts were the only arm that improved neighbours. No "
         "arm beats untrained. Odd plies were not graded, fixed for step 4."),
        (4, "2026-10-27/28", "Encoder layout: which parts destroy information?", "A, B, C, D, E, F, F2, C_anchor, C seeds",
         "The attention readout did most of the damage; the encoder transformer cost 0.13-0.17 more neighbour "
         "quality. C (3 GAT layers, mean readout, no transformer) became the baseline. Unpool (F) collapsed its "
         "square tokens; F2 fixed the collapse but its unpooled nodes carried no board information. Seed noise "
         "measured: single-seed differences under ~0.1 are not trustworthy."),
        (5, "2026-10-28/29", "Is the masked-square loss needed? Is it attention's mixing or depth that hurts B/E?",
         "C_nomask, C_zpred_nomask, B/E _mlp, _mixer, 3 seeds",
         "The masked-square loss can be swapped for masked z prediction at no cost. Per-token MLPs beat attention "
         "(+0.13-0.16 kNN) but stay below C. The MLP-Mixer collapses neighbours to ~0."),
        (6, "2026-10-29", "Predictor suite P1-P9: how should the predictors describe moves and see the board?",
         "P3-P9, 3 seeds", "No predictor beats P1 (C); predictor choice moves z by at most 0.04. Naming squares costs "
         "nothing. The a1 anchor edge and node flag give almost no left/right signal."),
        (7, "2026-10-29 to 11-10", "Is the mean readout or the GAT averaging the bottleneck? (Jason's concat GAT design)",
         "G1, G1r, G2, C_short, G1r_short, nores, lr, sig", "The concat GAT collapses z's neighbour structure (kNN ~0) "
         "while linear probes survive. Readout capacity is ruled out (G2, B_mlp). Rollout length does not matter "
         "(2 plies = 20). Whitened kNN shows raw kNN mixes loudness with content. On every probe, untrained >= trained: "
         "the objective is what drains z."),
        (8, "2026-11-11", "Jason's 'everything in z' bundle on G1r.", "Z, Z_weak",
         "The worst trained encoder so far; predictor size is not the lever. The move-guessing head learned nothing."),
        (9, "2026-11-12", "Heads graded against the board: piece counts and a legal-move head.",
         "C_count, C_legal, C_count_legal, C_legal seeds",
         "The legal-move head is the first objective positive on every probe at every seed (material +0.06, eval +0.02, "
         "all token labels up). The count decoder only learns what it trains."),
        (10, "2026-11-12/13", "Can the encoder tell left from right, and does mirror-blindness cause the decline?",
         "C_files, P7_files, C_edges + seeds",
         "Node file marks make z left/right aware but cost 0.2-0.3; edge marks cost ~0.03 at 3 seeds. The more z "
         "knows of left/right, the lower material and eval: the hypothesis is not supported."),
        (11, "2026-11-13", "Legal head on top of edge marks; G1r with the full Z bundle + legal head.",
         "C_edges_legal + seeds, Z_weak_edges_legal",
         "C + edge marks + legal head becomes the new 3-seed baseline. With the head on, the marks are used much less."),
        (12, "2026-11-12", "G1r + edge marks + legal head alone.", "G1r_edges_legal",
         "Best concat run at the time; the raw-kNN collapse mostly lifts. Still behind C. Fully mirror-blind."),
        (13, "2026-11-12", "Sum branch (GINE-like) beside each GAT layer.", "C_edges_legal_sum + seeds, G1r_..._sum",
         "Buys nothing measurable; lower training losses, not better probes."),
        (14, "2026-11-12", "A board decoder from z (direct, own z).", "C/G1r x decode, legal + decode",
         "The decoder hurts C's neighbour probes and cancels the legal head's gain; helps G1r a little. C becomes "
         "fully left/right aware with it; G1r stays blind."),
        (15, "2026-11-13", "Every head as a latent objective: decode or legal moves on predicted next z, predicted "
         "whole z, or own z.", "16 arms, C and G1r",
         "The z-level legal head helps in every form, most on own z. The decoder hurts C in every form. Predicted z "
         "reads as well as real target z: the limit is what z holds, not the predictors. G1r mirror-blind in all."),
        (16, "2026-11-13", "Is G1r's blindness a stuck point? Do the heads keep gaining with time and longer rollouts?",
         "G_dec_now_marks4/16, G_base_marks16, C_*_long",
         "With a x16 mark start plus the decoder, G1r becomes left/right aware: the capability was there. Long runs: "
         "best move (a side-dependent probe) is where trained z first clearly beats untrained, and the gains are the "
         "heads'; training longer without a head changes nothing."),
        (17, "2026-11-14", "Online decoder: the head learns from the online encoder's z (gradient on).",
         "C/G1r x next/whole, target-fit controls", "Every online arm looks like 'decode own z': left/right aware, "
         "decodes ~55%, best move up, kNN material/eval down. The prediction part adds nothing visible on C."),
    ]


def write_readme(ws, n_runs, seed0):
    lines = [
        ("Minotaur Labyrinth JEPA experiments", "title"),
        (f"{n_runs} training runs from runs/step3 to runs/step17, built by reports/build_workbook.py "
         "from each run's history.jsonl, run_config.json and the analysis outputs in runs/analysis. The narrative "
         "report is reports/labyrinth_jepa_report.html.", None),
        ("", None),
        ("Sheets", "h"),
        ("Runs: one row per run. Settings, final z probes, the same encoder untrained, trained minus untrained "
         "(formulas), evaluation metrics, head scores, left/right checks and token probes.", None),
        ("Comparisons: paired arm-minus-baseline differences. Formulas look both runs up in Runs.", None),
        ("Trajectories: every probe evaluation of every run (step 0 = the untrained encoder).", None),
        ("Orientation: every left/right check run so far (mirror gap; king side read from z).", None),
        ("Tokens: every square-token probe run so far (linear AUC / R2).", None),
        ("Steps: what each round asked and what it found.", None),
        ("", None),
        ("Probes", "h"),
        ("Material = material balance R2 (val, 8,000 positions). Eval = engine win probability R2 (8,000 analysed "
         "positions). Themes = puzzle theme AUC. Best move = engine best move AUC, labels in the side to move's "
         "frame but distinguishing files, so it is the one headline probe where left/right matters.", None),
        ("lin = ridge / logistic probe. kNN = k nearest neighbours by cosine (mixes how loud a feature is with "
         "whether it is there). white = kNN after whitening the top 64 directions (added in step 7; blank before).",
         None),
        ("King side = can a probe read which half of the board the side to move's king is on? Every position "
         "is paired with its mirror image, so a mirror-blind z scores exactly 0.50. Mirror gap = distance from a "
         "position's z to its mirror's, over the typical distance between positions.", None),
        ("", None),
        ("Caveats", "h"),
        ("One seed per arm unless _seed1/_seed2 rows exist. Paired noise (same seed, one change) is about 0.02; "
         "unpaired single-seed differences under about 0.1 are not trustworthy (step 4 seed study).", None),
        ("The trainer draws its probe positions from the run's seed, so seed-1 and seed-2 rows use a different "
         "probe set from seed 0. Compare within a seed (as the Comparisons sheet does).", None),
        ("The trainer's validation subset (val losses, RankMe, LiDAR, head scores in 'Evaluation') is the first "
         "2,048 positions of an alphabetically sorted split: endgame-heavy. Like for like between runs only.", None),
        ("Steps trained differs: most runs 4,000 steps, step 4 F 21,333 at batch 48, step 16 long runs 16,000.", None),
        ("", None),
        ("Reference: untrained C encoder and board one-hot (seed 0 probe set)", "h"),
    ]
    r = 1
    for text, kind in lines:
        c = ws.cell(row=r, column=1, value=text)
        if kind == "title":
            c.font = Font(name=FONT, size=16, bold=True, color="1F3A4D")
        elif kind == "h":
            c.font = Font(name=FONT, size=11, bold=True, color="1F3A4D")
        else:
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
    header = ["", "Material lin", "Material kNN", "Material white", "Eval lin", "Eval kNN", "Eval white",
              "Themes lin", "Best move lin"]
    for i, h in enumerate(header, start=1):
        c = ws.cell(row=r, column=i, value=h)
        c.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
        c.fill = HEAD_FILL
    for name, block in [("Untrained C (step15/C_base, step 0)", seed0["untrained"]["z"]),
                        ("Board one-hot", seed0["board_onehot"].get("onehot") or next(iter(seed0["board_onehot"].values())))]:
        r += 1
        ws.cell(row=r, column=1, value=name).font = Font(name=FONT, size=10)
        vals = [val(block, "val/material_balance", k) for k in ("linear", "knn", "knn_white")] + \
               [val(block, "labeled/eval_win_prob", k) for k in ("linear", "knn", "knn_white")] + \
               [val(block, "puzzles/theme/auc", "linear"), val(block, "labeled/best_move/auc", "linear")]
        for i, v in enumerate(vals, start=2):
            c = ws.cell(row=r, column=i, value=v)
            c.number_format = "0.00"
            c.font = Font(name=FONT, size=10)
    ws.column_dimensions["A"].width = 110
    for c in "BCDEFGHI":
        ws.column_dimensions[c].width = 12


if __name__ == "__main__":
    main()
