"""Explanations built entirely by code from python-chess facts (no LLM, so no false claims).

Style follows "feigned discovery" (Master Distillation, arXiv 2603.20510): notice what stands out,
pick the forcing move, calculate the line, conclude. Optionally (search=True) first considers and
refutes one tempting alternative using Stockfish's multi-PV analysis (a code-built search trace).

code_trace(puzzle, analysis=None, search=False) -> text ending with FINAL_LINE / FINAL_MOVE.
"""
import random

import chess

from line_facts import SIDE, THEME_WORDS, facts_for_line, start_facts

MOTIF_THEMES = ["fork", "pin", "skewer", "discoveredAttack", "doubleCheck", "deflection", "attraction",
                "clearance", "interference", "hangingPiece", "trappedPiece", "capturingDefender",
                "xRayAttack", "intermezzo", "quietMove", "zugzwang", "promotion", "backRankMate",
                "smotheredMate", "exposedKing"]


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _clause(f):
    """What a move does, in words (no side/move prefix)."""
    bits = []
    if f.get("captures"):
        c = f["captures"]
        bits.append(f"takes the {c['piece']} on {c['square']}")
    if f.get("promotes_to"):
        bits.append(f"promotes to a {f['promotes_to']}")
    if f.get("checkmate"):
        bits.append("gives checkmate")
    elif f.get("double_check"):
        bits.append("gives double check")
    elif f.get("discovered_check"):
        bits.append("uncovers a discovered check")
    elif f.get("check"):
        bits.append("gives check")
    if f.get("hero_move") and f.get("discovered_attacks") and not f.get("discovered_check"):
        d = f["discovered_attacks"][0]
        bits.append(f"and uncovers an attack by the {d['by'].split(chr(39) + 's ', 1)[1]} "
                    f"on the {d['target'].split(chr(39) + 's ', 1)[1]}")
    if f.get("hero_move") and not f.get("fork") and not f.get("check"):
        my = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 0}[f["piece"]]
        big = [t for t in f["attacks"] if not t["king"] and t["value"] >= 3 and (t["value"] > my or not t["defended"])]
        if big:
            bits.append("attacks the " + " and the ".join(t["desc"].split(chr(39) + "s ", 1)[1] for t in big))
    if f.get("fork") and f.get("hero_move"):
        tg = [t["desc"].split("'s ", 1)[1] for t in f["attacks"] if t["king"] or t["value"] >= 3]
        if len(tg) >= 2:
            bits.append("and attacks the " + " and the ".join(tg) + " at the same time")
    if not bits:
        return "is a quiet move"
    return bits[0] + "".join((" and " if not b.startswith("and ") else " ") + b for b in bits[1:])


def code_trace(p, analysis=None, search=False, seed=None):
    rng = random.Random(seed if seed is not None else p["puzzle_id"])
    board = chess.Board(p["fen"])
    hero, opp = SIDE[board.turn], SIDE[not board.turn]
    s = start_facts(p["fen"])
    facts, _ = facts_for_line(p["fen"], p["full_solution"])
    first = facts[0]
    out = []

    # 1. what stands out
    obs = []
    if s["in_check"]:
        obs.append(f"{hero} is in check, so that has to be dealt with first.")
    if s["hanging"]:
        obs.append(f"The {', the '.join(h.split(chr(39) + 's ', 1)[1] for h in s['hanging'][:2])} "
                   f"{'is' if len(s['hanging']) == 1 else 'are'} attacked and undefended.")
    if s["pins"]:
        obs.append(s["pins"][0].replace("White's", "the white").replace("Black's", "the black").capitalize() + ".")
    if s["checks"]:
        obs.append(f"{hero} has {_plural(len(s['checks']), 'check')} available ({', '.join(s['checks'][:4])}).")
    motifs = [THEME_WORDS[t] for t in p.get("themes", []) if t in MOTIF_THEMES]
    if motifs and rng.random() < 0.7:
        m0 = motifs[0]
        art = "" if m0.startswith(("a ", "an ")) or m0 in ("zugzwang", "promotion", "clearance", "interference",
                                                            "attraction (luring a piece)", "deflection") else "a "
        obs.append(f"The piece placement suggests looking for {art}{m0}.")
    if not obs:
        obs.append(f"{hero} to move; the forcing moves are the place to start.")
    out.append(" ".join(obs[:3]))

    # 2. optionally: a tempting alternative, refuted (code-built search)
    if search and analysis:
        alts = [c for c in analysis["candidates"] if c["move"] != p["correct_move"] and c["win_pct"] < 70]
        if alts:
            a = alts[0]
            reply = a["line_san"][1] if len(a["line_san"]) > 1 else None
            verdict = "is about equal" if a["win_pct"] > 30 else f"{hero} is worse"
            if reply:
                out.append(f"{a['san']} looks tempting, but after {reply} "
                           f"{'the position ' + verdict if a['win_pct'] > 30 else verdict}, so it doesn't work.")

    # 3. the key move and the line
    threat = any(not a["king"] and a["value"] >= 3 for a in first.get("attacks", []))
    lead = ("The most forcing try is" if first.get("check") or first.get("captures") else
            "The key idea is" if threat else "The key idea is the quiet")
    sac = first.get("sacrifice")
    out.append(f"{lead} {first['san']}: the {first['piece']} {_clause(first)}." +
               (" It gives up material, but it works." if sac else ""))
    seq = []
    for i in range(1, len(facts), 2):
        reply = facts[i]
        only = " (the only legal reply)" if facts[i - 1].get("n_replies") == 1 else ""
        txt = f"after {reply['san']}{only}"
        if reply.get("captures"):
            txt += f", which takes the {reply['captures']['piece']} on {reply['captures']['square']}"
        elif reply.get("exposed"):
            e = reply["exposed"][0]
            txt += (f", which exposes the {e['target'].split(chr(39) + 's ', 1)[1]} to the "
                    f"{e['by'].split(chr(39) + 's ', 1)[1]}")
        if i + 1 < len(facts):
            nxt = facts[i + 1]
            txt += f", {nxt['san']} {_clause(nxt)}"
        seq.append(txt)
    if seq:
        s0 = "; then ".join(seq)
        out.append(s0[0].upper() + s0[1:] + ".")
    last = facts[-1]
    if last.get("checkmate"):
        out.append("That is checkmate.")
    else:
        bal = last.get("balance_after", 0)
        if bal > 0:
            out.append(f"{hero} comes out {_plural(bal, 'point')} of material ahead.")
        else:
            out.append(f"The material is {'level' if bal == 0 else 'behind'} for now, but {hero}'s position is decisive.")
    out.append(f"So the move is {first['san']}.")
    text = " ".join(x.strip() for x in out).replace(".  ", ". ")
    return text + f"\nFINAL_LINE: {' '.join(p['full_solution'])}\nFINAL_MOVE: {p['correct_move']}"


if __name__ == "__main__":
    import json
    import sys
    pz = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "pool_night1_engine98.jsonl")]
    an = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("results/night1_engine_analysis.jsonl")}
    for p in pz[::20][:5]:
        print(p["puzzle_id"], p["rating"], p["themes"])
        print(code_trace(p))
        print("-- with search:")
        print(code_trace(p, an.get(p["puzzle_id"]), search=True))
        print()
