"""Validasi ChainedAlfredThorEnv di seluruh ruangan valid_unseen.

Tanpa --manifest, semua goal unik tiap ruangan dirangkai menjadi satu rangkaian tanpa reset, dan sebelum
setiap tugas genap objek target tugas itu dipindahkan diam-diam (relocate). Dengan --manifest, urutan tugas
dan perpindahan objek diambil dari manifest (scripts/build_chain_manifest.py) dan diputar ulang.

Solver:
  gt     : pemecah dengan lokasi objek sebenarnya, perintah dikirim lewat antarmuka agen (uji kelayakan)
  expert : expert ALFWorld. Dengan --via-agent-api, perintahnya dikirim tanpa nomor objek dan
           observasinya tanpa daftar objek, sehingga resolusi perintah dan sanitasi observasi ikut teruji.

Contoh:
  python scripts/validate_chains.py --out /tmp/chains.json
  python scripts/validate_chains.py --manifest data/chains_valid_unseen.json
  python scripts/validate_chains.py --solver expert --via-agent-api --rooms 308 --max-tasks 3
"""
import argparse
import collections
import glob
import json
import os
import re
import time
import traceback

from alfworld.agents.environment.alfred_chained_thor_env import ChainedAlfredThorEnv
from alfworld.agents.environment.chain_manifest import iter_chain, load_manifest, room_trials, split_root

ROOT = split_root("valid_unseen")


def goal_key(t):
    p = t["pddl_params"]
    return (t["task_type"], p["object_target"], p["parent_target"], p.get("toggle_target") or "")


def unique_goal_trials(split_root):
    """{scene_num: [traj_data.json untuk satu trial per goal unik]} seperti yang dimuat ALFWorld."""
    by_room = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{split_root}/*/trial_*/traj_data.json")):
        r = os.path.dirname(f)
        if "movable" in r or "Sliced" in r or not os.path.exists(os.path.join(r, "game.tw-pddl")):
            continue
        with open(f) as fh:
            t = json.load(fh)
        by_room[t["scene"]["scene_num"]].setdefault(goal_key(t), f)
    return {s: list(g.values()) for s, g in by_room.items()}


def default_entries(task_files, relocate=True):
    """Entri rangkaian awal: relocate objek target sebelum setiap tugas genap (selain tugas pertama)."""
    out = []
    for k, f in enumerate(task_files):
        with open(f) as fh:
            target = json.load(fh)["pddl_params"]["object_target"]
        out.append({"task_file": f,
                    "relocate": {"object_type": target} if relocate and k > 0 and k % 2 == 0 else None})
    return out


def strip_object_numbers(cmd, receptacles):
    """'take mug 1 from countertop 1' -> 'take mug from countertop 1' (nomor wadah tetap)."""
    rec = set(receptacles)
    words, out, i = cmd.split(), [], 0
    while i < len(words):
        if i + 1 < len(words) and words[i + 1].isdigit():
            pair = words[i] + " " + words[i + 1]
            out.append(pair if pair in rec else words[i])
            i += 2
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def leaks_in_trace(env):
    """Observasi agen pada tugas saat ini tidak boleh memuat daftar objek atau nomor objek."""
    kinds = "|".join(sorted({r.split()[0] for r in env.receptacle_names()}))
    pat = re.compile(r"\b(?!(?:%s)\b)[a-z]+ \d+\b" % kinds)
    return [("leak", t["action"], t["obs"]) for t in env.trace
            if t["task_idx"] == env.task_idx and ("you see" in t["obs"].lower() or pat.search(t["obs"].lower()))]


def solve_with_expert(env, via_agent_api):
    """Selesaikan tugas saat ini dengan expert. Mengembalikan (won, langkah, alasan, catatan_api)."""
    api_issues = []
    if not via_agent_api:
        won, steps, reason = env.run_expert()
        return won, steps, reason, api_issues
    saved, env.max_steps = env.max_steps, env.expert_max_steps
    reason = "ok"
    try:
        while not env.done:
            a = env.expert_next_action()
            agent_cmd = strip_object_numbers(a, env.receptacle_names())
            was_admissible = a in env.controller.get_admissible_commands()
            env.step(agent_cmd)
            # perintah expert yang sah harus tetap sah lewat antarmuka agen
            if was_admissible and env.trace[-1]["command"] is None and a not in ("help",):
                api_issues.append(("rejected", a, agent_cmd))
        won = bool(env.env.get_goal_satisfied())
        if not won:
            reason = "step_limit"
        api_issues += leaks_in_trace(env)
    except Exception as e:
        won = False
        name = type(e).__name__
        reason = {"HandCodedAgentTimeout": "expert_timeout",
                  "HandCodedAgentFailed": "expert_failed"}.get(name, "error: " + traceback.format_exc(limit=3))
    finally:
        env.max_steps = saved
    return won, env.steps, reason, api_issues


def run_room(env, scene, entries, solver="gt", via_agent_api=False, trials=None, verbose=True):
    """Jalankan satu rangkaian dan selesaikan tiap tugas dengan solver. Mengembalikan list catatan per tugas."""
    results = []
    chain = iter_chain(env, entries, trials=trials)
    for k in range(len(entries)):
        t0 = time.time()
        rec = {"scene": scene, "k": k, "task_file": entries[k]["task_file"]}
        n_log = len(env.task_log)
        try:
            ctx = next(chain)
            info = ctx["info"]
            if k == 0 and verbose:
                print("  wadah tambahan: %d dari receps.json trial lain, ditemukan %s"
                      % (env.merged_receptacles, env.discovered_receptacles))
            if entries[k].get("relocate"):
                rec["relocation"] = ctx["relocation"]
                rec["relocation_replayed"] = ctx["relocation_replayed"]
            rec["task_desc_matches"] = ctx["task_desc_matches"]
            rec.update(task_desc=info["task_desc"], won_initial=info["won_initial"],
                       gc_initial=info["gc_initial"],
                       target_locations=env.task_info["target_locations_initial"])
            if info["won_initial"]:
                rec.update(won=None, steps=0, reason="already_satisfied", api_issues=[])
            elif solver == "gt":
                won, steps, reason = env.run_gt_solver()
                rec.update(won=won, steps=steps, reason=reason, api_issues=leaks_in_trace(env))
            else:
                won, steps, reason, issues = solve_with_expert(env, via_agent_api)
                rec.update(won=won, steps=steps, reason=reason, api_issues=issues)
            rec["target_full"] = env.task_info["target_full"]
            rec["target_full_events"] = env.task_info["target_full_events"]
        except Exception:
            rec.update(won=False, steps=None, reason="crash: " + traceback.format_exc(limit=4), api_issues=[])
        rec["seconds"] = round(time.time() - t0, 1)
        rec["release"] = next((e["release"] for e in env.task_log[n_log:] if "release" in e), None)
        fb = [f["kind"] for f in env.fallback_log if f["task_idx"] == env.task_idx] if k == env.task_idx else []
        rec["fallbacks"] = len(fb)
        rec["fallback_kinds"] = dict(collections.Counter(fb))
        if verbose:
            print_record(rec)
        results.append(rec)
        if rec["reason"].startswith("crash"):
            break   # keadaan ruangan tidak pasti, hentikan rangkaian ini
    return results


def print_record(rec):
    reloc = rec.get("relocation")
    rtxt = "" if "relocation" not in rec else (
        " | reloc %s->%s%s" % (reloc["before"], reloc["after"],
                               "" if rec.get("relocation_replayed") is not False else " (dicari ulang)")
        if reloc else " | reloc gagal")
    if rec["release"]:
        rtxt += " | lepas %s->%s" % (rec["release"]["object"], rec["release"]["to"])
    if rec["fallbacks"]:
        rtxt += " | fallback %s" % rec["fallback_kinds"]
    if rec.get("target_full"):
        rtxt += " | wadah tujuan penuh" + (" (wadah sejenis lain dipakai)" if rec["won"] else "")
    if rec.get("task_desc_matches") is False:
        rtxt += " | DESKRIPSI BEDA DARI MANIFEST"
    print(f"  [{rec['k']:2d}] {rec.get('task_desc', '?')[:45]:45s} won={rec['won']} steps={rec['steps']} "
          f"gc0={rec.get('gc_initial', 0):.2f} {rec['reason'][:60]} api_issues={len(rec['api_issues'])}"
          f" ({rec['seconds']}s){rtxt}", flush=True)


def summarize(results, solver):
    done = [r for r in results if r["won"] is not None]
    print("\n===== ringkasan")
    print("tugas:", len(results), "| sudah terpenuhi di awal:", sum(r["won"] is None for r in results),
          "| berhasil diselesaikan (%s):" % solver, sum(bool(r["won"]) for r in done), "/", len(done))
    print("gc_initial > 0:", sum(r.get("gc_initial", 0) > 0 for r in results))
    relocs = [r for r in results if "relocation" in r]
    print("relocate berhasil:", sum(bool(r["relocation"]) for r in relocs), "/", len(relocs),
          "| diputar ulang persis dari manifest:", sum(r.get("relocation_replayed") is True for r in relocs))
    rel = [r["release"] for r in results if r.get("release")]
    print("objek dilepas antar tugas:", len(rel), "| gagal dilepas:", sum(not x["ok"] for x in rel))
    print("tugas dengan fallback alat penuh (efek langsung):",
          sum(r.get("fallback_kinds", {}).get("appliance", 0) > 0 for r in results),
          "| penempatan dari sudut pandang lain:",
          sum(r.get("fallback_kinds", {}).get("put_view", 0) > 0 for r in results))
    print("wadah tujuan penuh (gagal):", sum(bool(r.get("target_full")) and not r["won"] for r in results),
          "| deskripsi tugas beda dari manifest:", sum(r.get("task_desc_matches") is False for r in results))
    print("masalah antarmuka agen:", sum(len(r["api_issues"]) for r in results))
    print("per alasan:", dict(collections.Counter(r["reason"].split(":")[0] for r in results)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="putar ulang rangkaian dari manifest (build_chain_manifest.py)")
    ap.add_argument("--rooms", type=int, nargs="*", help="FloorPlan yang diuji (default: semua)")
    ap.add_argument("--max-tasks", type=int, default=0, help="batas tugas per ruangan (0 = semua)")
    ap.add_argument("--solver", choices=["gt", "expert"], default="gt",
                    help="gt: pemecah dengan lokasi sebenarnya lewat antarmuka agen (uji kelayakan); "
                         "expert: expert ALFWorld (eksplorasi heuristik, bisa melebihi batas langkah)")
    ap.add_argument("--via-agent-api", action="store_true", help="khusus --solver expert")
    ap.add_argument("--no-relocate", action="store_true", help="tanpa --manifest saja")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expert-steps", type=int, default=400, help="batas langkah expert per tugas")
    ap.add_argument("--out", default="/tmp/validate_chains.json")
    args = ap.parse_args()

    if args.manifest:
        manifest = load_manifest(args.manifest)
        seed = manifest["seed"]
        rooms = {int(s): r["tasks"] for s, r in manifest["rooms"].items()}
    else:
        seed = args.seed
        rooms = {s: default_entries(fs, relocate=not args.no_relocate) for s, fs in unique_goal_trials(ROOT).items()}
    trials = room_trials("valid_unseen")

    env = ChainedAlfredThorEnv(seed=seed, expert_max_steps=args.expert_steps)
    results = []
    try:
        for scene in sorted(rooms, key=lambda s: -len(rooms[s])):
            if args.rooms and scene not in args.rooms:
                continue
            entries = rooms[scene][: args.max_tasks or None]
            print(f"\n##### FloorPlan{scene}: {len(entries)} tugas")
            results += run_room(env, scene, entries, args.solver, args.via_agent_api, trials)
    finally:
        env.close()
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1, default=str)
    summarize(results, args.solver)
    print("hasil lengkap:", args.out)


if __name__ == "__main__":
    main()
