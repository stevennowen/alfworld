"""Bangun manifest rangkaian tugas yang tervalidasi untuk ChainedAlfredThorEnv.

Untuk setiap ruangan, semua goal unik dirangkai (relocate sebelum setiap tugas genap) lalu diselesaikan
dengan pemecah GT lewat antarmuka agen. Tugas yang tidak bisa diselesaikan dibuang:
  already_satisfied : tujuan sudah terpenuhi saat tugas diberikan
  capacity          : wadah tujuan penuh oleh objek dari tugas sebelumnya (lihat env.task_info["target_full"])
  lainnya           : dicatat dengan peringatan karena menandakan masalah lingkungan
Membuang satu tugas mengubah urutan relocate dan keadaan ruangan sesudahnya, jadi rangkaian dijalankan
ulang sampai semua tugas yang tersisa berhasil. Manifest menyimpan urutan akhir beserta perpindahan objek
yang terjadi pada putaran terakhir, sehingga setiap agen menerima rangkaian yang sama persis.

Contoh:
  python scripts/build_chain_manifest.py --out configs/chains/valid_unseen_seed0.json
  python scripts/build_chain_manifest.py --rooms 424 --out /tmp/m424.json
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_chains import ROOT, default_entries, run_room, unique_goal_trials  # noqa: E402

from alfworld.agents.environment.alfred_chained_thor_env import ChainedAlfredThorEnv  # noqa: E402
from alfworld.agents.environment.chain_manifest import rel_path, room_trials  # noqa: E402


def drop_reason(rec):
    if rec["reason"] == "already_satisfied":
        return "already_satisfied"
    if rec["reason"] == "place_failed" and rec.get("target_full"):
        return "capacity"
    return rec["reason"].split(":")[0]


def manifest_entry(rec):
    reloc = rec.get("relocation")
    with open(rec["task_file"]) as fh:
        traj = json.load(fh)
    p = traj["pddl_params"]
    return {
        "task_file": rel_path(rec["task_file"]),
        "task_desc": rec["task_desc"],
        "task_type": traj["task_type"],
        "object_target": p["object_target"],
        "parent_target": p["parent_target"],
        "toggle_target": p.get("toggle_target") or "",
        "relocate": None if not reloc else {
            "object_type": reloc["object_type"], "object_name": reloc["object_name"],
            "source": rel_path(reloc["source"]), "before": reloc["before"], "after": reloc["after"]},
        "relocation_failed": "relocation" in rec and not reloc,
        "gt_steps": rec["steps"],
        "fallbacks": rec["fallback_kinds"],
    }


def build_room(env, scene, task_files, trials, max_rounds):
    """Kegagalan already_satisfied dan capacity langsung dibuang. Kegagalan lain bisa berasal dari fisika
    THOR yang tidak sepenuhnya deterministik, jadi tugasnya dicoba sekali lagi pada putaran berikutnya
    dan baru dibuang bila gagal lagi."""
    files, dropped, strikes = list(task_files), [], {}
    for rnd in range(1, max_rounds + 1):
        print(f"\n##### FloorPlan{scene} putaran {rnd}: {len(files)} tugas", flush=True)
        recs = run_room(env, scene, default_entries(files), "gt", trials=trials)
        bad = [r for r in recs if r["won"] is not True]
        complete = len(recs) == len(files)
        if not bad and complete:
            return [manifest_entry(r) for r in recs], dropped, rnd
        bad_files = set()
        for r in bad:
            reason = drop_reason(r)
            if reason not in ("already_satisfied", "capacity"):
                strikes[r["task_file"]] = strikes.get(r["task_file"], 0) + 1
                if strikes[r["task_file"]] < 2:
                    print("  [%d] %s gagal (%s), dicoba lagi di putaran berikutnya"
                          % (r["k"], r.get("task_desc"), r["reason"][:80]))
                    continue
                print("  PERINGATAN: [%d] %s dibuang karena %s" % (r["k"], r.get("task_desc"), r["reason"][:200]))
            bad_files.add(r["task_file"])
            dropped.append({"task_file": rel_path(r["task_file"]), "task_desc": r.get("task_desc"),
                            "reason": reason, "round": rnd, "position": r["k"],
                            "target_full_events": r.get("target_full_events", [])})
        files = [f for f in files if f not in bad_files]
    raise RuntimeError("FloorPlan%d belum stabil setelah %d putaran" % (scene, max_rounds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, nargs="*", help="FloorPlan yang dibangun (default: semua)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--out", default="configs/chains/valid_unseen_seed0.json")
    ap.add_argument("--update", action="store_true",
                    help="muat manifest --out yang ada dan ganti hanya ruangan pada --rooms")
    args = ap.parse_args()

    rooms = unique_goal_trials(ROOT)
    trials = room_trials("valid_unseen")
    manifest = {"version": 1, "split": "valid_unseen", "seed": args.seed,
                "created": datetime.date.today().isoformat(),
                "builder": "scripts/build_chain_manifest.py (solver gt, relocate sebelum setiap tugas genap)",
                "rooms": {}}
    if args.update and os.path.exists(args.out):
        with open(args.out) as fh:
            old = json.load(fh)
        if old["seed"] != args.seed:
            raise SystemExit("seed manifest lama (%d) berbeda dari --seed" % old["seed"])
        manifest["rooms"] = old["rooms"]
    env = ChainedAlfredThorEnv(seed=args.seed)
    try:
        for scene in sorted(rooms, key=lambda s: -len(rooms[s])):
            if args.rooms and scene not in args.rooms:
                continue
            tasks, dropped, rounds = build_room(env, scene, rooms[scene], trials, args.max_rounds)
            manifest["rooms"][str(scene)] = {"tasks": tasks, "dropped": dropped, "rounds": rounds,
                                             "n_unique_goals": len(rooms[scene])}
    finally:
        env.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=1)

    print("\n===== manifest")
    for scene, r in manifest["rooms"].items():
        print("FloorPlan%s: %d/%d tugas dipakai, %d putaran, dibuang: %s" % (
            scene, len(r["tasks"]), r["n_unique_goals"], r["rounds"],
            [(d["task_desc"], d["reason"]) for d in r["dropped"]]))
    total = sum(len(r["tasks"]) for r in manifest["rooms"].values())
    print("total tugas:", total, "| tersimpan di", args.out)


if __name__ == "__main__":
    main()
