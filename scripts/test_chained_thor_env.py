"""Uji ChainedAlfredThorEnv pada FP10 (valid_unseen):
tugas 1 dikerjakan agen lewat perintah tanpa nomor, tugas 2-3 dikerjakan expert.
Sebelum tugas 3, relocate mug dari coffeemachine harus ditolak dan relocate apel harus berhasil."""
import glob
import os

from alfworld.agents.environment.alfred_chained_thor_env import ChainedAlfredThorEnv

ROOT = os.path.expandvars("$ALFWORLD_DATA/json_2.1.1/valid_unseen")


def trial(folder, i=0):
    return sorted(glob.glob(f"{ROOT}/{folder}/trial_*/traj_data.json"))[i]


def show_task(obs, info, env):
    print("\n" + "=" * 70)
    print(f"[tugas {info['task_idx']}] {info['task_desc']}")
    print("obs awal:", obs[:160].replace("\n", " "), "...")
    print(f"won_initial={info['won_initial']} gc_initial={info['gc_initial']:.2f}")
    print("lokasi target (ground truth):", env.task_info["target_locations_initial"])


env = ChainedAlfredThorEnv(max_steps_per_task=50, seed=0)
try:
    # ---- tugas 1: agen manual dengan perintah tanpa nomor objek
    t1 = trial("pick_heat_then_place_in_recep-Apple-None-Fridge-10")
    obs, info = env.start_chain(t1)
    show_task(obs, info, env)
    apple_at = [r for r in env.task_info["target_locations_initial"]["Apple"] if r != "held"]
    print("\n-- perintah agen (tanpa nomor objek)")
    script = [f"go to {apple_at[0]}", "open " + apple_at[0], f"take apple from {apple_at[0]}",
              "take apple from countertop 1",            # sudah memegang -> harus gagal
              "heat apple with microwave 1",              # belum di microwave -> harus gagal
              "go to microwave 1", "heat apple with microwave 1", "examine apple",
              "go to fridge 1", "open fridge 1", "put apple in fridge 1"]
    for a in script:
        obs, done, info = env.step(a)
        print(f"  > {a:35s} | {obs}")
        if done:
            break
    print(f"tugas 1 selesai: won={info['won']} gc={info['goal_condition_success_rate']:.2f} langkah={info['steps']}")
    print("objek terlihat di frame terakhir:", env.trace[-1]["visible_types"])

    # ---- tugas 2: expert
    t2 = trial("pick_clean_then_place_in_recep-Mug-None-CoffeeMachine-10")
    obs, info = env.next_task(t2)
    show_task(obs, info, env)
    print("expert:", env.run_expert())
    print("mug sekarang di:", env.gt_object_locations("Mug"))

    # ---- perpindahan sebelum tugas 3
    sources = sorted(glob.glob(f"{ROOT}/*-10/trial_*/traj_data.json"))   # semua trial di FP10
    # satu-satunya mug ada di coffeemachine (wadah bertitik tempel tunggal), jadi harus ditolak
    reloc = env.relocate("Mug", sources)
    print("\n-- relocate mug dari coffeemachine (harus None):", reloc)
    assert reloc is None, "objek di coffeemachine tidak boleh dipindah lewat SetObjectPoses"
    print("apel sebelum dipindah:", env.gt_object_locations("Apple"))
    reloc = env.relocate("Apple", sources)
    print("-- relocate apel:", reloc)
    assert reloc and reloc["ok"]
    print("apel setelah dipindah:", env.gt_object_locations("Apple"))

    # ---- tugas 3: expert, mug harus dicari di lokasi baru
    t3 = trial("pick_cool_then_place_in_recep-Mug-None-Cabinet-10")
    obs, info = env.next_task(t3)
    show_task(obs, info, env)
    print("expert:", env.run_expert())

    # ---- ruang perintah untuk constrained decoding
    g = env.gbnf()
    print("\n-- gbnf:", len(g), "karakter;", g.splitlines()[1][:120], "...")
    print("\n-- ringkasan per tugas")
    for t in env.task_log:
        print(" ", {k: v for k, v in t.items() if k != "task_file"})
finally:
    env.close()
