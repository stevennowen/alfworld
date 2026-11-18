"""Manifest rangkaian tugas untuk ChainedAlfredThorEnv.

Manifest memuat, per ruangan, urutan tugas yang sudah dibuktikan bisa diselesaikan di tengah rangkaian
(scripts/build_chain_manifest.py) beserta perpindahan objek sebelum tiap tugas. Semua agen yang
dibandingkan menjalankan manifest yang sama.

Contoh pemakaian:
    manifest = load_manifest("chains_valid_unseen.json")
    env = ChainedAlfredThorEnv(seed=manifest["seed"])
    for ctx in iter_chain(env, manifest, room=10):
        obs, info = ctx["obs"], ctx["info"]
        while not env.done:
            obs, done, info = env.step(agent.act(obs, env.get_frame()))
"""
import collections
import glob
import json
import os

DATA = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))


def split_root(split="valid_unseen"):
    return os.path.join(DATA, "json_2.1.1", split)


def rel_path(task_file):
    """Path traj_data.json relatif terhadap $ALFWORLD_DATA, supaya manifest bisa dipakai di mesin lain."""
    return os.path.relpath(task_file, DATA)


def abs_path(rel):
    return os.path.join(DATA, rel)


def room_trials(split="valid_unseen"):
    """{scene_num: [semua traj_data.json di ruangan itu]}, sumber posisi untuk relocate."""
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(split_root(split), "*", "trial_*", "traj_data.json"))):
        with open(f) as fh:
            out[json.load(fh)["scene"]["scene_num"]].append(f)
    return out


def load_manifest(path):
    with open(path) as fh:
        return json.load(fh)


def iter_chain(env, manifest_or_tasks, room=None, trials=None):
    """Jalankan satu rangkaian. Menghasilkan dict per tugas setelah tugas itu diberikan ke agen:
        k, entry, obs, info, relocation (hasil env.relocate atau None), relocation_replayed.
    Pemanggil menyelesaikan tugasnya (env.step) sebelum meminta tugas berikutnya.

    manifest_or_tasks: manifest lengkap (dengan room) atau langsung list entri tugas.
    Entri tugas: {"task_file": relatif/absolut, "relocate": None | {"object_type", "object_name"?, "source"?}}.
    Bila relocate memuat object_name dan source, perpindahan yang tercatat diputar ulang persis. Bila
    tidak berhasil (keadaan ruangan berbeda karena tindakan agen), relocate dicari ulang dari semua trial
    ruangan itu dan relocation_replayed bernilai False."""
    if isinstance(manifest_or_tasks, dict):
        tasks = manifest_or_tasks["rooms"][str(room)]["tasks"]
        split = manifest_or_tasks.get("split", "valid_unseen")
    else:
        tasks, split = manifest_or_tasks, "valid_unseen"
    if trials is None:
        trials = room_trials(split)
    for k, entry in enumerate(tasks):
        task_file = entry["task_file"] if os.path.isabs(entry["task_file"]) else abs_path(entry["task_file"])
        reloc, replayed = None, None
        if k == 0:
            obs, info = env.start_chain(task_file)
        else:
            spec = entry.get("relocate")
            if spec:
                # jangan pindahkan objek ke wadah tujuan tugas berikutnya (tugas jadi setengah selesai)
                with open(task_file) as fh:
                    parent = json.load(fh)["pddl_params"].get("parent_target")
                avoid = (parent,) if parent else ()
                if spec.get("object_name") and spec.get("source"):
                    src = spec["source"] if os.path.isabs(spec["source"]) else abs_path(spec["source"])
                    reloc = env.relocate(spec["object_type"], [src], object_name=spec["object_name"],
                                         avoid_receptacle_types=avoid)
                    replayed = bool(reloc)
                if not reloc:
                    reloc = env.relocate(spec["object_type"], trials[env.scene_num],
                                         avoid_receptacle_types=avoid)
            obs, info = env.next_task(task_file)
        yield {"k": k, "entry": entry, "task_file": task_file, "obs": obs, "info": info,
               "relocation": reloc, "relocation_replayed": replayed,
               "task_desc_matches": entry.get("task_desc") in (None, info["task_desc"])}
