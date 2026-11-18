"""Rangkaian tugas ALFWorld (THOR) tanpa reset, untuk evaluasi memori jangka panjang.

Wrapper ini tidak mengubah kode ALFWorld. Isinya:
  1-3. Tugas berikutnya dimulai tanpa memuat ulang ruangan dan tanpa membuat controller baru.
       Batas langkah berlaku per tugas.
  4.   Status panas/dingin/bersih dikosongkan setiap tugas baru dimulai.
  5.   Reward sparse. Dense reward ALFWorld disusun dari rencana tugas aslinya.
  6.   Daftar objek dihapus dari observasi teks. Yang tersisa: daftar wadah, lokasi agen, hasil aksi.
  7.   Daftar perintah valid tidak diberikan ke agen. Gantinya command_space() / gbnf().
  8.   Agen menyebut objek tanpa nomor ("take mug from countertop 1"). Nomor objek dicari di sini.
  9-10. relocate() memindahkan objek ke posisinya pada trial lain di ruangan yang sama,
       lalu memperbarui objectId yang berubah.
  11.  Expert bawaan ALFWorld bisa dijalankan untuk memvalidasi tugas (run_expert).
  12.  Status tujuan dicek saat tugas diberikan (info["won_initial"]).
  13,15. Ground truth untuk evaluator: lokasi objek dan objek yang terlihat di frame.
  14.  Goal-condition di awal tugas dicatat (info["gc_initial"]).
Penyesuaian agar tugas tetap bisa diselesaikan di tengah rangkaian (ditemukan lewat scripts/validate_chains.py):
  a. Objek yang masih dipegang di akhir tugas (mis. tugas look_at) ditaruh sebelum tugas baru,
     dan tempatnya diberitahukan di observasi awal tugas berikutnya.
  b. Wadah dari receps.json semua trial di ruangan yang sama digabung, dan wadah statis yang tidak ada
     di receps.json mana pun dicari titik pandangnya (mis. garbagecan di FloorPlan424).
  c. Status buka/tutup wadah disamakan dengan THOR setelah setiap langkah.
  d. Tugas pick_two diberi salinan objek kedua dari trial-nya sendiri bila ruangan hanya punya satu.
  e. 'move' yang gagal karena titik penempatan tidak tampak diulang dari sudut pandang lain.
  f. heat/cool/clean yang gagal karena alat penuh memakai efek PDDL langsung (dicatat di fallback_log).
  g. relocate hanya diterima bila objek tampak dari titik pandang wadah barunya, dan tidak memindahkan
     objek keluar/masuk wadah bertitik tempel tunggal (coffeemachine, stoveburner, dll.) karena THOR
     tetap menganggap wadah itu penuh setelah isinya dipindah lewat SetObjectPoses. Lewat iter_chain,
     objek juga tidak dipindah ke wadah tujuan tugas berikutnya.
  i. Wadah tujuan yang penuh saat agen menaruh objek dicatat di info["target_full"]. Rangkaian yang dipakai
     berasal dari manifest tervalidasi (chain_manifest.py, scripts/build_chain_manifest.py).
  h. run_gt_solver() membuktikan tugas bisa diselesaikan lewat antarmuka agen.
Ground truth (gt_*) hanya untuk evaluator dan tidak boleh masuk prompt agen.
"""
import json
import math
import os
import random
import re
import traceback

import alfworld.agents
from alfworld.agents.controller import OracleAgent
from alfworld.agents.expert import HandCodedThorAgent, HandCodedAgentTimeout, HandCodedAgentFailed
from alfworld.env.thor_env import ThorEnv
from alfworld.gen.utils.image_util import decompress_mask

REWARD_CONFIG = os.path.join(alfworld.agents.__path__[0], "config", "rewards.json")
NOTHING = "Nothing happens."

# Wadah bertitik tempel tunggal (ObjectSpecificReceptacle di AI2-THOR). Bila isinya dipindah lewat
# SetObjectPoses, THOR tetap menganggapnya penuh ("X is full right now") walau metadata sudah kosong.
SLOT_RECEPTACLES = {"CoffeeMachine", "StoveBurner", "Toaster", "ToiletPaperHanger", "TowelHolder", "HandTowelHolder"}

# Seluruh jenis objek yang bisa disebut agen, terlepas dari ada atau tidaknya di ruangan.
OBJECT_VOCAB = sorted({o.lower() for o in OracleAgent.OBJECTS if "Sliced" not in o})


class _TaskArgs:
    reward_config = REWARD_CONFIG


def _load_traj(task_file):
    with open(task_file) as f:
        return json.load(f)


def _obj_type(num_id):
    """'mug 2' -> 'mug'."""
    return num_id.rsplit(" ", 1)[0]


def _strip_num(name):
    """'mug 2' -> 'mug', 'mug' -> 'mug'."""
    return re.sub(r"\s+\d+$", "", name.strip())


class ChainedAlfredThorEnv:

    def __init__(self, screen_size=300, max_steps_per_task=50, expert_max_steps=200,
                 release_held_between_tasks=True, seed=0):
        self.env = ThorEnv(player_screen_height=screen_size, player_screen_width=screen_size,
                           smooth_nav=False, save_frames_to_disk=False)
        self.max_steps = max_steps_per_task
        self.expert_max_steps = expert_max_steps
        self.release_held = release_held_between_tasks
        self.seed = seed
        self.rng = random.Random(seed)
        random.seed(seed)  # template goal ALFWorld memakai RNG global

        self.controller = None
        self.expert = HandCodedThorAgent(self.env, max_steps=expert_max_steps)
        self.scene_num = None
        self.task_idx = -1
        self.trace = []          # log per langkah untuk metrik
        self.task_log = []       # log per tugas
        self.fallback_log = []   # heat/cool/clean yang memakai efek langsung karena alat penuh
        self._release_note = ""
        self._last_good_recep = "nothing"

    # ------------------------------------------------------------------ rangkaian tugas
    def start_chain(self, task_file):
        """Tugas pertama: muat ruangan seperti ALFWorld biasa."""
        traj = _load_traj(task_file)
        scene = traj["scene"]
        self.scene_num = scene["scene_num"]
        # RNG diulang per rangkaian supaya hasilnya tidak bergantung pada urutan ruangan yang dijalankan
        self.rng = random.Random(self.seed)
        random.seed(self.seed)
        self.env.reset("FloorPlan%d" % self.scene_num)
        self.env.restore_scene(scene["object_poses"], scene["object_toggles"], scene["dirty_and_empty"])
        self.env.step(dict(scene["init_action"]))
        self.controller = OracleAgent(self.env, traj, os.path.dirname(task_file),
                                      load_receps=True, debug=False, goal_desc_human_anns_prob=0.0)
        self.merged_receptacles = self._merge_room_receptacles(task_file)
        self.discovered_receptacles = self._discover_missing_receptacles()
        self.task_idx = -1
        self.trace, self.task_log, self.fallback_log = [], [], []
        return self._begin_task(task_file, traj)

    def _merge_room_receptacles(self, task_file):
        """receps.json ALFWorld dibuat per trial dan hanya memuat wadah yang terjelajahi pada trial itu.
        Rangkaian memakai receps.json tugas pertama, sehingga wadah tujuan tugas lain (mis. garbagecan)
        bisa tidak ada. Tambahkan wadah dari receps.json semua trial di ruangan yang sama. Nomor wadah
        yang sudah ada tidak berubah, wadah baru diberi nomor berikutnya. Mengembalikan jumlah wadah baru."""
        split_root = os.path.dirname(os.path.dirname(os.path.dirname(task_file)))
        suffix = "-%d" % self.scene_num
        live = {o["objectId"]: o for o in self.env.last_event.metadata["objects"]}
        c, added = self.controller, 0
        for folder in sorted(os.listdir(split_root)):
            if not folder.endswith(suffix):
                continue
            for trial in sorted(os.listdir(os.path.join(split_root, folder))):
                f = os.path.join(split_root, folder, trial, "receps.json")
                if not os.path.isfile(f):
                    continue
                with open(f) as fh:
                    receps = json.load(fh)
                for oid, rec in receps.items():
                    if oid in c.receptacles or oid not in live:
                        continue
                    rec = dict(rec)
                    if "mask" in rec:
                        rec["mask"] = decompress_mask(rec["mask"])
                    if "Basin" in oid and not rec["object_type"].endswith("Basin"):
                        rec["object_type"] += "Basin"
                    rec["closed"] = (not live[oid]["isOpen"]) if live[oid].get("openable") else None
                    rec["num_id"] = "%s %d" % (rec["object_type"].lower(),
                                               c.get_next_num_id(rec["object_type"], c.receptacles))
                    c.receptacles[oid] = rec
                    added += 1
        return added

    def _discover_missing_receptacles(self):
        """Sebagian wadah statis tidak ada di receps.json trial mana pun (mis. garbagecan di FloorPlan424,
        padahal ada tugas 'put two soapbar in garbagecan'). Cari titik pandangnya dengan cara yang sama
        seperti OracleAgent.explore_scene: teleport ke titik-titik layout ALFRED dan ambil titik dengan
        piksel terbanyak. Pose agen dikembalikan. Mengembalikan daftar wadah yang ditambahkan."""
        c = self.controller
        missing = {o["objectId"]: o for o in self.env.last_event.metadata["objects"]
                   if o["objectId"] not in c.receptacles
                   and (o["objectType"] + ("Basin" if "Basin" in o["objectId"] and not o["objectType"].endswith("Basin") else ""))
                   in c.STATIC_RECEPTACLES}
        if not missing:
            return []
        agent = self.env.last_event.metadata["agent"]
        restore = dict(action="TeleportFull", x=agent["position"]["x"], y=agent["position"]["y"],
                       z=agent["position"]["z"], rotation=agent["rotation"]["y"], horizon=agent["cameraHorizon"],
                       rotateOnTeleport=False)
        best = {}
        for point in c.openable_points.values():
            loc = dict(action="TeleportFull", x=point[0], y=agent["position"]["y"], z=point[1],
                       rotateOnTeleport=False, rotation=point[2], horizon=point[3])
            if not self.env.step(loc).metadata["lastActionSuccess"]:
                continue
            counts, color_to_id = c.get_instance_seg()
            for color, n in counts.items():
                oid = color_to_id.get(color)
                if oid in missing and n > best.get(oid, (0, None))[0]:
                    best[oid] = (n, loc)
        # titik layout ALFRED tidak selalu menghadap semua wadah: cari dari posisi terjangkau terdekat
        still = [oid for oid in missing if oid not in best]
        if still:
            ev = self.env.step(dict(action="GetReachablePositions"))
            reach = ev.metadata.get("reachablePositions") or ev.metadata.get("actionReturn") or []
            for oid in still:
                op = missing[oid]["position"]
                near = sorted(reach, key=lambda q: math.dist((q["x"], q["z"]), (op["x"], op["z"])))
                near = [q for q in near if math.dist((q["x"], q["z"]), (op["x"], op["z"])) >= 0.5][:8]
                for q in near:
                    yaw = math.degrees(math.atan2(op["x"] - q["x"], op["z"] - q["z"]))
                    for horizon in (30, 60, 0):
                        loc = dict(action="TeleportFull", x=q["x"], y=agent["position"]["y"], z=q["z"],
                                   rotateOnTeleport=False, rotation=(round(yaw / 90.0) * 90) % 360,
                                   horizon=horizon)
                        if not self.env.step(loc).metadata["lastActionSuccess"]:
                            continue
                        counts, color_to_id = c.get_instance_seg()
                        n = sum(k for col, k in counts.items() if color_to_id.get(col) == oid)
                        if n > best.get(oid, (0, None))[0]:
                            best[oid] = (n, loc)
        self.env.step(restore)
        added = []
        for oid, (n, loc) in best.items():
            o = missing[oid]
            otype = o["objectType"] + ("Basin" if "Basin" in oid and not o["objectType"].endswith("Basin") else "")
            c.receptacles[oid] = {
                "object_id": oid, "object_type": otype, "locs": loc, "num_pixels": n,
                "num_id": "%s %d" % (otype.lower(), c.get_next_num_id(otype, c.receptacles)),
                "closed": (not o["isOpen"]) if o.get("openable") else None,
            }
            added.append(c.receptacles[oid]["num_id"])
        return added

    def next_task(self, task_file):
        """Tugas berikutnya di ruangan yang sama, tanpa reset lingkungan."""
        traj = _load_traj(task_file)
        if traj["scene"]["scene_num"] != self.scene_num:
            raise ValueError("next_task harus di FloorPlan%d, bukan FloorPlan%d"
                             % (self.scene_num, traj["scene"]["scene_num"]))
        if self.release_held:
            self._release_held_object()
        return self._begin_task(task_file, traj)

    def _begin_task(self, task_file, traj):
        self.task_idx += 1
        self.task_file, self.traj = task_file, traj
        self.env.reset_states()                                       # (4)
        spawned = self._ensure_target_instances(traj)
        self._sync_open_states()
        # Template deskripsi tugas dipilih dengan RNG global (utils/misc.py), yang juga dipakai expert
        # selama langkah. Diulang per posisi tugas supaya deskripsinya sama untuk setiap agen.
        random.seed(self.seed * 100003 + self.task_idx)
        self.env.set_task(traj, _TaskArgs, reward_type="sparse")      # (5)

        # controller lama dipakai ulang supaya agen tidak di-teleport (explore_scene)
        self.controller.traj_data = traj
        self.controller.traj_root = os.path.dirname(task_file)
        self.controller.print_intro()                                 # teks tujuan baru
        self.task_desc = self.controller.intro.split("Your task is to: ", 1)[-1].strip()
        self._raw_feedback = self.controller.feedback

        self.steps, self.done = 0, False
        won0 = self.env.get_goal_satisfied()
        s, ts = self.env.get_goal_conditions_met()
        self.task_info = {
            "task_idx": self.task_idx,
            "task_file": task_file,
            "task_type": traj["task_type"],
            "task_desc": self.task_desc,
            "won_initial": bool(won0),                                # (12)
            "gc_initial": s / float(ts),                              # (14)
            "target_locations_initial": self.gt_target_locations(),  # (13)
            "agent_location_initial": self.controller.curr_recep,
            "spawned_instances": spawned,
            # wadah tujuan tugas ini ternyata penuh saat agen menaruh objek ke sana (lihat _put_from_other_views)
            "target_full": False,
            "target_full_events": [],
        }
        self.task_log.append(self.task_info)

        # expert untuk validasi (11)
        self.expert.reset(task_file)
        self._expert_prev = ""

        obs = self._intro_obs()
        self._release_note = ""
        self.last_obs = obs
        return obs, self._info(won=bool(won0))

    def _ensure_target_instances(self, traj, need=2):
        """Tugas pick_two butuh minimal dua objek sejenis. ALFRED membuat salinan objek lewat object_poses
        trial-nya sendiri (objectName yang sama muncul dua kali), sedangkan ruangan rangkaian berasal dari
        trial tugas pertama yang bisa hanya punya satu. Kekurangannya ditambahkan dari pose trial tugas ini,
        dipilih yang paling jauh dari objek sejenis yang sudah ada. Mengembalikan jumlah objek yang ditambah."""
        if not traj["task_type"].startswith("pick_two"):
            return 0
        typ = traj["pddl_params"]["object_target"]
        have = [o for o in self.env.last_event.metadata["objects"] if o["objectType"] == typ]
        if len(have) >= need:
            return 0
        src = [p for p in traj["scene"]["object_poses"] if p["objectName"].split("_")[0] == typ]
        pos = lambda p: (p["position"]["x"], p["position"]["y"], p["position"]["z"])
        src.sort(key=lambda p: min((math.dist(pos(p), pos(o)) for o in have), default=0.0), reverse=True)
        extra = src[: need - len(have)]
        if not extra:
            return 0
        self._set_poses(self._current_poses() + extra)
        n = sum(o["objectType"] == typ for o in self.env.last_event.metadata["objects"]) - len(have)
        self.task_log.append({"spawn": {"object_type": typ, "added": n}, "before_task_idx": self.task_idx})
        return n

    def _intro_obs(self):
        receps = ", ".join("a %s" % r for r in self.receptacle_names())
        where = ("You are at the %s." % self.controller.curr_recep
                 if self.controller.curr_recep != "nothing" else "You are in the middle of a room.")
        note = (self._release_note + " ") if self._release_note else ""
        return ("%s%s Looking quickly around the room, you see %s.\n\nYour task is to: %s"
                % (note, where, receps, self.task_desc))

    def _release_held_object(self):
        """Taruh objek yang masih dipegang (mis. sisa tugas look_at) sebelum tugas baru.
        Dicoba di wadah tempat agen berada (dibuka dulu bila tertutup), lalu di wadah permukaan terbuka
        lain. Tempat akhirnya diberitahukan ke agen di observasi awal tugas berikutnya."""
        if not self.controller.inventory:
            return None
        held = self.controller.inventory[0]
        here = self.controller.curr_recep
        surfaces = [r["num_id"] for r in self.controller.receptacles.values() if r["closed"] is None]
        candidates = ([here] if here != "nothing" else []) + [r for r in surfaces if r != here]
        placed = None
        for r in candidates:
            if self.controller.curr_recep != r:
                self.controller.step("go to %s" % r)
                if self.controller.curr_recep != r:
                    continue
            rec = next((x for x in self.controller.receptacles.values() if x["num_id"] == r), None)
            opened = bool(rec and rec["closed"])
            if opened:
                self.controller.step("open %s" % r)
            self.controller.step("move %s to %s" % (held, r))
            if opened:
                self.controller.step("close %s" % r)
            if not self.controller.inventory:
                placed = r
                break
        if placed is None:   # jalan terakhir: jatuhkan
            ev = self.env.step(dict(action="DropHandObject"))
            if ev.metadata["lastActionSuccess"]:
                self.controller.inventory.pop()
        self._release_note = ("You put the %s in/on the %s." % (_obj_type(held), placed)) if placed else ""
        self.task_log.append({"release": {"object": held, "to": placed,
                                          "ok": not self.controller.inventory},
                              "after_task_idx": self.task_idx})
        return held

    # ------------------------------------------------------------------ aksi agen
    def step(self, action):
        if self.done:
            return self.last_obs, True, self._info()
        cmd = self.resolve(action)                                     # (8)
        raw = self._controller_step(cmd) if cmd else NOTHING
        return self._after_step(action, cmd, raw)

    def step_raw(self, cmd):
        """Perintah format asli ALFWorld (bernomor). Hanya untuk expert/validasi."""
        if self.done:
            return self.last_obs, True, self._info()
        raw = self._controller_step(cmd)
        return self._after_step(cmd, cmd, raw)

    def _controller_step(self, cmd):
        self._last_good_recep = self.controller.curr_recep
        raw = self.controller.step(cmd)
        if raw == NOTHING:
            raw = (self._goto_nearby(cmd) or self._put_from_other_views(cmd)
                   or self._appliance_fallback(cmd) or raw)
        self._sync_open_states()
        return raw

    def _goto_nearby(self, cmd):
        """'go to X' gagal bila teleport ke titik ALFWorld bertabrakan (mis. objek di tangan menabrak pintu
        kulkas yang terbuka). Coba titik sedikit mundur atau bergeser dengan arah pandang yang sama."""
        m = re.fullmatch(r"go to (.+)", cmd or "")
        if not m or m.group(1) == self._last_good_recep:   # sudah di sana: tetap "Nothing happens."
            return None
        rec = next((r for r in self.controller.receptacles.values() if r["num_id"] == m.group(1)), None)
        if rec is None:
            return None
        c = self.controller
        prev_recep, prev_loc = self._last_good_recep, c.curr_loc
        orig = rec["locs"]
        yaw = math.radians(orig["rotation"])
        fwd = (math.sin(yaw), math.cos(yaw))        # arah pandang pada bidang x-z
        side = (fwd[1], -fwd[0])
        offsets = [(-0.25, 0), (-0.5, 0), (0, 0.25), (0, -0.25), (-0.25, 0.25), (-0.25, -0.25)]
        raw = None
        try:
            for back, lat in offsets:
                tx, tz = orig["x"] + back * fwd[0] + lat * side[0], orig["z"] + back * fwd[1] + lat * side[1]
                rec["locs"] = dict(orig, x=tx, z=tz)
                c.curr_recep = "nothing"            # cegah controller menganggap agen sudah di sana
                out = c.step(cmd)
                pos = self.env.last_event.metadata["agent"]["position"]
                if out != NOTHING and math.dist((pos["x"], pos["z"]), (tx, tz)) < 0.05:
                    raw = out
                    break
        finally:
            rec["locs"] = orig
        if raw is None:   # tetap di tempat semula
            c.curr_recep, c.curr_loc = prev_recep, prev_loc
        if raw:
            self.fallback_log.append({"task_idx": self.task_idx, "step": self.steps + 1, "command": cmd,
                                      "kind": "goto_nearby"})
        return raw

    # PutObject THOR hanya mencari titik penempatan yang tampak dari kamera. Di tengah rangkaian wadah bisa
    # sudah berisi objek lain sehingga titik yang tersisa tidak tampak dari sudut pandang bawaan ALFWorld.
    _PUT_VIEWS = (("LookDown",), ("LookDown", "LookDown"), ("LookUp",), ("MoveBack", "LookDown"),
                  ("RotateLeft",), ("RotateRight",))

    def _put_from_other_views(self, cmd):
        """Ulangi 'move X to Y' yang sah dari beberapa sudut pandang lain, lalu kembalikan pose agen."""
        m = re.fullmatch(r"move (.+?) to (.+)", cmd or "")
        inv = self.env.last_event.metadata["inventoryObjects"]
        if not m or not inv or self.controller.curr_recep != m.group(2):
            return None
        if cmd not in self.controller.get_admissible_commands():
            return None
        obj, tar = m.groups()
        rid = next((i for i, r in self.controller.receptacles.items() if r["num_id"] == tar), None)
        agent = self.env.last_event.metadata["agent"]
        pose = dict(action="TeleportFull", x=agent["position"]["x"], y=agent["position"]["y"],
                    z=agent["position"]["z"], rotation=agent["rotation"]["y"], horizon=agent["cameraHorizon"],
                    rotateOnTeleport=False)
        placed, errors = False, set()
        for view in self._PUT_VIEWS:
            for a in view:
                self.env.step(dict(action=a))
            ev = self.env.step(dict(action="PutObject", objectId=inv[0]["objectId"], receptacleObjectId=rid,
                                    forceAction=True, placeStationary=True))
            placed = ev.metadata["lastActionSuccess"]
            errors.add(ev.metadata.get("errorMessage", "")[:80])
            self.env.step(pose)
            if placed:
                break
        if not placed:
            self._note_target_full(cmd, rid, errors)
            return None
        if obj in self.controller.inventory:
            self.controller.inventory.remove(obj)
        self.fallback_log.append({"task_idx": self.task_idx, "step": self.steps + 1, "command": cmd,
                                  "kind": "put_view"})
        return "You put the %s to the %s." % (obj, tar)

    _FULL_ERRORS = ("full right now", "No valid positions", "No valid Receptacle")

    def _note_target_full(self, cmd, rid, errors):
        """Catat bila 'move' yang sah ke wadah tujuan tugas tetap gagal dari semua sudut pandang karena
        wadah itu penuh. Manifest rangkaian sudah membuang tugas yang penuh bila diselesaikan dengan benar,
        jadi kejadian ini berasal dari keadaan ruangan akibat tindakan agen sendiri. Kegagalannya tetap
        dihitung, tetapi dilaporkan terpisah (info["target_full"])."""
        parent = self.traj["pddl_params"].get("parent_target") if self.traj else None
        rec = self.controller.receptacles.get(rid) if rid else None
        if not parent or not rec or rec["object_type"] != parent:
            return
        if not any(k in e for e in errors for k in self._FULL_ERRORS):
            return
        self.task_info["target_full"] = True
        self.task_info["target_full_events"].append(
            {"step": self.steps + 1, "command": cmd, "errors": sorted(errors)})

    def _sync_open_states(self):
        """heat/cool/clean di controller ALFWorld membuka lalu menutup microwave/kulkas secara fisik tanpa
        memperbarui catatan 'closed' controller. Akibatnya agen diberi tahu 'The fridge 1 is open.' padahal
        tertutup, dan isinya tidak terlihat. Samakan catatan dengan metadata THOR."""
        meta = {o["objectId"]: o for o in self.env.last_event.metadata["objects"]}
        for oid, rec in self.controller.receptacles.items():
            o = meta.get(oid)
            if rec.get("closed") is not None and o is not None and o.get("openable"):
                rec["closed"] = not o["isOpen"]

    def _appliance_fallback(self, cmd):
        """heat/cool/clean di controller ALFWorld menaruh objek secara fisik ke microwave/kulkas/wastafel.
        Di tengah rangkaian alat itu bisa sudah penuh oleh objek dari tugas sebelumnya, sehingga PutObject
        THOR gagal ("No valid positions to place object found") dan tugas menjadi mustahil. Versi PDDL
        ALFWorld tidak membatasi kapasitas. Jadi bila perintahnya sah (objek dipegang, agen di alat yang
        benar), status objek diubah langsung seperti efek PDDL-nya."""
        m = re.fullmatch(r"(heat|cool|clean) (.+?) with (.+)", cmd or "")
        inv = self.env.last_event.metadata["inventoryObjects"]
        if not m or not inv or self.controller.curr_recep != m.group(3):
            return None
        if cmd not in self.controller.get_admissible_commands():
            return None
        verb, obj, tar = m.groups()
        oid = inv[0]["objectId"]
        {"heat": self.env.heated_objects, "cool": self.env.cooled_objects,
         "clean": self.env.cleaned_objects}[verb].add(oid)
        if verb == "clean":   # tampilan visual ikut bersih bila objeknya bisa kotor
            o = next((x for x in self.env.last_event.metadata["objects"] if x["objectId"] == oid), None)
            if o and o.get("dirtyable") and o.get("isDirty"):
                self.env.step(dict(action="CleanObject", objectId=oid, forceAction=True))
        self.fallback_log.append({"task_idx": self.task_idx, "step": self.steps + 1, "command": cmd,
                                  "kind": "appliance"})
        return "You %s the %s using the %s." % (verb, obj, tar)

    def _after_step(self, action, cmd, raw):
        self.steps += 1
        self._raw_feedback = raw
        obs = self.sanitize(raw, cmd)                                  # (6)
        won = bool(self.env.get_goal_satisfied())
        self.done = won or self.steps >= self.max_steps                # (3)
        self.last_obs = obs
        self.trace.append({
            "task_idx": self.task_idx, "step": self.steps, "action": action, "command": cmd,
            "obs": obs, "raw": raw, "success": raw != NOTHING, "location": self.controller.curr_recep,
            "visible_types": self.gt_visible_types(),                  # (15)
            "won": won,
        })
        return obs, self.done, self._info(won=won)

    def _info(self, won=None):
        if won is None:
            won = bool(self.env.get_goal_satisfied())
        s, ts = self.env.get_goal_conditions_met()
        return {"won": won, "goal_condition_success_rate": s / float(ts), "steps": self.steps,
                "task_idx": self.task_idx, "task_desc": self.task_desc,
                "won_initial": self.task_info["won_initial"], "gc_initial": self.task_info["gc_initial"],
                "target_full": self.task_info["target_full"]}

    # ------------------------------------------------------------------ resolusi perintah (8)
    def receptacle_names(self):
        return [r["num_id"] for r in self.controller.receptacles.values()]

    def _held(self):
        return self.controller.inventory[0] if self.controller.inventory else None

    def _visible_of_type(self, otype, recep=None):
        """Objek bernomor berjenis otype yang tampak. visible_objects controller berisi semua objek di
        frame, jadi bila recep disebut, utamakan objek yang benar-benar berada di wadah itu."""
        inv = set(self.controller.inventory)
        cands = [n for n in (getattr(self.controller, "visible_objects", []) or [])
                 if n not in inv and _obj_type(n) == otype]
        if recep and len(cands) > 1:
            rid = next((oid for oid, r in self.controller.receptacles.items() if r["num_id"] == recep), None)
            id_by_num = {r["num_id"]: oid for oid, r in self.controller.objects.items()}
            meta = {o["objectId"]: o for o in self.env.last_event.metadata["objects"]}
            on_recep = lambda n: rid in ((meta.get(id_by_num.get(n)) or {}).get("parentReceptacles") or [])
            cands.sort(key=lambda n: not on_recep(n))
        return cands[0] if cands else None

    def resolve(self, action):
        """Ubah perintah agen (tanpa nomor objek) ke perintah ALFWorld bernomor.
        Mengembalikan None bila perintah tidak bisa dijalankan pada keadaan saat ini."""
        a = " ".join(str(action).lower().strip().rstrip(".").split())
        receps = set(self.receptacle_names())
        held = self._held()
        cmd = None

        if a in ("look", "inventory"):
            cmd = a
        elif m := re.fullmatch(r"(go to|open|close) (.+)", a):
            if m.group(2) in receps:
                cmd = a
        elif m := re.fullmatch(r"take (.+?) from (.+)", a):
            obj = self._visible_of_type(_strip_num(m.group(1)), m.group(2))
            if obj and m.group(2) in receps:
                cmd = "take %s from %s" % (obj, m.group(2))
        elif m := re.fullmatch(r"(?:put|move) (.+?) (?:in/on|in|on|to|into|onto) (.+)", a):
            if held and _obj_type(held) == _strip_num(m.group(1)) and m.group(2) in receps:
                cmd = "move %s to %s" % (held, m.group(2))
        elif m := re.fullmatch(r"(heat|cool|clean) (.+?) with (.+)", a):
            if held and _obj_type(held) == _strip_num(m.group(2)) and m.group(3) in receps:
                cmd = "%s %s with %s" % (m.group(1), held, m.group(3))
        elif m := re.fullmatch(r"use (.+)", a):
            obj = self._visible_of_type(_strip_num(m.group(1)))
            if obj:
                cmd = "use %s" % obj
        elif m := re.fullmatch(r"examine (.+)", a):
            tgt = m.group(1)
            if tgt in receps:
                cmd = "examine %s" % tgt
            elif held and _obj_type(held) == _strip_num(tgt):
                cmd = "examine %s" % held

        # prasyarat dicek dengan daftar perintah valid ALFWorld secara internal (tidak diberikan ke agen)
        if cmd and cmd not in ("look", "inventory") and cmd not in self.controller.get_admissible_commands():
            return None
        return cmd

    # ------------------------------------------------------------------ observasi (6)
    def sanitize(self, raw, cmd):
        """Observasi teks tanpa daftar objek dan tanpa nomor objek."""
        if raw == NOTHING or not cmd:
            return NOTHING
        c = self.controller
        verb = cmd.split()[0]
        held_type = _obj_type(self._held()) if self._held() else None

        if verb == "go":
            r = c.curr_recep
            rec = next((x for x in c.receptacles.values() if x["num_id"] == r), None)
            state = ""
            if rec and rec["closed"] is not None:
                state = " The %s is %s." % (r, "closed" if rec["closed"] else "open")
            return "You arrive at the %s.%s" % (r, state)
        if verb == "open":
            r = cmd[len("open "):]
            return "You open the %s. The %s is open." % (r, r)
        if verb == "close":
            return "You close the %s." % cmd[len("close "):]
        if verb == "take":
            m = re.fullmatch(r"take (.+?) from (.+)", cmd)
            return "You pick up the %s from the %s." % (_obj_type(m.group(1)), m.group(2))
        if verb == "move":
            m = re.fullmatch(r"move (.+?) to (.+)", cmd)
            return "You put the %s in/on the %s." % (_obj_type(m.group(1)), m.group(2))
        if verb in ("heat", "cool", "clean"):
            m = re.fullmatch(r"\w+ (.+?) with (.+)", cmd)
            return "You %s the %s using the %s." % (verb, _obj_type(m.group(1)), m.group(2))
        if verb == "use":
            return "You turn on the %s." % _obj_type(cmd[len("use "):])
        if verb == "inventory":
            return "You are carrying: a %s." % held_type if held_type else "You are not carrying anything."
        if verb == "look":
            if c.curr_recep == "nothing":
                return "You are in the middle of a room."
            return "You are facing the %s." % c.curr_recep
        if verb == "examine":
            tgt = cmd[len("examine "):]
            if tgt in set(self.receptacle_names()):
                return "You look closely at the %s." % tgt
            # status objek yang dipegang (panas/dingin/bersih) boleh diketahui, nomornya dihapus
            return re.sub(re.escape(tgt), _obj_type(tgt), raw)
        return NOTHING

    # ------------------------------------------------------------------ ruang perintah (7)
    def command_space(self):
        receps = self.receptacle_names()
        tool = lambda key: [r for r in receps if key in r]
        return {
            "receptacles": receps,
            "objects": OBJECT_VOCAB,
            "microwaves": tool("microwave"),
            "fridges": tool("fridge"),
            "cleaners": [r for r in receps if "sinkbasin" in r or "bathtubbasin" in r],
            "templates": ["go to {recep}", "open {recep}", "close {recep}", "take {obj} from {recep}",
                          "move {obj} to {recep}", "heat {obj} with {microwave}", "cool {obj} with {fridge}",
                          "clean {obj} with {cleaner}", "use {obj}", "examine {obj}", "examine {recep}",
                          "inventory", "look"],
        }

    def gbnf(self):
        """Grammar llama.cpp yang hanya bergantung pada daftar wadah dan kosakata objek ALFWorld."""
        cs = self.command_space()
        alt = lambda xs: " | ".join('"%s"' % x for x in xs) if xs else '"none"'
        return "\n".join([
            'root ::= "go to " recep | "open " recep | "close " recep | "take " obj " from " recep'
            ' | "move " obj " to " recep | "heat " obj " with " microwave | "cool " obj " with " fridge'
            ' | "clean " obj " with " cleaner | "use " obj | "examine " obj | "examine " recep'
            ' | "inventory" | "look"',
            "recep ::= " + alt(cs["receptacles"]),
            "obj ::= " + alt(cs["objects"]),
            "microwave ::= " + alt(cs["microwaves"]),
            "fridge ::= " + alt(cs["fridges"]),
            "cleaner ::= " + alt(cs["cleaners"]),
        ])

    # ------------------------------------------------------------------ perpindahan objek (9-10)
    def relocate(self, object_type, source_task_files, min_distance=0.3, object_name=None,
                 avoid_receptacle_types=()):
        """Pindahkan satu objek berjenis object_type ke posisinya pada trial lain di ruangan yang sama.
        source_task_files: satu path atau list path traj_data.json, dicoba berurutan.
        object_name: bila diisi, hanya objek bernama itu yang boleh dipindah (untuk memutar ulang manifest).
        avoid_receptacle_types: jenis wadah tujuan yang tidak boleh dipakai, biasanya parent_target tugas
        berikutnya, supaya perpindahan tidak menyelesaikan sebagian tugas itu.
        Perpindahan hanya diterima bila objek berakhir di wadah yang dikenal agen dan berbeda dari
        wadah sebelumnya. Bila tidak, posisi dikembalikan dan kandidat berikutnya dicoba.
        Mengembalikan dict keterangan, atau None bila tidak ada perpindahan yang valid."""
        if isinstance(source_task_files, str):
            source_task_files = [source_task_files]
        if self.release_held:
            self._release_held_object()   # perpindahan terjadi di antara tugas, tangan agen kosong

        attempts = 0
        for src_file in source_task_files:
            src = _load_traj(src_file)
            if src["scene"]["scene_num"] != self.scene_num:
                raise ValueError("trial sumber harus di ruangan yang sama")
            src_pose = {p["objectName"]: p for p in src["scene"]["object_poses"]}
            meta = self.env.last_event.metadata
            held_ids = {o["objectId"] for o in meta["inventoryObjects"]}
            cands = [o for o in meta["objects"] if o["objectType"] == object_type and o["pickupable"]
                     and o["objectId"] not in held_ids and o["name"] in src_pose
                     and not self._in_slot_receptacle(o)
                     and (object_name is None or o["name"] == object_name)]
            self.rng.shuffle(cands)
            for o in cands:
                p, q = src_pose[o["name"]]["position"], o["position"]
                if math.dist((p["x"], p["y"], p["z"]), (q["x"], q["y"], q["z"])) <= min_distance:
                    continue
                attempts += 1
                before = self._recep_names(o.get("parentReceptacles"))
                snapshot = self._current_poses()
                ev = self._set_poses(self._current_poses(override={o["name"]: src_pose[o["name"]]}))
                after_obj = self._meta_by_name(o["name"])
                after = self._recep_names(after_obj and after_obj.get("parentReceptacles"))
                if (ev.metadata["lastActionSuccess"] and after and set(after) != set(before)
                        and not self._in_slot_receptacle(after_obj)
                        and not (self._parent_types(after_obj) & set(avoid_receptacle_types))
                        and any(self._visible_from_recep(after_obj["objectId"], r) for r in after)
                        and self._still_in(o["name"], after)):
                    result = {"ok": True, "object_type": object_type, "object_name": o["name"],
                              "before": before, "after": after, "source": src_file, "attempts": attempts}
                    self.task_log.append({"relocation": result, "after_task_idx": self.task_idx})
                    return result
                self._set_poses(snapshot)   # tidak valid, kembalikan
        self.task_log.append({"relocation": {"ok": False, "object_type": object_type, "attempts": attempts},
                              "after_task_idx": self.task_idx})
        return None

    def _still_in(self, name, receps, settle_steps=2):
        """Cek pemeriksaan visibilitas membuka lalu menutup wadah. Pose dari trial lain bisa membuat objek
        terdorong keluar saat laci/lemari ditutup dan jatuh ke lantai, padahal lantai bukan wadah ALFWorld
        sehingga objek itu tidak bisa diambil lagi. Pastikan objek masih di wadah barunya."""
        for _ in range(settle_steps):
            self.env.step(dict(action="Pass"))
        o = self._meta_by_name(name)
        return bool(o) and set(self._recep_names(o.get("parentReceptacles"))) == set(receps)

    def _parent_types(self, obj):
        """Jenis objek THOR wadah tempat obj berada, mis. {'CounterTop'} atau {'SinkBasin', 'Sink'}."""
        meta = {o["objectId"]: o["objectType"] for o in self.env.last_event.metadata["objects"]}
        return {meta.get(p, p.split("|")[0]) for p in ((obj or {}).get("parentReceptacles") or [])}

    def _in_slot_receptacle(self, obj):
        """Objek sedang berada di wadah bertitik tempel tunggal (lihat SLOT_RECEPTACLES)."""
        return bool(self._parent_types(obj) & SLOT_RECEPTACLES)

    def _visible_from_recep(self, object_id, recep_name):
        """Apakah objek tampak dari titik pandang ALFWorld di wadah itu (wadah dibuka bila tertutup).
        Pose sebelumnya ditaruh dari trial lain, jadi bisa tersembunyi dan mustahil diambil agen.
        Pose agen dan keadaan wadah dikembalikan seperti semula."""
        rec = next((r for r in self.controller.receptacles.values() if r["num_id"] == recep_name), None)
        if rec is None:
            return False
        agent = self.env.last_event.metadata["agent"]
        pose = dict(action="TeleportFull", x=agent["position"]["x"], y=agent["position"]["y"],
                    z=agent["position"]["z"], rotation=agent["rotation"]["y"], horizon=agent["cameraHorizon"],
                    rotateOnTeleport=False)
        self.controller.navigate(rec["locs"])
        r_meta = next((x for x in self.env.last_event.metadata["objects"] if x["objectId"] == rec["object_id"]), {})
        opened = bool(r_meta.get("openable") and not r_meta.get("isOpen"))
        if opened:
            self.env.step(dict(action="OpenObject", objectId=rec["object_id"], forceAction=True))
        counts, color_to_id = self.controller.get_instance_seg()
        seen = any(color_to_id.get(c) == object_id and n > 0 for c, n in counts.items())
        if opened:
            self.env.step(dict(action="CloseObject", objectId=rec["object_id"], forceAction=True))
        self.env.step(pose)
        return seen

    def _meta_by_name(self, name):
        return next((o for o in self.env.last_event.metadata["objects"] if o["name"] == name), None)

    def _current_poses(self, override=None):
        """Pose semua objek yang bisa dipindah, kecuali yang sedang dipegang."""
        meta = self.env.last_event.metadata
        held_ids = {o["objectId"] for o in meta["inventoryObjects"]}
        poses = []
        for o in meta["objects"]:
            if not (o.get("pickupable") or o.get("moveable")) or o["objectId"] in held_ids:
                continue
            if override and o["name"] in override:
                poses.append(override[o["name"]])
            else:
                poses.append({"objectName": o["name"], "position": o["position"], "rotation": o["rotation"]})
        return poses

    def _set_poses(self, poses, settle_steps=2):
        old_id_by_name = {o["name"]: o["objectId"] for o in self.env.last_event.metadata["objects"]}
        ev = self.env.step(dict(action="SetObjectPoses", objectPoses=poses))
        for _ in range(settle_steps):   # beri waktu fisika agar parentReceptacles terbarui
            self.env.step(dict(action="Pass"))
        self._remap_object_ids(old_id_by_name)
        return ev

    def _remap_object_ids(self, old_id_by_name):
        """objectId AI2-THOR bisa berubah setelah SetObjectPoses. Samakan kembali catatan controller.
        Nama yang muncul lebih dari sekali (salinan objek pada tugas pick_two) tidak bisa dipasangkan,
        jadi catatan lamanya dibuang dan controller mencatat ulang saat objeknya terlihat lagi."""
        meta_objs = self.env.last_event.metadata["objects"]
        name_count = {}
        for o in meta_objs:
            name_count[o["name"]] = name_count.get(o["name"], 0) + 1
        new_id_by_name = {o["name"]: o["objectId"] for o in meta_objs if name_count[o["name"]] == 1}
        mapping = {old: new_id_by_name[n] for n, old in old_id_by_name.items()
                   if n in new_id_by_name and new_id_by_name[n] != old}
        live = {o["objectId"] for o in meta_objs}
        dropped_types = set()
        for attr in ("objects", "receptacles"):
            remapped = {}
            for oid, rec in getattr(self.controller, attr).items():
                nid = mapping.get(oid, oid)
                if attr == "objects" and nid not in live:
                    dropped_types.add(rec["object_type"])
                    continue
                rec["object_id"] = nid
                remapped[nid] = rec
            setattr(self.controller, attr, remapped)
        # nomor objek controller dihitung dari jumlah catatan, jadi rapatkan lagi agar tidak bentrok
        for t in dropped_types:
            k = 0
            for rec in self.controller.objects.values():
                if rec["object_type"] == t:
                    k += 1
                    old = rec["num_id"]
                    rec["num_id"] = "%s %d" % (old.rsplit(" ", 1)[0], k)
                    self.controller.inventory = [rec["num_id"] if i == old else i
                                                 for i in self.controller.inventory]
        if not mapping:
            return 0
        self._last_id_mapping = mapping
        for attr in ("heated_objects", "cooled_objects", "cleaned_objects"):
            setattr(self.env, attr, {mapping.get(i, i) for i in getattr(self.env, attr)})
        return len(mapping)

    # ------------------------------------------------------------------ ground truth evaluator (13, 15)
    def _recep_names(self, parent_ids):
        recs = self.controller.receptacles if self.controller else {}
        return sorted({recs[p]["num_id"] for p in (parent_ids or []) if p in recs})

    def gt_object_locations(self, object_type):
        """Wadah yang berisi objek berjenis object_type. 'held' bila sedang dipegang."""
        meta = self.env.last_event.metadata
        held = {o["objectId"] for o in meta["inventoryObjects"]}
        locs = set()
        for o in meta["objects"]:
            if o["objectType"].lower() == object_type.lower():
                if o["objectId"] in held:
                    locs.add("held")
                else:
                    locs.update(self._object_receps(o))
        return sorted(locs)

    def _object_receps(self, o):
        """Wadah tempat objek o berada. Sebagian objek tidak punya parentReceptacles yang dikenal ALFWorld
        (mis. kain di tepi bak). Untuk itu dipakai wadah terdekat secara 3D."""
        names = self._recep_names(o.get("parentReceptacles"))
        if names:
            return names
        return self._receps_by_distance(o)[:1]

    def _receps_by_distance(self, o):
        """Nama wadah diurutkan dari yang paling dekat (3D) ke objek o."""
        meta = {x["objectId"]: x for x in self.env.last_event.metadata["objects"]}
        p = o["position"]
        pos = lambda m: (m["position"]["x"], m["position"]["y"], m["position"]["z"])
        return [n for _, n in sorted((math.dist((p["x"], p["y"], p["z"]), pos(meta[rid])), r["num_id"])
                                     for rid, r in self.controller.receptacles.items() if rid in meta)]

    def task_targets(self):
        p = self.traj["pddl_params"]
        targets = [p["object_target"]]
        if p.get("toggle_target"):
            targets.append(p["toggle_target"])
        return targets

    def gt_target_locations(self):
        return {t: self.gt_object_locations(t) for t in self.task_targets()}

    def gt_visible_types(self, min_pixels=50):
        """Jenis objek yang tampak pada frame saat ini, dari segmentasi THOR."""
        counts, color_to_id = self.controller.get_instance_seg()
        types = {}
        for color, n in counts.items():
            oid = color_to_id.get(color)
            if oid and n >= min_pixels:
                t = oid.split("|")[0]
                if t in OracleAgent.OBJECTS:
                    types[t.lower()] = types.get(t.lower(), 0) + n
        return sorted(types)

    def get_frame(self):
        """Frame RGB saat ini dalam format BGR (sama dengan env.get_frames() ALFWorld)."""
        return self.env.last_event.frame[:, :, ::-1]

    # ------------------------------------------------------------------ validasi dengan expert (11)
    def expert_next_action(self):
        acs = self.controller.get_admissible_commands()
        if not self._expert_prev:
            self.expert.observe(self._raw_feedback)   # teks intro: daftar wadah
            # Di tengah rangkaian agen sudah berdiri di suatu wadah. Expert ALFWorld mengira agen
            # di tengah ruangan, jadi beri tahu lokasi dan isi wadah saat ini (tanpa mengubah dunia).
            here = self.controller.curr_recep
            if here != "nothing":
                pol = self.expert.policy
                pol.curr_recep = here
                rec = next((r for r in self.controller.receptacles.values() if r["num_id"] == here), None)
                pol.checked_inside_curr_recep = not (rec and rec["closed"])
                fresh = self.controller.step("examine %s" % here)
                if "you see" in fresh:
                    self.expert.observe(fresh)
            self._expert_prev = "look"
            return "look"
        game_state = {"admissible_commands": acs, "feedback": self._raw_feedback,
                      "won": self.env.get_goal_satisfied()}
        action = self.expert.act(game_state, 0, game_state["won"], self._expert_prev)
        self._expert_prev = action
        return action

    def run_expert(self):
        """Jalankan expert sampai tugas saat ini selesai. Mengembalikan (berhasil, jumlah langkah, alasan)."""
        saved_max, self.max_steps = self.max_steps, self.expert_max_steps
        reason = "ok"
        try:
            while not self.done:
                self.step_raw(self.expert_next_action())
            won = bool(self.env.get_goal_satisfied())
            if not won:
                reason = "step_limit"
        except HandCodedAgentTimeout:
            won, reason = False, "expert_timeout"
        except HandCodedAgentFailed:
            won, reason = False, "expert_failed"
        except Exception:
            won, reason = False, "error: " + traceback.format_exc(limit=2)
        finally:
            self.max_steps = saved_max
        return won, self.steps, reason

    # ------------------------------------------------------------------ validasi dengan ground truth
    def run_gt_solver(self, max_steps=300):
        """Selesaikan tugas saat ini memakai lokasi objek sebenarnya, tetapi semua perintah dikirim lewat
        antarmuka agen (step, tanpa nomor objek). Berhasil berarti tugas terbukti bisa diselesaikan agen
        pada keadaan rangkaian saat ini. Jumlah langkahnya menjadi acuan langkah mendekati minimal.
        Mengembalikan (berhasil, jumlah langkah, alasan)."""
        saved, self.max_steps = self.max_steps, max_steps
        try:
            reason = self._gt_solve()
        except Exception:
            reason = "error: " + traceback.format_exc(limit=2)
        finally:
            self.max_steps = saved
        won = bool(self.env.get_goal_satisfied())
        return won, self.steps, ("ok" if won else reason)

    def _gt_act(self, cmd):
        obs, done, info = self.step(cmd)
        return obs != NOTHING

    def _gt_goto(self, r):
        if self.controller.curr_recep != r and not self._gt_act("go to %s" % r):
            return False
        rec = next((x for x in self.controller.receptacles.values() if x["num_id"] == r), None)
        if rec and rec["closed"]:
            self._gt_act("open %s" % r)
        return True

    def _gt_receps_of_type(self, otype):
        return [r["num_id"] for r in self.controller.receptacles.values() if r["object_type"] == otype]

    def _gt_solve(self):
        p = self.traj["pddl_params"]
        ttype, otype = self.traj["task_type"], p["object_target"]
        name = otype.lower()
        targets = set(self._gt_receps_of_type(p["parent_target"])) if p.get("parent_target") else set()
        n_objects = 2 if ttype.startswith("pick_two") else 1
        order = sorted(targets)
        if n_objects == 2:
            # kedua objek harus di wadah yang sama: utamakan wadah tujuan yang sudah berisi objek sejenis
            filled = [r for o in self.env.last_event.metadata["objects"] if o["objectType"] == otype
                      for r in self._recep_names(o.get("parentReceptacles")) if r in targets]
            order.sort(key=lambda r: -filled.count(r))
        used = set()
        for _ in range(n_objects):
            if self.done:
                break
            # 1. ambil objek yang belum berada di wadah tujuan
            meta = self.env.last_event.metadata
            cands = [o for o in meta["objects"] if o["objectType"] == otype and o["pickupable"]
                     and o["objectId"] not in used]
            if n_objects == 2 and order:
                cands = [o for o in cands if order[0] not in self._recep_names(o.get("parentReceptacles"))]
            got = False
            for o in cands:
                # wadah tercatat dulu, lalu wadah terdekat (objek bisa bergeser karena fisika)
                near = [r for r in self._receps_by_distance(o)[:6] if r not in self._object_receps(o)]
                for r in self._object_receps(o) + near:
                    if self._gt_goto(r) and self._gt_act("take %s from %s" % (name, r)):
                        got = True
                        break
                if got:
                    held = self.env.last_event.metadata["inventoryObjects"]
                    used.add(held[0]["objectId"] if held else o["objectId"])
                    break
            if not got:
                return "take_failed" if cands else "no_candidate"
            # 2. ubah status objek
            tool = {"pick_heat_then_place_in_recep": ("heat", "Microwave"),
                    "pick_cool_then_place_in_recep": ("cool", "Fridge"),
                    "pick_clean_then_place_in_recep": ("clean", "SinkBasin")}.get(ttype)
            if tool:
                verb, ttype_obj = tool
                tools = self._gt_receps_of_type(ttype_obj) or self._gt_receps_of_type("BathtubBasin")
                if not any(self._gt_goto(t) and self._gt_act("%s %s with %s" % (verb, name, t)) for t in tools):
                    return verb + "_failed"
            # 3. nyalakan lampu atau taruh di wadah tujuan
            if ttype == "look_at_obj_in_light":
                lamp = p["toggle_target"]
                lamp_receps = self.gt_object_locations(lamp) or self.receptacle_names()
                if not any(self._gt_goto(r) and self._gt_act("use %s" % lamp.lower()) for r in lamp_receps):
                    return "toggle_failed"
            else:
                if not any(self._gt_goto(r) and self._gt_act("move %s to %s" % (name, r)) for r in order):
                    return "place_failed"
                if n_objects == 2:   # objek kedua harus ke wadah yang sama dengan yang pertama
                    order = [self.controller.curr_recep]
        return "not_satisfied"

    def close(self):
        self.env.stop()
