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

REWARD_CONFIG = os.path.join(alfworld.agents.__path__[0], "config", "rewards.json")
NOTHING = "Nothing happens."

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
        self.rng = random.Random(seed)
        random.seed(seed)  # template goal ALFWorld memakai RNG global

        self.controller = None
        self.expert = HandCodedThorAgent(self.env, max_steps=expert_max_steps)
        self.scene_num = None
        self.task_idx = -1
        self.trace = []          # log per langkah untuk metrik
        self.task_log = []       # log per tugas

    # ------------------------------------------------------------------ rangkaian tugas
    def start_chain(self, task_file):
        """Tugas pertama: muat ruangan seperti ALFWorld biasa."""
        traj = _load_traj(task_file)
        scene = traj["scene"]
        self.scene_num = scene["scene_num"]
        self.env.reset("FloorPlan%d" % self.scene_num)
        self.env.restore_scene(scene["object_poses"], scene["object_toggles"], scene["dirty_and_empty"])
        self.env.step(dict(scene["init_action"]))
        self.controller = OracleAgent(self.env, traj, os.path.dirname(task_file),
                                      load_receps=True, debug=False, goal_desc_human_anns_prob=0.0)
        self.task_idx = -1
        self.trace, self.task_log = [], []
        return self._begin_task(task_file, traj)

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
        }
        self.task_log.append(self.task_info)

        # expert untuk validasi (11)
        self.expert.reset(task_file)
        self._expert_prev = ""

        obs = self._intro_obs()
        self.last_obs = obs
        return obs, self._info(won=bool(won0))

    def _intro_obs(self):
        receps = ", ".join("a %s" % r for r in self.receptacle_names())
        where = ("You are at the %s." % self.controller.curr_recep
                 if self.controller.curr_recep != "nothing" else "You are in the middle of a room.")
        return ("%s Looking quickly around the room, you see %s.\n\nYour task is to: %s"
                % (where, receps, self.task_desc))

    def _release_held_object(self):
        """Taruh objek yang masih dipegang ke wadah tempat agen berada sebelum tugas baru."""
        if not self.controller.inventory:
            return None
        held = self.controller.inventory[0]
        recep = self.controller.curr_recep
        if recep != "nothing":
            self.controller.step("move %s to %s" % (held, recep))
        if self.controller.inventory:   # gagal ditaruh, jatuhkan saja
            ev = self.env.step(dict(action="DropHandObject"))
            if ev.metadata["lastActionSuccess"]:
                self.controller.inventory.pop()
        return held

    # ------------------------------------------------------------------ aksi agen
    def step(self, action):
        if self.done:
            return self.last_obs, True, self._info()
        cmd = self.resolve(action)                                     # (8)
        raw = self.controller.step(cmd) if cmd else NOTHING
        return self._after_step(action, cmd, raw)

    def step_raw(self, cmd):
        """Perintah format asli ALFWorld (bernomor). Hanya untuk expert/validasi."""
        if self.done:
            return self.last_obs, True, self._info()
        raw = self.controller.step(cmd)
        return self._after_step(cmd, cmd, raw)

    def _after_step(self, action, cmd, raw):
        self.steps += 1
        self._raw_feedback = raw
        obs = self.sanitize(raw, cmd)                                  # (6)
        won = bool(self.env.get_goal_satisfied())
        self.done = won or self.steps >= self.max_steps                # (3)
        self.last_obs = obs
        self.trace.append({
            "task_idx": self.task_idx, "step": self.steps, "action": action, "command": cmd,
            "obs": obs, "success": raw != NOTHING, "location": self.controller.curr_recep,
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
                "won_initial": self.task_info["won_initial"], "gc_initial": self.task_info["gc_initial"]}

    # ------------------------------------------------------------------ resolusi perintah (8)
    def receptacle_names(self):
        return [r["num_id"] for r in self.controller.receptacles.values()]

    def _held(self):
        return self.controller.inventory[0] if self.controller.inventory else None

    def _visible_of_type(self, otype):
        inv = set(self.controller.inventory)
        for n in getattr(self.controller, "visible_objects", []) or []:
            if n not in inv and _obj_type(n) == otype:
                return n
        return None

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
            obj = self._visible_of_type(_strip_num(m.group(1)))
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
    def relocate(self, object_type, source_task_files, min_distance=0.3):
        """Pindahkan satu objek berjenis object_type ke posisinya pada trial lain di ruangan yang sama.
        source_task_files: satu path atau list path traj_data.json, dicoba berurutan.
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
                     and o["objectId"] not in held_ids and o["name"] in src_pose]
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
                if ev.metadata["lastActionSuccess"] and after and set(after) != set(before):
                    result = {"ok": True, "object_type": object_type, "object_name": o["name"],
                              "before": before, "after": after, "source": src_file, "attempts": attempts}
                    self.task_log.append({"relocation": result, "after_task_idx": self.task_idx})
                    return result
                self._set_poses(snapshot)   # tidak valid, kembalikan
        self.task_log.append({"relocation": {"ok": False, "object_type": object_type, "attempts": attempts},
                              "after_task_idx": self.task_idx})
        return None

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
        """objectId AI2-THOR bisa berubah setelah SetObjectPoses. Samakan kembali catatan controller."""
        new_id_by_name = {o["name"]: o["objectId"] for o in self.env.last_event.metadata["objects"]}
        mapping = {old: new_id_by_name[n] for n, old in old_id_by_name.items()
                   if n in new_id_by_name and new_id_by_name[n] != old}
        if not mapping:
            return 0
        for attr in ("objects", "receptacles"):
            remapped = {}
            for oid, rec in getattr(self.controller, attr).items():
                nid = mapping.get(oid, oid)
                rec["object_id"] = nid
                remapped[nid] = rec
            setattr(self.controller, attr, remapped)
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
                    locs.update(self._recep_names(o.get("parentReceptacles")))
        return sorted(locs)

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

    def close(self):
        self.env.stop()
