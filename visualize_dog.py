import time
import mujoco as mj
import numpy as np
from mujoco import viewer

ACT_TO_JOINT = {
    "fl_upper_leg_motor": "fl_upper_pitch",
    "fl_knee_motor":      "fl_knee",

    "fr_upper_leg_motor": "fr_upper_pitch",
    "fr_knee_motor":      "fr_knee",

    "bl_upper_leg_motor": "bl_upper_pitch",
    "bl_knee_motor":      "bl_knee",

    "br_upper_leg_motor": "br_upper_pitch",
    "br_knee_motor":      "br_knee",
}

ACTUATOR_CTRL_IDX = {
    "fl_hip_motor": 0, "fl_upper_leg_motor": 1, "fl_knee_motor": 2, "fl_ankle_motor": 3,
    "fr_hip_motor": 4, "fr_upper_leg_motor": 5, "fr_knee_motor": 6, "fr_ankle_motor": 7,
    "bl_hip_motor": 8, "bl_upper_leg_motor": 9, "bl_knee_motor": 10, "bl_ankle_motor": 11,
    "br_hip_motor": 12, "br_upper_leg_motor": 13, "br_knee_motor": 14, "br_ankle_motor": 15,
}

def angle_pd(data, qpos_idx, qvel_idx, q_des, kp=5.0, kd=0.2):
    q  = data.qpos[qpos_idx]
    qd = data.qvel[qvel_idx]
    return kp*(q_des - q) - kd*qd

LEGS    = ["fl","fr","bl","br"]

# Walking
OFFSETS = {"bl":0.0, "fr":0.0, "fl":0.5, "br":0.5}
T_GAIT  = 0.35

POSE_A = (-1.05, 2.65)
POSE_B = (-1.65, 1.5)
POSE_C = (-1.65, 3.1)

PHASES = (0.7, 0.1, 0.2)

# Running
# OFFSETS = {"bl":0.5, "fr":0.0, "fl":0.0, "br":0.5}
# T_GAIT  = 0.5

# POSE_A = (0.8, 0.4)
# POSE_B = (-0.8, 2.6)
# POSE_C = (-1.1, -0.1)

# PHASES = (0.5, 0.2, 0.3)

def _lerp(a, b, t): return a + (b - a) * t
def _blend(p0, p1, t): return (_lerp(p0[0], p1[0], t), _lerp(p0[1], p1[1], t))

def leg_targets(p):
    a, b, c = PHASES
    s = a + b + c
    if abs(s - 1.0) > 1e-9:
        a, b, c = a/s, b/s, c/s

    if p < a:
        t = p / a
        return _blend(POSE_A, POSE_B, t)
    elif p < a + b:
        t = (p - a) / b
        return _blend(POSE_B, POSE_C, t)
    else:
        t = (p - a - b) / c
        return _blend(POSE_C, POSE_A, t)

def dog_walk():
    model = mj.MjModel.from_xml_path("xml/dog.xml")
    data  = mj.MjData(model)
    dt = model.opt.timestep

    joint_addr = {}
    for jname in set(ACT_TO_JOINT.values()):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        joint_addr[jname] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))

    phi = 0.0

    with mj.viewer.launch_passive(model, data) as v:
        while v.is_running():
            phi = (phi + dt / T_GAIT) % 1.0

            targets = {}
            for leg in LEGS:
                lp = (phi + OFFSETS[leg]) % 1.0
                upper, knee = leg_targets(lp)
                targets[f"{leg}_upper_pitch"] = upper
                targets[f"{leg}_knee"]        = knee

            for aname, jname in ACT_TO_JOINT.items():
                aid = ACTUATOR_CTRL_IDX[aname]
                qpos_idx, qvel_idx = joint_addr[jname]
                data.ctrl[aid] = angle_pd(data, qpos_idx, qvel_idx, targets[jname])

            mj.mj_step(model, data)
            v.sync()
            time.sleep(dt)

if __name__ == "__main__":
    dog_walk()
