import time
import mujoco as mj
import numpy as np
from mujoco import viewer

ACT_TO_JOINT = {
    "fl_hip_motor":       "fl_hip_pitch",
    "fl_upper_leg_motor": "fl_upper_pitch",
    "fl_knee_motor":      "fl_knee",

    "fr_hip_motor":       "fr_hip_pitch",
    "fr_upper_leg_motor": "fr_upper_pitch",
    "fr_knee_motor":      "fr_knee",

    "bl_hip_motor":       "bl_hip_pitch",
    "bl_upper_leg_motor": "bl_upper_pitch",
    "bl_knee_motor":      "bl_knee",

    "br_hip_motor":       "br_hip_pitch",
    "br_upper_leg_motor": "br_upper_pitch",
    "br_knee_motor":      "br_knee",
}

ACTUATOR_CTRL_IDX = {
    "fl_hip_motor":       0, "fl_upper_leg_motor": 1, "fl_knee_motor": 2, "fl_ankle_motor": 3,
    "fr_hip_motor":       4, "fr_upper_leg_motor": 5, "fr_knee_motor": 6, "fr_ankle_motor": 7,
    "bl_hip_motor":       8, "bl_upper_leg_motor": 9, "bl_knee_motor": 10, "bl_ankle_motor": 11,
    "br_hip_motor":      12, "br_upper_leg_motor": 13, "br_knee_motor": 14, "br_ankle_motor": 15,
}

def angle_pd(data, qpos_idx, qvel_idx, q_des, kp=8.0, kd=0.2):
    q  = data.qpos[qpos_idx]
    qd = data.qvel[qvel_idx]
    return kp*(q_des - q) - kd*qd

LEGS    = ["fl","fr","bl","br"]
OFFSETS = {"fl":0.0, "br":0.0, "fr":0.5, "bl":0.5}
T_GAIT  = 0.7
DUTY    = 0.6

BASE_UPPER = -1.0
BASE_KNEE  = 2.0

HIP_FWD = +0.25
HIP_BCK = -0.25
UPPER_SW = +0.40
KNEE_SW  = -0.60

def leg_targets(local_phase):
    if local_phase < DUTY:
        s = local_phase / DUTY
        hip   = HIP_FWD + (HIP_BCK - HIP_FWD)*s
        upper = BASE_UPPER
        knee  = BASE_KNEE
    else:
        s = (local_phase - DUTY) / (1.0 - DUTY)
        hip   = HIP_BCK + (HIP_FWD - HIP_BCK)*s
        lift  = np.sin(np.pi*s)
        upper = BASE_UPPER + UPPER_SW*lift
        knee  = BASE_KNEE  + KNEE_SW*lift
    return hip, upper, knee

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
                hip, upper, knee = leg_targets(lp)
                targets[f"{leg}_hip_pitch"]   = hip
                targets[f"{leg}_upper_pitch"] = upper
                targets[f"{leg}_knee"]        = knee

            for aname, jname in ACT_TO_JOINT.items():
                aid = ACTUATOR_CTRL_IDX[aname]
                qpos_idx, qvel_idx = joint_addr[jname]
                u = angle_pd(data, qpos_idx, qvel_idx, targets[jname])
                data.ctrl[aid] = u

            mj.mj_step(model, data)
            v.sync()
            time.sleep(dt)

if __name__ == "__main__":
    dog_walk()
