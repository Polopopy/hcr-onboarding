import time
import mujoco as mj
import numpy as np
from mujoco import viewer

def dog_walk():
    model = mj.MjModel.from_xml_path("xml/dog.xml")
    data = mj.MjData(model)
    
    t = 0.0
    dt = model.opt.timestep
    
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            # data.ctrl[2] = 10 * np.sin(t)
            # data.ctrl[6] = 10 * np.sin(t)
            # data.ctrl[10] = 10 * np.sin(t)
            
            
            mj.mj_step(model, data)
            v.sync()
            
            t += dt
            time.sleep(dt)

if __name__ == "__main__":
    dog_walk()