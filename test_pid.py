import numpy as np
import sys
sys.path.append(r"d:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile")
from Controller.F16PIDController2_1 import F16PIDController
from math import pi

controller = F16PIDController()

def get_action(delta_head_deg):
    state = np.zeros(14)
    state[0] = 5000/5000
    state[13] = 5000/5000
    state[1] = delta_head_deg * pi / 180
    state[2] = 200/340
    state[4] = 200/340
    act = controller.flight_output(state)
    return act[0] # aileron

print("Target Left (-30 deg), aileron:", get_action(-30))
print("Target Right (+30 deg), aileron:", get_action(30))
