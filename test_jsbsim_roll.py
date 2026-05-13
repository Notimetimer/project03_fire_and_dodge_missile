import sys
sys.path.append(r"d:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile")
from Envs.UAVmodel6d import UAVModel

uav = UAVModel(dt=0.02)
uav.reset(psi0=0)

# Simulate 5 seconds with 0 control inputs
for _ in range(250):
    uav.move(target_height=0, delta_heading=0, target_speed=0.5, e2e=True, rudder=0)

print(f"Final roll angle (phi): {uav.phi * 180 / 3.14159:.2f} degrees")
print(f"Final heading (psi): {uav.psi * 180 / 3.14159:.2f} degrees")
