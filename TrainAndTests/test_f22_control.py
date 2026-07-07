"""F22控制链路自动化测试：加载模型，给副翼/升降舵指令，打印完整LQR链路"""
import jsbsim
import os, sys

print(f"Python解释器: {sys.executable}")
print(f"jsbsim包路径: {jsbsim.__file__}")
xml_path = os.path.join(os.path.dirname(jsbsim.__file__), "aircraft", "f22", "f22.xml")
with open(xml_path, encoding="utf-8") as f:
    content = f.read()
print(f"实际加载的XML: {xml_path}")
print(f"XML检查: 残留lag配置数={content.count('<lag>')}, pos-rad输出数={content.count('<output>fcs/') and sum(1 for l in content.splitlines() if '<output>' in l and 'pos-rad' in l and not l.strip().startswith('<!--'))}")

dt = 0.02
sim = jsbsim.FGFDMExec(None, None)
sim.set_debug_level(0)
sim.set_dt(dt)
sim.load_model("f22")

sim["ic/vt-kts"] = 250 * 1.94384
sim["ic/long-gc-deg"] = 116.0
sim["ic/lat-gc-deg"] = 39.0
sim["ic/h-sl-ft"] = 3000 * 3.28084
sim["ic/psi-true-deg"] = -60
sim["ic/phi-deg"] = 0
sim["ic/theta-deg"] = 2
sim["ic/alpha-deg"] = 2
sim["ic/beta-deg"] = 0
sim["ic/gamma-deg"] = 0
sim.run_ic()
sim["gear/gear-cmd-norm"] = 0.0
sim.set_property_value('propulsion/set-running', -1)

print(f"初始: vt={sim['velocities/vt-fps']*0.3048:.1f} m/s, vc={sim['velocities/vc-kts']:.0f} kts, "
      f"h={sim['position/h-sl-ft']*0.3048:.0f} m, alpha={sim['aero/alpha-deg']:.1f} deg")

for step in range(int(5.0 / dt)):
    sim["fcs/throttle-cmd-norm[0]"] = 0.8
    sim["fcs/throttle-cmd-norm[1]"] = 0.8
    sim["fcs/aileron-cmd-norm"] = 1.0  # 持续满舵右滚
    sim["fcs/elevator-cmd-norm"] = 0.0
    sim["fcs/rudder-cmd-norm"] = 0.0
    if not sim.run():
        print(f"step {step} 仿真失败")
        break
    if step % 25 == 0:
        t = step * dt
        print(f"t={t:4.1f}s | scale={sim['fcs/roll-reg-scale']:+.3f} -> act={sim['fcs/aileron-act']:+.3f} "
              f"| ail_l={sim['fcs/left-aileron-pos-norm']:+.3f} ail_r={sim['fcs/right-aileron-pos-norm']:+.3f} "
              f"| p={sim['velocities/p-aero-rad_sec']:+.3f} rad/s, phi={sim['attitude/phi-deg']:+.1f} deg, "
              f"vc={sim['velocities/vc-kts']:.0f} kts, alpha={sim['aero/alpha-deg']:+.1f}")

print("\n结论: 若act跟随scale且phi(滚转角)变化 => 控制已生效")
