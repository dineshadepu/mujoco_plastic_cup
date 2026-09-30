"""Watch any discrete-shell experiment live in the MuJoCo viewer.

The stock viewer (python -m mujoco.viewer --mjcf=...) does not run the shell's spring callback, so
the panels would fall apart. This script builds the experiment, installs the callback, and steps it.

  ../.venv/bin/mjpython view.py beam           # 01: cantilever strip, sudden gravity
  ../.venv/bin/mjpython view.py ring           # 02: wall in zero g, ovalisation kick
  ../.venv/bin/mjpython view.py drop           # 03: wall dropped on the floor
  ../.venv/bin/mjpython view.py cup 0.5        # 04: hung cup, cube of 0.5 kg (0.01 / 0.1 / 0.5)
  ../.venv/bin/mjpython view.py cup 0.5 --ms 0.2    # simulated milliseconds per frame (default 1)
  ../.venv/bin/mjpython view.py panda 0.5 --ms 5     # 05: the robot trial with a 0.5 kg cube

On macOS the viewer needs `mjpython` (installed with mujoco in the venv), not `python`.
The time steps are micro-seconds, so this runs in slow motion (the cup at about 1/200 real time).
Double-click a body to select it. Mouse perturbation of panels has no effect: the
callback overwrites their applied forces every step.
"""
import sys, time, importlib, numpy as np, mujoco, mujoco.viewer
import dshell as ds


def build(which, mass):
    if which == "beam":
        sh, m, d, w = importlib.import_module("01_beam_gravity").build(20)
    elif which == "ring":
        mod = importlib.import_module("02_ring_zero_gravity")
        sh, m, d, w = mod.build()
        mujoco.mj_forward(m, d)
        for b in sh.bodies[sh.free]:                       # n=2 ovalisation kick, as in experiment 2
            th = np.arctan2(d.xipos[b, 1], d.xipos[b, 0])
            a = m.jnt_dofadr[m.body_jntadr[b]]
            d.qvel[a:a + 3] = mod.V0 * np.cos(2 * th) * np.array([np.cos(th), np.sin(th), 0])
    elif which == "drop":
        mod = importlib.import_module("03_ring_drop")
        sh = ds.cone_ring(**ds.CUP, n_theta=mod.N_THETA, n_z=mod.N_Z, z0=mod.DROP_H + ds.CUP["t"], s_m=mod.S_M)
        m = mujoco.MjModel.from_xml_string(ds.scene_xml(sh, 1e-5, gravity=True, floor=True))
        d = mujoco.MjData(m); sh.bind(m, d)
        m.opt.timestep, w = sh.stable_dt(m, d)
    elif which == "cup":
        sh, m, d, w = importlib.import_module("04_cup_hang_cube").build_cup(mass)
    elif which == "panda":
        P = importlib.import_module("05_panda_cup_shell")
        s, m, d, sh = P.build(mass)
        arm_act = [m.actuator(f"actuator{i}").id for i in range(1, 8)]
        arm_q = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)]
        fj = [m.jnt_qposadr[m.joint(f"finger_joint{i}").id] for i in (1, 2)]
        ts, qs, q_pre = P.plan(m, m.key("home").qpos[:7].copy())
        d.qpos[arm_q] = q_pre; d.qpos[fj] = 0.04
        g8 = m.actuator("actuator8").id

        def control(m, d):                                 # same hard-coded motion as the batch run
            d.ctrl[arm_act] = [np.interp(d.time, ts, qs[:, j]) for j in range(7)]
            d.ctrl[g8] = P.grip_ctrl(d.time, d.qpos[fj[0]], P.F_GRIP)
        mujoco.set_mjcb_control(control)
    else:
        sys.exit(__doc__)
    return sh, m, d


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    which = args[0]
    mass = float(args[1]) if len(args) > 1 else 0.1
    ms = float(sys.argv[sys.argv.index("--ms") + 1]) if "--ms" in sys.argv else 1.0
    print(f"building '{which}' (computing the stable time step takes a few seconds)...")
    sh, m, d = build(which, mass)
    n = max(1, int(round(ms * 1e-3 / m.opt.timestep)))
    print(f"dt = {m.opt.timestep:.2e} s, {n} steps per frame = {ms} ms simulated per frame")
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            with v.lock():
                for _ in range(n):
                    mujoco.mj_step(m, d)
            v.sync()
            print(f"\rt = {d.time*1e3:7.1f} ms   ({time.time()-t0:.2f} s per frame)", end="")
