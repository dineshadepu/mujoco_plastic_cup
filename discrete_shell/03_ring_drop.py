"""Experiment 3: the cup wall (bottomless cone of panels) dropped onto the floor under gravity.

The wall falls base-down from DROP_H onto a plane with MuJoCo's soft contact (elliptic friction
cone, mu = 0.5, solref 0.002 s critically damped: dshell.CONTACT_SOFT, as in the rigid pipeline).
Panels don't collide with each other.

Checks: no blow-up, it lands and comes to rest standing upright, the impact dent recovers
(elastic), the final shape change under self-weight is tiny, and the energy drops at impact and
settles at m g z_com. The small rise right after impact is the rebound from contact penetration
(MuJoCo's soft contact stores energy there, which this energy measure doesn't count).

  python 03_ring_drop.py               # writes out/ring_drop.png and out/ring_drop.mp4
"""
import os, time, numpy as np, mujoco, mediapy as media
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dshell as ds

N_THETA, N_Z = 24, 6
S_M = 0.1
DROP_H = 0.02                   # m, gap between the wall's bottom edge and the floor at release
T_END = 0.5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    sh = ds.cone_ring(**ds.CUP, n_theta=N_THETA, n_z=N_Z, z0=DROP_H + ds.CUP["t"], s_m=S_M)
    m = mujoco.MjModel.from_xml_string(ds.scene_xml(sh, 1e-5, gravity=True, floor=True))
    d = mujoco.MjData(m)
    sh.bind(m, d)
    dt, w = sh.stable_dt(m, d)
    m.opt.timestep = dt
    free = sh.bodies[sh.free]
    names = [m.body(b).name for b in free]
    base = np.array([n.endswith("_0") for n in names]); rim = np.array([n.endswith(f"_{N_Z-1}") for n in names])
    mass = m.body_mass[free].sum()
    print(f"cup wall {N_THETA}x{N_Z} panels, t={ds.CUP['t']*1e3:.2f} mm, mass={mass*1e3:.2f} g, "
          f"dt={dt:.2e} s, drop {DROP_H*1e3:.0f} mm")

    mujoco.mj_forward(m, d)
    X0 = sh.panel_centers(d)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def mech_energy():
        return ds.kinetic_energy(m, d) + sh.energy - mass * m.opt.gravity[2] * d.subtree_com[0][2]

    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.05]
    cam.distance, cam.elevation, cam.azimuth = 0.35, -20, 90
    frames, log, fps, slow = [], [], 30, 5
    t0 = time.time()
    while d.time < T_END:
        mujoco.mj_step(m, d)
        X = sh.panel_centers(d)
        rms, mx = ds.kabsch_residual(X, X0)
        axis = X[rim].mean(0) - X[base].mean(0)
        tilt = np.degrees(np.arccos(axis[2] / np.linalg.norm(axis)))
        ncon = sum(1 for c in d.contact[:d.ncon] if floor in (c.geom1, c.geom2))
        log.append([d.time, d.subtree_com[0][2], rms, mx, tilt, mech_energy(), ncon])
        if len(frames) < d.time * fps * slow:
            r.update_scene(d, cam); frames.append(r.render())
    wall = time.time() - t0
    log = np.array(log)
    assert np.all(np.isfinite(log)), "blew up"
    tt, zc, rms, mx, tilt, E, ncon = log.T
    i_hit = np.argmax(ncon > 0)
    E_after = E[i_hit:]
    print(f"simulated {T_END} s in {wall:.1f} s wall")
    print(f"impact at t = {tt[i_hit]*1e3:.1f} ms; max shape change during impact {mx.max()*1e3:.3f} mm "
          f"(rms {rms.max()*1e3:.3f} mm)")
    print(f"final: CoM z = {zc[-1]*1e3:.2f} mm, tilt {tilt[-1]:.3f} deg, {int(ncon[-1])} floor contacts, "
          f"shape change max {mx[-1]*1e3:.4f} mm (rms {rms[-1]*1e3:.4f} mm)")
    print(f"final max |qvel| = {np.abs(d.qvel).max():.2e}; energy after impact: start {E_after[0]:.3e} J, "
          f"end {E_after[-1]:.3e} J, largest step-to-step rise {np.max(np.diff(E_after)):.2e} J")
    media.write_video(os.path.join(OUT, "ring_drop.mp4"), frames, fps=fps)

    C1, C2, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a85"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(tt * 1e3, zc * 1e3, color=C1, lw=1.5)
    ax[0].set(xlabel="time [ms]", ylabel="centre of mass height [mm]", title="Fall and landing")
    ax[1].plot(tt * 1e3, mx * 1e3, color=C1, lw=1.2, label="max over panels")
    ax[1].plot(tt * 1e3, rms * 1e3, color=C2, lw=1.2, label="rms")
    ax[1].set(xlabel="time [ms]", ylabel="shape change [mm]",
              title="Deformation (rigid motion removed)")
    ax[1].legend(frameon=False)
    ax[2].plot(tt * 1e3, E * 1e3, color=C1, lw=1.5)
    ax[2].axvline(tt[i_hit] * 1e3, color=MUTED, lw=1, ls=":")
    ax[2].set(xlabel="time [ms]", ylabel="kinetic + elastic + gravity [mJ]",
              title="Mechanical energy (impact at dotted line)")
    for a in ax: a.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ring_drop.png"), dpi=130)
    print("\nwrote out/ring_drop.png, out/ring_drop.mp4")
