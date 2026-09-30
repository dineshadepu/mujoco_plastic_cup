"""Experiment 2: the cup wall (a bottomless truncated cone of panels) floating in zero gravity.

Three stability checks, none with extra damping:
  a. rest    - released at the rest shape: nothing should move (curved rest shape is stress-free).
  b. ovalise - an n=2 radial velocity kick v_r = V0 cos(2 theta): the wall should ovalise back and forth
               at the n=2 frequency and return to its rest shape, and the energy must not grow.
  c. random  - random velocity on every panel (all modes, including the stiffest): energy must not grow.

Calibration check first: for a cylinder (r_bot = r_top) the model's n=2 ovalisation frequency must
match Rayleigh's inextensional ring mode  w^2 = D n^2 (n^2-1)^2 / (rho t R^4 (n^2+1)).  This tests the
circumferential bending springs derived from D. The cup is a cone (R = 25..36 mm), for which the ring
formula at mid radius is only a rough guide; the measured kick frequency is compared with the
model's own linearised modes.

  python 02_ring_zero_gravity.py       # writes out/ring_zero_gravity.png and out/ring_zero_gravity.mp4
"""
import os, time, numpy as np, mujoco, mediapy as media
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dshell as ds

N_THETA, N_Z = 24, 6
S_M = 0.1
V0 = 1.0                                   # m/s, peak radial kick speed for the ovalisation test
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def build():
    sh = ds.cone_ring(**ds.CUP, n_theta=N_THETA, n_z=N_Z, z0=0.05, s_m=S_M)
    m = mujoco.MjModel.from_xml_string(ds.scene_xml(sh, 1e-5, gravity=False, floor=False))
    d = mujoco.MjData(m)
    sh.bind(m, d)
    dt, w = sh.stable_dt(m, d)
    m.opt.timestep = dt
    return sh, m, d, w


def energy(sh, m, d):
    return ds.kinetic_energy(m, d) + sh.energy


def run(sh, m, d, T, cb=None):
    X0 = sh.panel_centers(d)
    log = []
    while d.time < T:
        mujoco.mj_step(m, d)
        rms, mx = ds.kabsch_residual(sh.panel_centers(d), X0)
        log.append([d.time, energy(sh, m, d), rms, mx] + ([] if cb is None else cb(d)))
    log = np.array(log)
    assert np.all(np.isfinite(log)), "blew up"
    return log


def cylinder_check(R=0.03):
    t, rho = ds.CUP["t"], ds.PP["rho"]
    sh = ds.cone_ring(R, R, ds.CUP["height"], t, N_THETA, N_Z, s_m=S_M)
    m = mujoco.MjModel.from_xml_string(ds.scene_xml(sh, 1e-5, gravity=False, floor=False))
    d = mujoco.MjData(m); sh.bind(m, d)
    w = sh.modes(m, d); w = w[w > 1e-3 * w[-1]]
    w_ray = np.sqrt(sh.D / (rho * t * R**4) * 36 / 5)
    print(f"calibration: cylinder R={R*1e3:.0f} mm, n=2 mode {w[0]/2/np.pi:.2f} Hz, "
          f"Rayleigh ring {w_ray/2/np.pi:.2f} Hz ({100*(w[0]/w_ray-1):+.2f} %)\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cylinder_check()
    sh, m, d, w = build()
    free = sh.bodies[sh.free]
    mass = m.body_mass[free].sum()
    D, t, rho = sh.D, ds.CUP["t"], ds.PP["rho"]
    R = 0.5 * (ds.CUP["r_bot"] + ds.CUP["r_top"])
    w_ring = np.sqrt(D / (rho * t * R**4) * 4 * 9 / 5)
    w_def = w[w > 1e-3 * w[-1]]                                  # drop the 6 rigid-body modes
    print(f"cup wall {N_THETA}x{N_Z} panels, t={t*1e3:.2f} mm, mass={mass*1e3:.2f} g, D={D:.4g} N m, "
          f"s_m={S_M}")
    print(f"stable dt = {m.opt.timestep:.2e} s (w_max = {w[-1]:.3g} rad/s)")
    print(f"lowest deformation modes [Hz]: {np.round(w_def[:6]/2/np.pi, 1)}")
    print(f"thin-ring n=2 estimate at R_mid (rough, cone): {w_ring/2/np.pi:.1f} Hz\n")
    q0 = d.qpos.copy()

    # a. rest
    loga = run(sh, m, d, 0.05)
    print(f"a. rest    : max shape change {loga[:, 3].max()*1e9:.3f} nm, max energy {loga[:, 1].max():.2e} J")

    # b. ovalisation kick (zero net momentum)
    mujoco.mj_resetData(m, d); d.qpos[:] = q0; mujoco.mj_forward(m, d)
    th = np.arctan2(d.xipos[free, 1], d.xipos[free, 0])
    radial = np.column_stack([np.cos(th), np.sin(th), 0 * th])
    for k, b in enumerate(free):
        a = m.jnt_dofadr[m.body_jntadr[b]]
        d.qvel[a:a + 3] = V0 * np.cos(2 * th[k]) * radial[k]
    mujoco.mj_forward(m, d)
    rim = np.array([m.body(b).name.endswith(f"_{N_Z-1}") for b in free])
    r0 = np.linalg.norm(d.xipos[free][rim, :2], axis=1); th_rim = th[rim]

    def oval(d):
        xy = d.xipos[free][rim, :2] - d.xipos[free, :2].mean(0)
        return [2 * np.mean((np.linalg.norm(xy, axis=1) - r0) * np.cos(2 * th_rim))]

    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.09]
    cam.distance, cam.elevation, cam.azimuth = 0.3, -60, 90
    frames, fps, slow, Tb = [], 30, 50, 0.3
    t0 = time.time()

    def cb(d):
        if len(frames) < d.time * fps * slow:
            r.update_scene(d, cam); frames.append(r.render())
        return oval(d)
    logb = run(sh, m, d, Tb, cb)
    wall = time.time() - t0
    tb, Eb, rmsb, mxb, a2 = logb.T
    zc = np.where(np.diff(np.sign(a2)) != 0)[0]
    f_meas = (len(zc) - 1) / (2 * (tb[zc[-1]] - tb[zc[0]]))
    E0 = Eb[0]
    print(f"b. ovalise : V0={V0} m/s, peak rim ovalisation {np.abs(a2).max()*1e3:.3f} mm, "
          f"freq {f_meas:.1f} Hz (linear modes: {w_def[0]/2/np.pi:.1f} Hz)")
    print(f"             energy: start {E0:.3e} J, max gain {100*(Eb.max()-E0)/E0:+.3f} %, "
          f"end {100*(Eb[-1]/E0-1):+.2f} %  ({Tb} s in {wall:.1f} s wall)")
    print(f"             centre-of-mass drift {1e6*np.linalg.norm(d.subtree_com[0]-[0,0,d.subtree_com[0][2]]):.2f} um")
    media.write_video(os.path.join(OUT, "ring_zero_gravity.mp4"), frames, fps=fps)

    # c. random kick on every DOF
    mujoco.mj_resetData(m, d); d.qpos[:] = q0
    rng = np.random.default_rng(0)
    d.qvel[:] = 0.05 * rng.standard_normal(m.nv)
    mujoco.mj_forward(m, d)
    logc = run(sh, m, d, 0.1)
    Ec = logc[:, 1]
    print(f"c. random  : energy start {Ec[0]:.3e} J, max gain {100*(Ec.max()-Ec[0])/Ec[0]:+.3f} %, "
          f"end {100*(Ec[-1]/Ec[0]-1):+.1f} %; shape change max {logc[:, 3].max()*1e3:.3f} mm, "
          f"end {logc[-1, 3]*1e3:.4f} mm")

    C1, C2, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a85"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(tb * 1e3, a2 * 1e3, color=C1, lw=1.2)
    ax[0].set(xlabel="time [ms]", ylabel="rim n=2 ovalisation [mm]",
              title=f"b. Ovalisation kick, {f_meas:.0f} Hz")
    ax[0].grid(alpha=0.2)
    ax[1].plot(tb * 1e3, Eb / Eb[0], color=C1, lw=1.5, label="b. ovalisation kick")
    ax[1].plot(logc[:, 0] * 1e3, Ec / Ec[0], color=C2, lw=1.5, label="c. random kick")
    ax[1].set(xlabel="time [ms]", ylabel="energy / initial energy", ylim=(0, 1.1),
              title="Kinetic + elastic energy (must not grow)")
    ax[1].legend(frameon=False); ax[1].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ring_zero_gravity.png"), dpi=130)
    print("\nwrote out/ring_zero_gravity.png, out/ring_zero_gravity.mp4")
