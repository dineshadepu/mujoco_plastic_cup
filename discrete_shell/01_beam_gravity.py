"""Experiment 1: a cantilever strip of rigid panels under gravity.

Checks that the panel + spring-damper model is stable and that its bending response matches
thin-plate theory (cylindrical bending, rigidity D per unit width):

  1. Static tip deflection (dynamic relaxation) for n = 5..40 panels, compared with
     - the discrete-model linear theory (sum of joint rotations; should match to ~0.1 %), and
     - Euler-Bernoulli with D:  delta = q L^4 / (8 D b)  (the discretisation error, O(h^2)).
  2. Linearised natural frequencies compared with beam theory f_k = (beta_k L)^2/(2 pi) sqrt(D/(rho t L^4)).
  3. Dynamic stability: gravity switched on suddenly from the straight pose, no extra damping.
     The tip should oscillate around the static value with an overshoot of about 2x, the
     mechanical energy must not grow, and nothing blows up.

  python 01_beam_gravity.py            # writes out/beam_gravity.png and out/beam_gravity.mp4
"""
import os, time, numpy as np, mujoco, mediapy as media
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dshell as ds

L, B, T = 0.10, 0.02, 0.5e-3            # length, width, thickness (m)
S_M = 0.1                                # membrane penalty scale (see dshell docstring)
NS = [5, 10, 20, 40]
N_DYN = 20                               # panels for the dynamic run and the video
Z = 0.1
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CLAMP = ('<body name="clamp" pos="-0.005 0 {z}"><geom type="box" size="0.005 {w} 0.004" '
         'contype="0" conaffinity="0" rgba=".4 .4 .4 1"/></body>').format(z=Z, w=0.6 * B)


def build(n, dt=1e-5):
    sh = ds.beam(L, B, T, n, z=Z, s_m=S_M)
    m = mujoco.MjModel.from_xml_string(ds.scene_xml(sh, dt, floor=False, extra_body=CLAMP))
    d = mujoco.MjData(m)
    sh.bind(m, d)
    dt, w = sh.stable_dt(m, d)
    m.opt.timestep = dt
    return sh, m, d, w


def tip_z(sh, d, n):
    b = sh.bodies[-1]                                   # last panel (body ids ascend with panel order)
    return (d.xpos[b] + d.xmat[b].reshape(3, 3) @ [L / n / 2, 0, 0])[2] - Z


def discrete_theory(sh, m, n):
    """Tip deflection of the discrete chain from linear statics (point loads at panel centroids)."""
    h, g = L / n, 9.81
    mp = m.body_mass[sh.bodies[-1]]
    xj = np.arange(n) * h                               # joint j at x = j h (j = 0 is the clamp)
    xc = (np.arange(n) + 0.5) * h
    M = np.array([np.sum(mp * g * (xc[xc > x] - x)) for x in xj])
    V = np.array([np.sum(mp * g * (xc > x)) for x in xj])
    return np.sum(M / sh.k_rot[:, 1] * (L - xj)) + np.sum(V / sh.k_lin[:, 2]), mp * g / h


def mech_energy(sh, m, d):
    pot = -np.sum(m.body_mass * (d.xipos @ m.opt.gravity))
    return ds.kinetic_energy(m, d) + sh.energy + pot


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    D = ds.plate_D(ds.PP["E"], T, ds.PP["nu"])
    q = ds.PP["rho"] * 9.81 * B * T
    eb = q * L**4 / (8 * D * B)
    f_eb = np.array([1.8751, 4.6941, 7.8548])**2 / (2 * np.pi) * np.sqrt(D / (ds.PP["rho"] * T * L**4))
    print(f"PP strip L={L*1e3:.0f} mm, b={B*1e3:.0f} mm, t={T*1e3:.2f} mm, E={ds.PP['E']:.2e} Pa, "
          f"D={D:.4g} N m, s_m={S_M}")
    print(f"Euler-Bernoulli (D) tip deflection = {eb*1e3:.4f} mm; bending modes {np.round(f_eb, 2)} Hz\n")
    print(f"{'n':>3} {'dt [s]':>9} {'sim [mm]':>9} {'discrete':>9} {'err':>7} {'EB err':>7}   f1 f2 [Hz]")

    rows = []
    for n in NS:
        sh, m, d, w = build(n)
        sh.relax = 2 * 2 * np.pi * f_eb[0]                  # ~critical damping of mode 1
        while d.time < 0.25:
            mujoco.mj_step(m, d)
        sim = -tip_z(sh, d, n)
        th, _ = discrete_theory(sh, m, n)
        rows.append((n, sim, th))
        print(f"{n:>3} {m.opt.timestep:9.2e} {sim*1e3:9.4f} {th*1e3:9.4f} {100*(sim/th-1):6.2f}% "
              f"{100*(sim/eb-1):6.2f}%   {w[0]/2/np.pi:.2f} {w[1]/2/np.pi:.2f}")

    # ---- dynamic run: sudden gravity, no extra damping
    n = N_DYN
    sh, m, d, w = build(n)
    th, _ = discrete_theory(sh, m, n)
    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [L / 2, 0, Z]
    cam.distance, cam.elevation, cam.azimuth = 0.16, 0, 90
    frames, log, fps, Tend, slow = [], [], 30, 0.6, 10       # video slowed down 10x
    E0 = None; t0 = time.time()
    while d.time < Tend:
        mujoco.mj_step(m, d)
        if E0 is None: E0 = mech_energy(sh, m, d)
        log.append((d.time, -tip_z(sh, d, n), mech_energy(sh, m, d) - E0))
        if len(frames) < d.time * fps * slow:
            r.update_scene(d, cam); frames.append(r.render())
    wall = time.time() - t0
    log = np.array(log)
    tt, tip, dE = log.T
    assert np.all(np.isfinite(log)), "blew up"
    zc = np.where(np.diff(np.sign(tip - th)) != 0)[0]
    f_meas = (len(zc) - 1) / (2 * (tt[zc[-1]] - tt[zc[0]])) if len(zc) > 2 else np.nan
    print(f"\nDynamic run (n={n}, dt={m.opt.timestep:.2e} s, {Tend} s in {wall:.1f} s wall):")
    print(f"  peak tip = {tip.max()*1e3:.4f} mm = {tip.max()/th:.3f} x static (linear theory: 2x)")
    zc2 = zc[: 1 + 2 * ((len(zc) - 1) // 2)]                  # crossings spanning whole periods
    mean = tip[zc2[0]:zc2[-1]].mean()
    print(f"  mean tip (whole periods) = {mean*1e3:.4f} mm = {mean/th:.3f} x static")
    print(f"  oscillation frequency = {f_meas:.2f} Hz (beam theory f1 = {f_eb[0]:.2f} Hz)")
    print(f"  max energy gain = {dE.max():.2e} J  (energy scale {m.body_mass[sh.bodies[sh.free]].sum()*9.81*th:.2e} J)")
    media.write_video(os.path.join(OUT, "beam_gravity.mp4"), frames, fps=fps)

    # ---- figure
    C1, C2, C3, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(tt * 1e3, tip * 1e3, color=C1, lw=1.5, label=f"simulation, n={n}")
    ax[0].axhline(th * 1e3, color=C2, lw=2, ls="--", label="static (discrete theory)")
    ax[0].axhline(2 * th * 1e3, color=MUTED, lw=1, ls=":", label="2 x static")
    ax[0].set(xlabel="time [ms]", ylabel="tip deflection [mm]",
              title="Sudden gravity, no extra damping")
    ax[0].set_ylim(-0.3, 8.5); ax[0].legend(frameon=False, loc="upper center", ncol=3, fontsize=8); ax[0].grid(alpha=0.2)
    ns = np.array([r_[0] for r_ in rows]); sim = np.array([r_[1] for r_ in rows])
    thr = np.array([r_[2] for r_ in rows])
    ax[1].plot(ns, sim * 1e3, "o", color=C1, ms=8, label="simulation (static)")
    ax[1].plot(ns, thr * 1e3, "-", color=C3, lw=2, label="discrete theory")
    ax[1].axhline(eb * 1e3, color=C2, lw=2, ls="--", label="Euler-Bernoulli, D")
    ax[1].set(xscale="log", xticks=ns, xticklabels=ns, xlabel="panels n",
              ylabel="static tip deflection [mm]", title="Convergence to thin-plate theory")
    ax[1].minorticks_off(); ax[1].legend(frameon=False); ax[1].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "beam_gravity.png"), dpi=130)
    print(f"\nwrote out/beam_gravity.png, out/beam_gravity.mp4")
