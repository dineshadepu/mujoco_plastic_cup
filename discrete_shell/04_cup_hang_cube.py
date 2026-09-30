"""Experiment 4: full cup (wall + heel + bottom) hung by its rim, a cube released onto the bottom.

The cup is a shell of revolution (dshell.cup_profile): a flat bottom (centre panel + rings), one heel
row at an intermediate angle (a 3 mm fillet cut into one chamfer) and the conical wall. The rim edge
is clamped to the world. A cube (25 mm, as in the rigid pipeline) is released 2 mm above the bottom
for each assignment mass: 10, 100 and 500 g.

The cube uses dshell.CONTACT_HARD: with MuJoCo's default soft contact the cube corners sank ~1 mm
into the 50 ug panels, which moved the load to the bottom centre and doubled the reported sag.

Phases: 0 - T_FREE s natural dynamics (impact, bounce, vibration), then T_RELAX s of dynamic
relaxation (extra damping on panels and cube) to reach the static equilibrium that is reported.

s_m = 1 here (full membrane stiffness): the bottom sags more than its thickness under the heavier
cubes, so membrane stretching carries part of the load. With the s_m = 0.1 penalty of experiments
1-3 the 500 g sag came out 60 % too large.

Before the cup runs, the bottom mesh is checked against the clamped circular plate under its own
weight, w0 = q a^4 / (64 D).

  python 04_cup_hang_cube.py          # all three masses in parallel, then the figure
  python 04_cup_hang_cube.py 0.1      # one mass (kg): out/cup_hang_cube_100g.mp4 + .npz
"""
import os, sys, time, numpy as np, mujoco, mediapy as media
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multiprocessing import Pool
import dshell as ds

MASSES = [0.01, 0.1, 0.5]
S_M = 1.0
CUBE_HALF, GAP = 0.0125, 0.002
Z0 = 0.10                                   # bottom mid-surface height (hanging, no floor)
T_FREE, T_RELAX = 0.25, 0.15
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
MESH = {k: v for k, v in ds.CUP_MESH.items() if k != "n_theta"}
N_THETA, NB = ds.CUP_MESH["n_theta"], ds.CUP_MESH["n_bottom"]


def build_cup(cube_mass):
    sh = ds.revolved(ds.cup_profile(**ds.CUP, **MESH), ds.CUP["t"], N_THETA, z0=Z0,
                     center=True, clamp_top=True, s_m=S_M)
    for p in sh.panels:                     # bottom + heel opaque, wall see-through
        j = -1 if p["name"] == "c" else int(p["name"].split("_")[1])
        p["rgba"] = "0.2 0.45 0.85 1" if j <= NB else "0.3 0.6 0.9 0.2"
    zc = Z0 + ds.CUP["t"] / 2 + CUBE_HALF + GAP
    cube = (f'<body name="cube" pos="0 0 {zc}"><freejoint/><geom type="box" size="{CUBE_HALF} {CUBE_HALF} '
            f'{CUBE_HALF}" mass="{cube_mass}" {ds.CONTACT_HARD} rgba=".9 .3 .2 1"/></body>')
    m = mujoco.MjModel.from_xml_string(ds.scene_xml(sh, 1e-5, floor=False, extra_body=cube))
    d = mujoco.MjData(m)
    sh.bind(m, d)
    dt, w = sh.stable_dt(m, d)
    m.opt.timestep = dt
    return sh, m, d, w[w > 1e-3 * w[-1]]              # drop the cube's free rigid-body modes


def plate_check():
    """Bottom mesh alone (centre panel + NB rings, 24 around), edge clamped, under self-weight."""
    a = ds.CUP["r_bot"]
    prof = np.column_stack([np.linspace(a / (NB + 1), a, NB + 1), np.zeros(NB + 1)])
    sh = ds.revolved(prof, ds.CUP["t"], N_THETA, z0=Z0, center=True, clamp_top=True, s_m=S_M)
    m = mujoco.MjModel.from_xml_string(ds.scene_xml(sh, 1e-5, floor=False)); d = mujoco.MjData(m)
    sh.bind(m, d); dt, w = sh.stable_dt(m, d); m.opt.timestep = dt
    c = m.body("c").id; mujoco.mj_forward(m, d); z0 = d.xipos[c][2]
    sh.relax = 2 * w[0]
    while d.time < 20 / w[0]:
        mujoco.mj_step(m, d)
    th = ds.PP["rho"] * 9.81 * ds.CUP["t"] * a**4 / (64 * sh.D)
    sim = z0 - d.xipos[c][2]
    print(f"calibration: clamped bottom disk a={a*1e3:.0f} mm under self-weight: centre {sim*1e6:.3f} um, "
          f"plate theory {th*1e6:.3f} um ({100*(sim/th-1):+.1f} %)")


def section_ids(m):
    """Bodies along the x-z section (columns 0 and N_THETA/2) from rim to rim through the centre,
    bottom + heel + first wall row, with the sign of x for each."""
    half = N_THETA // 2
    left = [m.body(f"w{half}_{j}").id for j in range(NB + 1, -1, -1)]
    right = [m.body(f"w0_{j}").id for j in range(NB + 2)]
    return np.array(left + [m.body("c").id] + right)


MAG = 20                                    # displacement magnification for the static renders


def magnified(m, d, q0, k, rigid=()):
    """Copy of the state with every panel's displacement and rotation from q0 scaled by k. Bodies in
    `rigid` (the cube) only get their vertical drop scaled, keeping their real pose otherwise."""
    dm = mujoco.MjData(m); dm.qpos[:] = d.qpos
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        if m.jnt_bodyid[j] in rigid:
            a = m.jnt_qposadr[j] + 2
            dm.qpos[a] = q0[a] + k * (d.qpos[a] - q0[a])
            continue
        a = m.jnt_qposadr[j]
        dm.qpos[a:a + 3] = q0[a:a + 3] + k * (d.qpos[a:a + 3] - q0[a:a + 3])
        rot = np.zeros(3); mujoco.mju_subQuat(rot, d.qpos[a + 3:a + 7], q0[a + 3:a + 7])
        qm = q0[a + 3:a + 7].copy(); mujoco.mju_quatIntegrate(qm, rot, k)
        dm.qpos[a + 3:a + 7] = qm
    mujoco.mj_kinematics(m, dm)
    return dm


def run(cube_mass):
    sh, m, d, w = build_cup(cube_mass)
    tag = f"{int(round(cube_mass * 1000))}g"
    c, cube = m.body("c").id, m.body("cube").id
    heel = np.array([m.body(f"w{i}_{NB}").id for i in range(N_THETA)])
    bottom = np.array([m.body("c").id] + [m.body(f"w{i}_{j}").id for i in range(N_THETA) for j in range(NB)])
    sec = section_ids(m)
    mujoco.mj_forward(m, d)
    q0 = d.qpos.copy()
    zc0, zh0, zb0, zq0 = d.xipos[c][2], d.xipos[heel][:, 2].mean(), d.xipos[bottom][:, 2].copy(), d.xpos[cube][2]
    sec0 = d.xipos[sec].copy()

    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, Z0 + 0.004]
    cam.distance, cam.elevation, cam.azimuth = 0.085, -4, 90
    opt = mujoco.MjvOption()
    frames, log, fps, slow = [], [], 30, 10
    dof = m.jnt_dofadr[m.body_jntadr[cube]]
    relaxing = False
    t0 = time.time(); k = 0
    while d.time < T_FREE + T_RELAX:
        if not relaxing and d.time >= T_FREE:
            relaxing = True
            sh.relax = 2 * w[0]
            m.dof_damping[dof:dof + 3] = 400 * cube_mass
        mujoco.mj_step(m, d); k += 1
        if k % 20 == 0:
            log.append([d.time, zc0 - d.xipos[c][2], zh0 - d.xipos[heel][:, 2].mean(),
                        zq0 - d.xpos[cube][2], np.abs(d.qvel).max()])
        if len(frames) < d.time * fps * slow:
            r.update_scene(d, cam, opt); frames.append(r.render())
    log = np.array(log)
    assert np.all(np.isfinite(log)), "blew up"
    media.write_video(os.path.join(OUT, f"cup_hang_cube_{tag}.mp4"), frames, fps=fps)
    media.write_image(os.path.join(OUT, f"cup_hang_cube_{tag}_static.png"), frames[-1])
    q_ref = q0.copy(); q_ref[m.jnt_qposadr[m.body_jntadr[cube]] + 2] -= GAP   # cube touching the rest bottom
    r.update_scene(magnified(m, d, q_ref, MAG, rigid=(cube,)), cam, opt)
    media.write_image(os.path.join(OUT, f"cup_hang_cube_{tag}_x{MAG}.png"), r.render())
    w_bot = zb0 - d.xipos[bottom][:, 2]
    res = dict(mass=cube_mass, dt=m.opt.timestep, wall=time.time() - t0, log=log,
               sec0=sec0, sec=d.xipos[sec].copy(), w_centre=log[-1, 1], w_heel=log[-1, 2],
               w_bot_max=w_bot.max(), cube_drop=log[-1, 3], vmax=log[-1, 4], ncon=d.ncon,
               pen=-min([c.dist for c in d.contact[:d.ncon]], default=0.0))
    np.savez(os.path.join(OUT, f"cup_hang_cube_{tag}.npz"), **res)
    return res


def report(results):
    print(f"\n{'cube':>6} {'centre sag':>11} {'heel drop':>10} {'bottom sag':>11} {'cube drop':>10} "
          f"{'peak (free)':>12} {'penetr.':>8} {'end |qvel|':>10} {'wall':>6}")
    for R in results:
        tt, wc = R["log"][:, 0], R["log"][:, 1]
        pk = wc[tt < T_FREE].max()
        print(f"{R['mass']*1e3:5.0f}g {R['w_centre']*1e3:9.4f}mm {R['w_heel']*1e3:8.4f}mm "
              f"{(R['w_centre']-R['w_heel'])*1e3:9.4f}mm {R['cube_drop']*1e3:8.3f}mm {pk*1e3:10.4f}mm "
              f"{R['pen']*1e6:6.1f}um "
              f"{R['vmax']:10.1e} {R['wall']:5.0f}s")
    print("(centre sag: centre panel drop; heel drop: mean drop of the heel ring (wall stretch + rotation);"
          "\n bottom sag = centre - heel; cube drop includes the 2 mm release gap)")

    C = ["#2a78d6", "#eb6834", "#1baf7a"]; INK, MUTED = "#222222", "#8a8a85"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    s0 = results[0]["sec0"]
    x = np.sign(s0[:, 0]) * np.hypot(s0[:, 0], s0[:, 1]) * 1e3
    flat = np.abs(s0[:, 2] - Z0) < 1e-4                        # bottom panels (not heel / wall)
    for R, col in zip(results, C):
        wz = (s0[:, 2] - R["sec"][:, 2]) * 1e3
        ax[0].plot(x[flat], wz[flat], "o-", color=col, lw=1.5, ms=5, label=f"{R['mass']*1e3:.0f} g")
    ax[0].axvspan(-CUBE_HALF * np.sqrt(2) * 1e3, CUBE_HALF * np.sqrt(2) * 1e3, color=MUTED, alpha=0.1)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="x [mm] (section through the axis; shaded: cube corner radius)",
              ylabel="bottom sag [mm]", title="Static bottom deflection (panel centroids)")
    ax[0].legend(frameon=False)
    for R, col in zip(results, C):
        tt, wc = R["log"][:, 0], R["log"][:, 1]
        ax[1].plot(tt * 1e3, wc * 1e3, color=col, lw=1.2, label=f"{R['mass']*1e3:.0f} g")
    ax[1].axvspan(T_FREE * 1e3, (T_FREE + T_RELAX) * 1e3, color=MUTED, alpha=0.12)
    ax[1].text((T_FREE + T_RELAX / 2) * 1e3, ax[1].get_ylim()[1] * 0.95, "dynamic\nrelaxation",
               ha="center", va="top", color=MUTED, fontsize=9)
    ax[1].set(xlabel="time [ms]", ylabel="centre panel sag [mm]", title="Bottom centre after cube release")
    ax[1].legend(frameon=False, loc="center left")
    for a in ax: a.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cup_hang_cube.png"), dpi=130)

    for suf in ("static", f"x{MAG}"):
        ims = [media.read_image(os.path.join(OUT, f"cup_hang_cube_{int(round(R['mass']*1000))}g_{suf}.png"))
               for R in results]
        media.write_image(os.path.join(OUT, f"cup_hang_cube_{suf}.png"), np.concatenate(ims, axis=1))
    print(f"\nwrote out/cup_hang_cube.png, out/cup_hang_cube_static.png and out/cup_hang_cube_x{MAG}.png "
          f"(10 | 100 | 500 g; x{MAG} = displacements magnified {MAG}x), out/cup_hang_cube_<mass>.mp4")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1:
        R = run(float(sys.argv[1])); report([R])
    else:
        plate_check()
        sh, m, d, w = build_cup(0.1)
        print(f"cup: {len(sh.panels)} panels, {len(sh.joints)} joints, wall mass "
              f"{m.body_mass[sh.bodies[sh.free]].sum()*1e3:.2f} g, s_m={S_M}, dt={m.opt.timestep:.2e} s, "
              f"first cup mode {w[0]/2/np.pi:.0f} Hz; running {T_FREE + T_RELAX} s per mass in parallel...")
        with Pool(len(MASSES)) as pool:
            results = pool.map(run, MASSES)
        report(results)
