"""Franka Panda picks the deformable (discrete-shell) cup with a cube inside: the assignment trials.

Same robot, IK and force-controlled gripper as the rigid pipeline (../panda_cup_rigid.py); the rigid
cup is replaced by the discrete-shell cup of experiment 4 (s_m = 1) with one change: the bottom is recessed 1.5 mm and the cup stands on
a foot ring, as real PP cups do (265 panels). Only the foot ring touches the floor.
Hard-coded motion: start 10 cm above the grasp, settle, straight vertical descent, force-controlled
close, straight vertical lift of 15 cm, hold.

Logged for the objectives (all deformation is measured in the cup's own frame: the best rigid fit of
the panel positions is removed first, so lifting the cup is not counted as deformation):
  1. wall deformation while squeezed on the ground: radial displacement of the grasp-height ring,
     dent depth under each pad, ovalisation (bulge at 90 deg to the pads)
  2. the same during the lift and hold
  3. bottom deformation: centre panel sag relative to the foot ring the cup stands on
  4. friction: per-finger normal force, friction force, friction utilisation |f_t| / (mu f_n), the
     split of the vertical load into friction and "wedge" (normal force tilted by the cone and the
     dent), number of pad contacts, and slip of the wall under the pads

  python 05_panda_cup_shell.py                 # 3 trials (10, 100, 500 g) in parallel, then the figure
  python 05_panda_cup_shell.py 0.5             # one trial, cube mass in kg
  python 05_panda_cup_shell.py 0.5 --grip 8    # grip force per finger [N]
  python 05_panda_cup_shell.py 0.5 --until 1.5 --novideo   # quick look at the grasp only
Outputs in out/: panda_shell_<mass>_{overview,obj1_wall_grasp,obj2_wall_lift,obj3_bottom_x10,
obj4_friction}.mp4, panda_shell_<mass>.npz, and panda_shell.png (all trials). The bottom video is seen
from below with the bottom's deformation (centre panel, bottom rings, foot row) magnified 10x, after
removing the cup's rigid motion; the wall and all other videos are true scale.
The friction video draws contact forces as arrows (1 N = 2 mm).
"""
import os, sys, time, numpy as np, mujoco, mediapy as media
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from panda_cup_rigid import PANDA, R_DOWN, GRIP_KP, GRIP_KV, GRIP_FMAX, Arm, smooth
import dshell as ds

OUT = os.path.join(HERE, "out")
MASSES = [0.01, 0.1, 0.5]
MU = 0.5
F_GRIP = 8.0                               # N per finger (see README: lifts 500 g with margin)
S_M = 1.0
MASS_SCALE = 50.0                           # selective mass scaling (armature, no weight): dt x7, statics unchanged
CUP_POS = np.array([0.5, 0.0, 0.0])
MESH = {k: v for k, v in ds.CUP_MESH.items() if k != "n_theta"}
N_THETA, NB = ds.CUP_MESH["n_theta"], ds.CUP_MESH["n_bottom"]
PROFILE = ds.cup_profile(**ds.CUP, **MESH, **ds.FOOT)
WALL0 = NB + 2                              # rows: bottom 0..NB-1, foot NB, heel NB+1, wall NB+2..
ROW_G = WALL0 + 3                           # wall row whose centre is the grasp height
GRASP_Z = 0.5 * (PROFILE[ROW_G, 1] + PROFILE[ROW_G + 1, 1])      # 53.6 mm
PRE_Z, LIFT = 0.10, 0.15
CUBE_HALF = 0.0125
T_SETTLE, T_APPROACH, T_CLOSE, T_LIFT, T_END = 0.25, 0.75, 1.45, 2.45, 2.9
CTRL_EVERY = 5e-4                           # s between servo-target updates
LOG_EVERY = 1e-3


def build(cube_mass, mu=MU, extra=None):
    """extra(spec) is called just before compiling (e.g. to restyle the scene for a video)."""
    s = mujoco.MjSpec.from_file(PANDA)
    s.meshdir = os.path.join(os.path.dirname(PANDA), "assets")
    s.modelname = "panda_cup_shell"
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.impratio = 100
    s.visual.global_.offwidth, s.visual.global_.offheight = 1280, 720
    fr = [mu, 0.005, 0.0001]
    hard_ref, hard_imp = [1e-4, 1.0], [0.99, 0.999, 0.001, 0.5, 2]      # dshell.CONTACT_HARD
    for b in s.bodies:
        if b.name.startswith("link") or b.name in ("hand", "left_finger", "right_finger"):
            b.gravcomp = 1
    for g in s.geoms:                       # everything on the gripper that can touch the cup
        if g.parent.name in ("hand", "left_finger", "right_finger") and g.contype:
            g.friction, g.solref, g.solimp = fr, hard_ref, hard_imp
    a = s.actuator("actuator8")
    a.gainprm[0] = GRIP_KP * 0.04 / 255
    a.biasprm[:3] = [0, -GRIP_KP, -GRIP_KV]
    a.forcerange = [-GRIP_FMAX, GRIP_FMAX]
    s.body("hand").add_site(name="tcp", pos=[0, 0, 0.103], size=[0.004] * 3, rgba=[1, 0, 0, 1])

    tex = s.add_texture(name="g", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                        rgb1=[.85, .85, .85], rgb2=[.7, .7, .7], width=512, height=512)
    mat = s.add_material(name="g", texrepeat=[8, 8])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "g"
    w = s.worldbody
    w.add_light(pos=[0.5, -0.5, 1.5], dir=[0, 0.3, -1])
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], material="g",
               friction=fr, solref=[0.002, 1])                         # dshell.CONTACT_SOFT

    t = ds.CUP["t"]
    sh = ds.revolved(PROFILE, t, N_THETA, z0=t / 2 + 1e-4, center=True, theta0=-np.pi / N_THETA, s_m=S_M,
                     mass_scale=MASS_SCALE)
    sh.translate(CUP_POS)
    for p in sh.panels:
        j = -1 if p["name"] == "c" else int(p["name"].split("_")[1])
        p["rgba"] = "0.2 0.45 0.85 1" if j < WALL0 else "0.3 0.6 0.9 0.55"
    sh.add_to_spec(s, friction=fr)
    cube = w.add_body(name="cube", pos=list(CUP_POS + [0, 0, ds.FOOT["foot_h"] + t + 1e-4 + CUBE_HALF + 3e-4]))
    cube.add_freejoint()
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, mass=cube_mass, friction=fr,
                  solref=hard_ref, solimp=hard_imp, rgba=[0.9, 0.3, 0.2, 1])
    if extra is not None:
        extra(s)
    m = s.compile(); d = mujoco.MjData(m)
    sh.bind(m, d)
    dt, _ = sh.stable_dt(m, d)
    m.opt.timestep = dt
    m.vis.map.force = 0.002                 # contact-force arrows: 1 N -> 2 mm
    m.vis.scale.forcewidth = 0.02
    m.vis.headlight.diffuse[:] = 0.5        # camera light, so the view from below isn't dark
    return s, m, d, sh


def plan(m, q_home):
    """Joint targets every 10 ms. The robot starts at the pre-grasp pose."""
    arm = Arm(m)
    p_grasp = CUP_POS + [0, 0, GRASP_Z]
    p_pre = p_grasp + [0, 0, PRE_Z]
    q_pre, err = arm.ik(p_pre, R_DOWN, q_home)
    assert err < 1e-4, f"pre-grasp IK failed ({err:.2e})"

    def line(q, a, b, n):
        out = []
        for s_ in smooth(np.linspace(0, 1, n)):
            q, err = arm.ik(a + s_ * (b - a), R_DOWN, q)
            assert err < 1e-4, f"IK failed on line ({err:.2e})"
            out.append(q)
        return out
    n = lambda t0, t1: int(round((t1 - t0) / 0.01)) + 1
    ts, qs = [], []
    def seg(t0, t1, q_list):
        ts.extend(np.linspace(t0, t1, len(q_list))); qs.extend(q_list)
    seg(0, T_SETTLE, [q_pre] * n(0, T_SETTLE))
    app = line(q_pre, p_pre, p_grasp, n(T_SETTLE, T_APPROACH)); seg(T_SETTLE, T_APPROACH, app)
    seg(T_APPROACH, T_CLOSE, [app[-1]] * n(T_APPROACH, T_CLOSE))
    lift = line(app[-1], p_grasp, p_grasp + [0, 0, LIFT], n(T_CLOSE, T_LIFT)); seg(T_CLOSE, T_LIFT, lift)
    seg(T_LIFT, T_END, [lift[-1]] * n(T_LIFT, T_END))
    return np.array(ts), np.array(qs), q_pre


def grip_ctrl(t, width, f_grip):
    """Open until T_APPROACH, then force control: servo target F/kp inside the current opening,
    so the tendon force is 2 f_grip (f_grip per finger), ramped up over the close phase."""
    if t < T_APPROACH:
        return 255.0
    F = 2 * f_grip * smooth((t - T_APPROACH) / (T_CLOSE - T_APPROACH - 0.2))
    return float(np.clip((width - F / GRIP_KP) * 255 / 0.04, 0, 255))


class Probe:
    """Measurements in the cup frame (rigid motion removed) and pad contact forces."""
    def __init__(self, m, d, sh, mu):
        self.m, self.mu = m, mu
        self.ids = sh.bodies[sh.free]
        self.X0 = d.xipos[self.ids].copy()
        self.q0 = d.qpos.copy()
        self.c0 = self.X0.mean(0)
        names = [m.body(b).name for b in self.ids]
        idx = {n: k for k, n in enumerate(names)}
        self.ring = np.array([idx[f"w{i}_{ROW_G}"] for i in range(N_THETA)])
        self.heel = np.array([idx[f"w{i}_{NB}"] for i in range(N_THETA)])
        self.centre = idx["c"]
        self.wall = np.array([k for k, n in enumerate(names) if n != "c" and int(n.split("_")[1]) >= WALL0])
        rel = self.X0[self.ring] - self.c0
        self.th = np.arctan2(rel[:, 1], rel[:, 0])                    # ring panel angles; 0 and pi face the pads
        self.er = np.column_stack([np.cos(self.th), np.sin(self.th), 0 * self.th])
        self.pad_cols = [int(np.argmin(np.abs(np.angle(np.exp(1j * (self.th - a)))))) for a in (0, np.pi)]
        self.fingers = {m.body("left_finger").id: 0, m.body("right_finger").id: 1}
        self.cup = set(int(b) for b in self.ids)
        self.cube = m.body("cube").id

    def rigid(self, X):
        """Best rigid fit X ~ R (X0 - c0) + t, returns R, t and the deformation u in the cup frame."""
        xc = X.mean(0)
        H = (self.X0 - self.c0).T @ (X - xc)
        U, _, Vt = np.linalg.svd(H)
        S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R = (U @ S @ Vt).T
        u = (X - xc) @ R - (self.X0 - self.c0)
        return R, xc, u

    def pads(self, d):
        """Per finger: normal force, friction force (magnitudes), vertical force on the cup from the
        normal part (wedge) and from friction, number of contacts."""
        out = np.zeros((2, 5)); f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b1 in self.fingers and b2 in self.cup:
                k, sgn = self.fingers[b1], 1.0           # force on geom2 (the cup) along the frame
            elif b2 in self.fingers and b1 in self.cup:
                k, sgn = self.fingers[b2], -1.0
            else:
                continue
            mujoco.mj_contactForce(self.m, d, i, f6)
            fr = c.frame.reshape(3, 3)                    # rows: normal, tangent 1, tangent 2
            fn_w = sgn * f6[0] * fr[0]; ft_w = sgn * (f6[1] * fr[1] + f6[2] * fr[2])
            out[k] += [abs(f6[0]), np.hypot(f6[1], f6[2]), fn_w[2], ft_w[2], 1]
        return out

    def sample(self, d, tcp_z, width):
        X = d.xipos[self.ids]
        R, xc, u = self.rigid(X)
        ur = np.einsum("ij,ij->i", u[self.ring], self.er)             # + outward
        sag = -(u[self.centre, 2] - u[self.heel, 2].mean())            # + downward
        pads = self.pads(d)
        pad_z = X[self.ring[self.pad_cols], 2].mean()                  # wall panels under the pads
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        cube_rel = (R.T @ (d.xpos[self.cube] - xc))[2]
        return np.concatenate([[d.time, tcp_z, width, xc[2], tilt, sag, np.abs(u[self.wall]).max(),
                                pad_z, cube_rel], pads.ravel(), ur])


COLS = ["t", "tcp_z", "grip_width", "cup_z", "tilt_deg", "bottom_sag", "wall_umax", "pad_panel_z", "cube_rel_z"] + \
       [f"{f}_{q}" for f in ("L", "R") for q in ("fn", "ft", "fz_normal", "fz_friction", "ncon")]


MAG = 10                                    # deformation magnification in the bottom video


def magnified(m, d, dm, pr, k, bodies):
    """dm <- d with the deformation of `bodies` (after removing the cup's rigid motion) scaled by k;
    every other body keeps its true pose."""
    dm.qpos[:] = d.qpos
    R, xc, _ = pr.rigid(d.xipos[pr.ids])
    qR = np.zeros(4); mujoco.mju_mat2Quat(qR, R.ravel())
    qRi = np.zeros(4); mujoco.mju_negQuat(qRi, qR)
    for b in bodies:
        a = m.jnt_qposadr[m.body_jntadr[b]]
        p0, q0 = pr.q0[a:a + 3], pr.q0[a + 3:a + 7]
        u = R.T @ (d.qpos[a:a + 3] - xc) - (p0 - pr.c0)
        dm.qpos[a:a + 3] = xc + R @ (p0 - pr.c0 + k * u)
        q0i = np.zeros(4); mujoco.mju_negQuat(q0i, q0)
        qd = np.zeros(4); tmp = np.zeros(4)
        mujoco.mju_mulQuat(tmp, qRi, d.qpos[a + 3:a + 7]); mujoco.mju_mulQuat(qd, tmp, q0i)   # deformation rotation
        v = np.zeros(3); mujoco.mju_quat2Vel(v, qd, 1.0)
        ang = np.linalg.norm(v); qk = np.array([1.0, 0, 0, 0])
        if ang > 0: mujoco.mju_axisAngle2Quat(qk, v / ang, k * ang)
        mujoco.mju_mulQuat(tmp, qR, qk); mujoco.mju_mulQuat(dm.qpos[a + 3:a + 7], tmp, q0)
    mujoco.mj_kinematics(m, dm)


def cameras():
    def cam(lookat, dist, el, az):
        c = mujoco.MjvCamera(); c.lookat[:] = lookat; c.distance, c.elevation, c.azimuth = dist, el, az
        return c
    return dict(overview=cam([0.45, 0, 0.12], 0.75, -12, 120),
                pad=cam(CUP_POS + [0.0, 0, GRASP_Z - 0.01], 0.24, -12, 50),
                bottom=cam(CUP_POS + [0, 0, 0.01], 0.16, 35, 90))          # from below, looking up


def run(cube_mass, f_grip=F_GRIP, until=T_END, video=True):
    s, m, d, sh = build(cube_mass)
    tag = f"panda_shell_{int(round(cube_mass * 1000))}g"
    arm_act = [m.actuator(f"actuator{i}").id for i in range(1, 8)]
    q_home = m.key("home").qpos[:7].copy()
    ts, qs, q_pre = plan(m, q_home)
    arm_q = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)]
    fj = [m.jnt_qposadr[m.joint(f"finger_joint{i}").id] for i in (1, 2)]
    d.qpos[arm_q] = q_pre; d.qpos[fj] = 0.04
    d.ctrl[arm_act] = q_pre; d.ctrl[m.actuator("actuator8").id] = 255
    mujoco.mj_forward(m, d)
    pr = Probe(m, d, sh, MU)
    tcp = m.site("tcp").id

    cams = cameras()
    rend = mujoco.Renderer(m, 480, 640) if video else None
    rbig = mujoco.Renderer(m, 720, 1280) if video else None
    opt_f = mujoco.MjvOption()
    opt_f.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    frames = {k: [] for k in ("overview", "pad", "bottom", "friction")}
    dm = mujoco.MjData(m)
    bottom_ids = [b for b in pr.ids if m.body(b).name == "c" or int(m.body(b).name.split("_")[1]) < WALL0 - 1]
    qframes = []                                        # state at every video frame, to re-render later
    fps, slow = 30, 2                                   # videos at half speed

    n_ctrl = max(1, int(round(CTRL_EVERY / m.opt.timestep)))
    n_log = max(1, int(round(LOG_EVERY / m.opt.timestep)))
    log, step, t0 = [], 0, time.time()
    print(f"{tag}: dt={m.opt.timestep:.2e} s, grip {f_grip} N/finger, simulating {until} s...", flush=True)
    while d.time < until:
        if step % n_ctrl == 0:
            d.ctrl[arm_act] = [np.interp(d.time, ts, qs[:, j]) for j in range(7)]
            d.ctrl[m.actuator("actuator8").id] = grip_ctrl(d.time, d.qpos[fj[0]], f_grip)
        mujoco.mj_step(m, d); step += 1
        if step % n_log == 0:
            log.append(pr.sample(d, d.site_xpos[tcp][2], d.qpos[fj[0]] + d.qpos[fj[1]]))
        if video and len(frames["overview"]) < d.time * fps * slow:
            rbig.update_scene(d, cams["overview"]); frames["overview"].append(rbig.render())
            cp = cams["pad"]; cp.lookat[2] = d.site_xpos[tcp][2]          # follow the grasp
            rend.update_scene(d, cp); frames["pad"].append(rend.render())
            rend.update_scene(d, cp, opt_f); frames["friction"].append(rend.render())
            magnified(m, d, dm, pr, MAG, bottom_ids)
            qframes.append(d.qpos.copy())
            cb = cams["bottom"]; cb.lookat[2] = pr.rigid(d.xipos[pr.ids])[1][2] - 0.02
            rend.update_scene(dm, cb); frames["bottom"].append(rend.render())
    wall = time.time() - t0
    log = np.array(log)
    assert np.all(np.isfinite(log)), "blew up"
    np.savez(os.path.join(OUT, f"{tag}.npz"), log=log, cols=COLS, theta=pr.th, cube_mass=cube_mass,
             f_grip=f_grip, mu=MU, dt=m.opt.timestep, wall=wall,
             phases=[T_SETTLE, T_APPROACH, T_CLOSE, T_LIFT, T_END], qframes=np.array(qframes))
    if video:
        def clip(frs, t_a, t_b):
            i = lambda t: int(t * fps * slow)
            return frs[i(t_a):i(t_b)]
        media.write_video(os.path.join(OUT, f"{tag}_overview.mp4"), frames["overview"], fps=fps)
        media.write_video(os.path.join(OUT, f"{tag}_obj1_wall_grasp.mp4"),
                          clip(frames["pad"], T_SETTLE, T_CLOSE), fps=fps)
        media.write_video(os.path.join(OUT, f"{tag}_obj2_wall_lift.mp4"),
                          clip(frames["pad"], T_CLOSE - 0.1, T_END), fps=fps)
        media.write_video(os.path.join(OUT, f"{tag}_obj3_bottom_x{MAG}.mp4"), frames["bottom"], fps=fps)
        media.write_video(os.path.join(OUT, f"{tag}_obj4_friction.mp4"),
                          clip(frames["friction"], T_APPROACH, T_END), fps=fps)
    summary(tag, log, pr.th, cube_mass, wall)
    return tag


def summary(tag, log, th, cube_mass, wall):
    C = {c: i for i, c in enumerate(COLS)}
    t = log[:, 0]
    ur = log[:, len(COLS):]
    pad = [int(np.argmin(np.abs(np.angle(np.exp(1j * (th - a)))))) for a in (0, np.pi)]
    side = [int(np.argmin(np.abs(np.angle(np.exp(1j * (th - a)))))) for a in (np.pi / 2, -np.pi / 2)]
    def at(t_a, t_b):
        return (t >= t_a) & (t <= t_b)
    g = at(T_CLOSE - 0.1, T_CLOSE)                      # squeezed, on the ground
    h = at(T_LIFT + 0.1, T_END)                         # lifted, holding
    fn = log[:, C["L_fn"]] + log[:, C["R_fn"]]
    ft = log[:, C["L_ft"]] + log[:, C["R_ft"]]
    fzn = log[:, C["L_fz_normal"]] + log[:, C["R_fz_normal"]]
    fzf = log[:, C["L_fz_friction"]] + log[:, C["R_fz_friction"]]
    lifted = log[-1, C["cup_z"]] - log[0, C["cup_z"]]
    slip = (log[h, C["tcp_z"]] - log[h, C["pad_panel_z"]])
    slip = slip.max() - slip.min() if h.any() else np.nan
    print(f"\n{tag} ({wall/60:.1f} min wall):")
    for name, sel in (("on ground, squeezed", g), ("lifted, holding", h)):
        if not sel.any():
            continue
        print(f"  {name:20s}: pad dent {-ur[sel][:, pad].mean()*1e3:6.3f} mm, side bulge "
              f"{ur[sel][:, side].mean()*1e3:6.3f} mm, bottom sag {log[sel, C['bottom_sag']].mean()*1e3:6.3f} mm, "
              f"fn {fn[sel].mean():5.2f} N, ft {ft[sel].mean():5.2f} N, ft/(mu fn) {np.mean(ft[sel]/np.maximum(MU*fn[sel], 1e-9)):4.2f}, "
              f"lift by friction {fzf[sel].mean():5.2f} N + wedge {fzn[sel].mean():5.2f} N")
    print(f"  cup lifted {lifted*1e3:.1f} mm (target {LIFT*1e3:.0f}), tilt {log[-1, C['tilt_deg']]:.2f} deg, "
          f"pad slip during hold {slip*1e3:.3f} mm, cube stays in cup: {abs(log[-1, C['cube_rel_z']]) < 0.05}")


def figure():
    """All trials on one figure (from the saved logs): one panel per objective quantity."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = {c: i for i, c in enumerate(COLS)}
    col = {10: "#2a78d6", 100: "#eb6834", 500: "#1baf7a"}
    INK, MUTED = "#222222", "#8a8a85"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 3, figsize=(15, 8)); ax = ax.ravel()
    for g in (10, 100, 500):
        f = os.path.join(OUT, f"panda_shell_{g}g.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); log, th, c = z["log"], z["theta"], col[g]
        t = log[:, 0]; ur = log[:, len(COLS):] * 1e3
        pad = [int(np.argmin(np.abs(np.angle(np.exp(1j * (th - a)))))) for a in (0, np.pi)]
        o = np.argsort(th)
        for sel, ls, lab in (((t > T_CLOSE - 0.1) & (t <= T_CLOSE), "-", "on ground"),
                             (t > T_LIFT + 0.1, "--", "lifted")):
            if sel.any():
                ax[0].plot(np.degrees(th[o]), ur[sel].mean(0)[o], ls, color=c, lw=1.5, label=f"{g} g, {lab}")
        ax[1].plot(t, -ur[:, pad].mean(1), color=c, lw=1.3, label=f"{g} g")
        ax[2].plot(t, log[:, C["bottom_sag"]] * 1e3, color=c, lw=1.3, label=f"{g} g")
        fn = log[:, C["L_fn"]] + log[:, C["R_fn"]]; ft = log[:, C["L_ft"]] + log[:, C["R_ft"]]
        ax[3].plot(t, fn / 2, color=c, lw=1.3, label=f"{g} g normal")
        ax[3].plot(t, ft / 2, ":", color=c, lw=1.3, label=f"{g} g friction")
        util = np.where(fn > 0.2, ft / np.maximum(MU * fn, 1e-9), np.nan)
        ax[4].plot(t, util, color=c, lw=1.3, label=f"{g} g")
        ax[5].plot(t, log[:, C["L_fz_friction"]] + log[:, C["R_fz_friction"]], color=c, lw=1.3, label=f"{g} g friction")
        ax[5].plot(t, log[:, C["L_fz_normal"]] + log[:, C["R_fz_normal"]], "--", color=c, lw=1.3, label=f"{g} g wedge")
    for a in ax[1:]:
        for t0, t1, lab in ((T_APPROACH, T_CLOSE, "close"), (T_CLOSE, T_LIFT, "lift")):
            a.axvspan(t0, t1, color=MUTED, alpha=0.08)
    ax[0].set(xlabel="angle around the cup [deg] (pads at 0 and +-180)", ylabel="radial displacement [mm] (+ out)",
              title="Obj 1-2: grasp-height ring shape")
    ax[1].set(xlabel="time [s]", ylabel="dent under the pads [mm]", title="Obj 1-2: wall dent (grey: close, lift)")
    ax[2].set(xlabel="time [s]", ylabel="bottom sag rel. to foot ring [mm]", title="Obj 3: bottom deformation")
    ax[3].set(xlabel="time [s]", ylabel="force per finger [N]", title="Obj 4: pad normal and friction force")
    ax[4].set(xlabel="time [s]", ylabel="|f_t| / (mu f_n)", title="Obj 4: friction utilisation (1 = sliding)", ylim=(0, 1.1))
    ax[5].set(xlabel="time [s]", ylabel="vertical force on cup [N]", title="Obj 4: what carries the load")
    for a in ax:
        a.grid(alpha=0.2); a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "panda_shell.png"), dpi=130)
    print("wrote out/panda_shell.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "--figure" in sys.argv:
        figure(); sys.exit()
    args = sys.argv[1:]
    opt = lambda k, dflt: float(args[args.index(k) + 1]) if k in args else dflt
    f_grip, until = opt("--grip", F_GRIP), opt("--until", T_END)
    video = "--novideo" not in args
    vals = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--grip", "--until"))]
    if vals:
        run(float(vals[0]), f_grip, until, video)
    else:
        from multiprocessing import Pool
        with Pool(len(MASSES)) as pool:
            pool.starmap(run, [(mm, f_grip, until, video) for mm in MASSES])
        figure()
