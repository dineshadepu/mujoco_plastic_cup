"""Franka Panda picks a rigid cup (with a cube inside) vertically off the ground.

Robot: Franka Emika Panda from MuJoCo Menagerie (third_party/mujoco_menagerie).
Motion is hard-coded: joint-space move to a pre-grasp pose above the cup, straight vertical
descent, force-controlled grasp, straight vertical lift, hold. Cartesian segments are solved
offline with damped-least-squares IK and tracked by the Panda's joint position servos.

Top-down grasp. The Panda fingertip pads are 83 mm apart fully open, so the cup is a typical
200 ml disposable cup (72 mm rim, 50 mm base, 90 mm tall) that the fingers can straddle. (A
horizontal side grasp this close to the ground puts the Panda's wrist at its joint limits.)

  python panda_cup_rigid.py                  # all 3 trials (10 g, 100 g, 500 g) -> mp4 + npz
  python panda_cup_rigid.py 0.1              # one trial, cube mass in kg
  python panda_cup_rigid.py 0.1 --grip 20    # grip force per finger [N]
  mjpython panda_cup_rigid.py 0.1 --view     # live viewer instead of video
"""
import sys, os, time, numpy as np, mujoco, mujoco.viewer, mediapy as media
from cup_rigid import cup_rigid_parts

HERE = os.path.dirname(os.path.abspath(__file__))
PANDA = os.path.join(HERE, "third_party/mujoco_menagerie/franka_emika_panda/panda.xml")

CUP = dict(r_top=0.036, r_bot=0.025, h=0.090, t=0.001, cup_mass=0.005)
CUP_POS = np.array([0.5, 0.0, 0.0])
GRASP_Z = 0.060                        # fingertip-pad centre height: 30 mm below the rim
PRE_Z = 0.10                           # pre-grasp height above the grasp point
LIFT = 0.15                            # vertical lift [m]
R_DOWN = np.array([[0, 1, 0],          # hand pointing down, fingers closing along world x
                   [1, 0, 0],          # (the orientation of the Panda's home pose)
                   [0, 0, -1.]])
GRIP_KP, GRIP_KV, GRIP_FMAX = 2000.0, 20.0, 140.0   # tendon servo; 140 N tendon = 70 N per finger

# Phase schedule [s]
T_HOME, T_PRE, T_APPROACH, T_CLOSE, T_LIFT, T_END = 0.3, 2.0, 3.5, 4.3, 6.3, 7.5


def build_spec(cube_mass=0.1, cube_half=0.0125, mu=0.5, dt=5e-4, solref=(0.002, 1)):
    s = mujoco.MjSpec.from_file(PANDA)
    s.meshdir = os.path.join(os.path.dirname(PANDA), "assets")   # absolute, so to_xml works anywhere
    s.modelname = "panda_cup_rigid"
    s.option.timestep = dt
    # elliptic friction cone + high impratio: with the default pyramidal cone the pads creep on
    # the tapered wall (cup rose 4.5 mm while squeezed, slid 9.6 mm in the hand during the lift)
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.impratio = 100
    s.visual.global_.offwidth, s.visual.global_.offheight = 1280, 720
    fr = [mu, 0.005, 0.0001]

    # gravity compensation on the arm, so joint servos track without steady-state sag
    for b in s.bodies:
        if b.name.startswith("link") or b.name in ("hand", "left_finger", "right_finger"):
            b.gravcomp = 1
    # fingertip pads: same friction / contact stiffness as cup and cube
    for g in s.geoms:
        if g.classname.name.startswith("fingertip_pad"):
            g.friction, g.solref = fr, solref
    # stronger, force-controllable gripper servo (stock one gives only ~2 N squeeze on the cup)
    a = s.actuator("actuator8")
    a.gainprm[0] = GRIP_KP*0.04/255
    a.biasprm[:3] = [0, -GRIP_KP, -GRIP_KV]
    a.forcerange = [-GRIP_FMAX, GRIP_FMAX]
    # tool-centre point between the fingertip pads
    s.body("hand").add_site(name="tcp", pos=[0, 0, 0.103], size=[0.004]*3, rgba=[1, 0, 0, 1])

    tex = s.add_texture(name="g", type=mujoco.mjtTexture.mjTEXTURE_2D,
                        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                        rgb1=[.85, .85, .85], rgb2=[.7, .7, .7], width=512, height=512)
    mat = s.add_material(name="g", texrepeat=[8, 8])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "g"
    w = s.worldbody
    w.add_light(pos=[0.5, -0.5, 1.5], dir=[0, 0.3, -1])
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01],
               material="g", friction=fr, solref=solref)

    cup = w.add_body(name="cup", pos=list(CUP_POS))
    cup.add_freejoint()
    gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}
    for p in cup_rigid_parts(**CUP):
        cup.add_geom(name=p["name"], type=gtype[p["type"]], size=p["size"], pos=p["pos"],
                     quat=p["quat"], density=p["density"], friction=fr, solref=solref,
                     rgba=[0.3, 0.6, 0.9, 0.5])
    cube_pos = CUP_POS + [0, 0, CUP["t"] + cube_half + 0.002]
    cube = w.add_body(name="cube", pos=list(cube_pos))
    cube.add_freejoint()
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[cube_half]*3, mass=cube_mass,
                  friction=fr, solref=solref, rgba=[0.9, 0.3, 0.2, 1])
    # the Menagerie "home" keyframe covers only the robot; append cup and cube free-joint poses
    key = s.key("home")
    key.qpos = list(key.qpos)[:9] + [*CUP_POS, 1, 0, 0, 0] + [*cube_pos, 1, 0, 0, 0]
    return s


class Arm:
    """IK and joint bookkeeping for the 7 arm joints."""
    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.jid = [m.joint(f"joint{i}").id for i in range(1, 8)]
        self.qadr = m.jnt_qposadr[self.jid]
        self.vadr = m.jnt_dofadr[self.jid]
        self.lo, self.hi = m.jnt_range[self.jid].T
        self.site = m.site("tcp").id

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, pos, R, q0, iters=300, lam=1e-2, tol=1e-6):
        q = q0.copy()
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, Rc = self.fk(q)
            e = np.r_[pos - p, 0.5*sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))]
            if e @ e < tol**2:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp, jr])[:, self.vadr]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + lam**2*np.eye(6), e), self.lo, self.hi)
        return q, np.sqrt(e @ e)


def smooth(s):                        # 0..1 -> 0..1, zero velocity at both ends
    s = np.clip(s, 0, 1); return s*s*(3 - 2*s)


def plan(m, q_home):
    """Joint targets sampled every 10 ms for the whole motion. Returns (times, joint targets)."""
    arm = Arm(m)
    p_grasp = CUP_POS + [0, 0, GRASP_Z]
    p_pre = p_grasp + [0, 0, PRE_Z]
    q_pre, err = arm.ik(p_pre, R_DOWN, q_home)
    assert err < 1e-4, f"pre-grasp IK failed ({err:.2e})"

    def line(q_start, a, b, n):       # straight Cartesian segment, IK seeded along the way
        qs, q = [], q_start
        for s in smooth(np.linspace(0, 1, n)):
            q, err = arm.ik(a + s*(b - a), R_DOWN, q)
            assert err < 1e-4, f"IK failed on line ({err:.2e})"
            qs.append(q)
        return qs

    dt = 0.01
    ts, qs = [], []
    def seg(t0, t1, q_list):
        for t, q in zip(np.linspace(t0, t1, len(q_list)), q_list):
            ts.append(t); qs.append(q)
    n = lambda t0, t1: int(round((t1 - t0)/dt)) + 1
    seg(0, T_HOME, [q_home]*n(0, T_HOME))
    seg(T_HOME, T_PRE, [q_home + s*(q_pre - q_home) for s in smooth(np.linspace(0, 1, n(T_HOME, T_PRE)))])
    app = line(q_pre, p_pre, p_grasp, n(T_PRE, T_APPROACH)); seg(T_PRE, T_APPROACH, app)
    seg(T_APPROACH, T_CLOSE, [app[-1]]*n(T_APPROACH, T_CLOSE))
    lift = line(app[-1], p_grasp, p_grasp + [0, 0, LIFT], n(T_CLOSE, T_LIFT)); seg(T_CLOSE, T_LIFT, lift)
    seg(T_LIFT, T_END, [lift[-1]]*n(T_LIFT, T_END))
    return np.array(ts), np.array(qs)


def grip_ctrl(t, width, f_grip):
    """actuator8 ctrl. Open before T_APPROACH; afterwards force control: the servo target is set
    F/kp inside the current opening so it squeezes with tendon force 2*f_grip (f_grip per finger)."""
    if t < T_APPROACH:
        return 255.0
    F = 2*f_grip*smooth((t - T_APPROACH)/(T_CLOSE - T_APPROACH - 0.2))
    return float(np.clip((width - F/GRIP_KP)*255/0.04, 0, 255))


def contact_forces(m, d, bodies_a, bodies_b):
    """Sum of normal and tangential force magnitudes over contacts between two body sets."""
    fn = ft = 0.0; f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in bodies_a and b2 in bodies_b) or (b1 in bodies_b and b2 in bodies_a):
            mujoco.mj_contactForce(m, d, i, f6)
            fn += abs(f6[0]); ft += np.hypot(f6[1], f6[2])
    return fn, ft


def run(cube_mass, f_grip=20.0, mu=0.5, view=False):
    s = build_spec(cube_mass=cube_mass, mu=mu)
    m = s.compile(); d = mujoco.MjData(m)
    tag = f"panda_cup_rigid_{int(round(cube_mass*1000))}g"
    open("panda_cup_rigid.xml", "w").write(s.to_xml())

    q_home = m.key("home").qpos[:7].copy()
    ts, qs = plan(m, q_home)
    arm_act = [m.actuator(f"actuator{i}").id for i in range(1, 8)]
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)

    fingers = {m.body("left_finger").id, m.body("right_finger").id}
    cup_id, cube_id = m.body("cup").id, m.body("cube").id
    fj = m.jnt_qposadr[m.joint("finger_joint1").id]

    def control(d):
        d.ctrl[arm_act] = [np.interp(d.time, ts, qs[:, j]) for j in range(7)]
        d.ctrl[m.actuator("actuator8").id] = grip_ctrl(d.time, d.qpos[fj], f_grip)

    log = []
    def record(d):
        fn, ft = contact_forces(m, d, fingers, {cup_id})
        log.append([d.time, d.site_xpos[m.site("tcp").id][2], d.xpos[cup_id][2], d.xpos[cube_id][2],
                    2*d.qpos[fj], fn, ft])

    if view:
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running():
                t0 = time.time()
                control(d); mujoco.mj_step(m, d); v.sync()
                if d.time > T_END + 1: mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
                time.sleep(max(0, m.opt.timestep - (time.time() - t0)))
        return

    r = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0.45, 0, 0.12]
    cam.distance, cam.elevation, cam.azimuth = 0.75, -12, 120
    frames, fps, every = [], 30, int(round(0.005/m.opt.timestep))
    step = 0
    while d.time < T_END:
        control(d); mujoco.mj_step(m, d); step += 1
        if step % every == 0: record(d)
        if len(frames) < d.time*fps:
            r.update_scene(d, cam); frames.append(r.render())
    media.write_video(f"{tag}.mp4", frames, fps=fps)
    log = np.array(log)
    np.savez(f"{tag}.npz", log=log, cols="t tcp_z cup_z cube_z grip_width f_normal f_tangent".split(),
             cube_mass=cube_mass, f_grip=f_grip, mu=mu)
    lifted = log[-1, 2] - CUP_POS[2]
    hold = log[:, 0] > T_LIFT
    slip = np.ptp(log[hold, 1] - log[hold, 2])
    print(f"{tag}: cup lifted {1e3*lifted:6.1f} mm (target {1e3*LIFT:.0f}), "
          f"cube in cup: {log[-1, 3] - log[-1, 2] < 0.05}, "
          f"finger normal {log[hold, 5].mean():5.1f} N, tangential {log[hold, 6].mean():5.2f} N, "
          f"slip during hold {1e3*slip:.2f} mm  -> {tag}.mp4")


if __name__ == "__main__":
    args = sys.argv[1:]
    view = "--view" in args
    f_grip = float(args[args.index("--grip") + 1]) if "--grip" in args else 20.0
    masses = [float(a) for a in args if not a.startswith("--") and
              (args.index(a) == 0 or args[args.index(a) - 1] != "--grip")]
    for cm in masses or [0.01, 0.1, 0.5]:
        run(cm, f_grip=f_grip, view=view)
