"""Showcase video: cube dropped into the deformable cup, Panda grips (wall dents), lifts, camera pulls back.

Same model and robot as 05_panda_cup_shell.py, with a video-friendly schedule: the robot starts 20 cm
above the grasp (out of the close-up), the cube hangs inside the cup and is released, then the usual
descent, force-controlled close and lift. The camera follows keyframes; each phase has its own
slow-motion factor. 1920x1080, 30 fps.

  python 06_linkedin_video.py          # 100 g cube -> out/showcase_100g.mp4
  python 06_linkedin_video.py 0.5      # cube mass in kg
"""
import os, sys, time, importlib, numpy as np, mujoco, mediapy as media
P = importlib.import_module("05_panda_cup_shell")
from panda_cup_rigid import Arm, R_DOWN, smooth

DROP_H = 0.040                       # cube bottom above the cup bottom at release [m]
PRE_Z = 0.20                         # start this far above the grasp point
T_REL, T_APPROACH, T_CLOSE, T_LIFT, T_END = 0.25, 0.8, 1.6, 2.3, 3.9
T_LIFT_END = T_LIFT + 1.0
F_GRIP = P.F_GRIP
W, H, FPS = 1920, 1080, 30
# (sim start, sim end, slow-motion factor)
SEGMENTS = [(0.0, T_REL, 4), (T_REL, T_APPROACH, 4), (T_APPROACH, T_CLOSE, 2),
            (T_CLOSE, T_LIFT, 3), (T_LIFT, T_END, 1.5)]
END_HOLD = 1.0                       # s of still frame at the end
CUP = P.CUP_POS
# camera keyframes: sim time, lookat, distance, elevation, azimuth
KEYS = [(0.00, CUP + [0, 0, 0.050], 0.34, -30, 55),
        (0.30, CUP + [0, 0, 0.045], 0.20, -30, 60),
        (T_APPROACH, CUP + [0, 0, 0.045], 0.21, -26, 62),
        (T_CLOSE, CUP + [0, 0, 0.050], 0.23, -18, 55),
        (T_LIFT, CUP + [0, 0, 0.050], 0.22, -22, 75),
        (T_LIFT + 0.25, CUP + [0, 0, 0.090], 0.26, -20, 80),
        (T_LIFT_END + 0.2, np.array([0.33, 0.0, 0.30]), 1.10, -14, 135),
        (T_END, np.array([0.33, 0.0, 0.30]), 1.10, -14, 138)]


def camera(t, cam):
    for (t0, l0, d0, e0, a0), (t1, l1, d1, e1, a1) in zip(KEYS, KEYS[1:]):
        if t <= t1:
            s = smooth((t - t0) / (t1 - t0))
            break
    else:
        (t0, l0, d0, e0, a0) = (t1, l1, d1, e1, a1) = KEYS[-1]; s = 1.0
    cam.lookat[:] = l0 + s * (l1 - l0)
    cam.distance, cam.elevation, cam.azimuth = d0 + s * (d1 - d0), e0 + s * (e1 - e0), a0 + s * (a1 - a0)


def frame_times():
    ts = []
    for t0, t1, slow in SEGMENTS:
        ts.extend(np.arange(t0, t1, 1.0 / (FPS * slow)))
    return np.array(ts)


def style(s):
    """Video look: gradient sky, finer shadows, full-HD offscreen buffer."""
    s.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                  rgb1=[0.55, 0.65, 0.78], rgb2=[0.08, 0.09, 0.12], width=512, height=3072)
    s.visual.global_.offwidth, s.visual.global_.offheight = W, H
    s.visual.quality.shadowsize = 8192
    s.visual.quality.offsamples = 8


def plan(m, q_home):
    arm = Arm(m)
    p_grasp = CUP + [0, 0, P.GRASP_Z]
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
    seg(0, T_APPROACH, [q_pre] * n(0, T_APPROACH))
    app = line(q_pre, p_pre, p_grasp, n(T_APPROACH, T_CLOSE)); seg(T_APPROACH, T_CLOSE, app)
    seg(T_CLOSE, T_LIFT, [app[-1]] * n(T_CLOSE, T_LIFT))
    lift = line(app[-1], p_grasp, p_grasp + [0, 0, P.LIFT], n(T_LIFT, T_LIFT_END)); seg(T_LIFT, T_LIFT_END, lift)
    seg(T_LIFT_END, T_END, [lift[-1]] * n(T_LIFT_END, T_END))
    return np.array(ts), np.array(qs), q_pre


def grip_ctrl(t, width):
    if t < T_CLOSE:
        return 255.0
    F = 2 * F_GRIP * smooth((t - T_CLOSE) / (T_LIFT - T_CLOSE - 0.2))
    return float(np.clip((width - F / P.GRIP_KP) * 255 / 0.04, 0, 255))


def run(cube_mass):
    _, m, d, _sh = P.build(cube_mass, extra=style)       # keep _sh alive: its callback holds the springs
    m.vis.headlight.diffuse[:] = 0.3; m.vis.headlight.ambient[:] = 0.15
    tag = f"showcase_{int(round(cube_mass * 1000))}g"
    arm_act = [m.actuator(f"actuator{i}").id for i in range(1, 8)]
    g8 = m.actuator("actuator8").id
    arm_q = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)]
    fj = [m.jnt_qposadr[m.joint(f"finger_joint{i}").id] for i in (1, 2)]
    ts, qs, q_pre = plan(m, m.key("home").qpos[:7].copy())
    d.qpos[arm_q] = q_pre; d.qpos[fj] = 0.04
    d.ctrl[arm_act] = q_pre; d.ctrl[g8] = 255

    # cube hangs inside the cup (yawed a little) until T_REL
    cj = m.body_jntadr[m.body("cube").id]
    qa, va = m.jnt_qposadr[cj], m.jnt_dofadr[cj]
    z_bottom = P.ds.FOOT["foot_h"] + P.ds.CUP["t"] + 1e-4
    yaw = np.radians(20)
    cube_q = np.r_[CUP + [0, 0, z_bottom + DROP_H + P.CUBE_HALF], np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qpos[qa:qa + 7] = cube_q
    mujoco.mj_forward(m, d)

    rend = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    t_frames = frame_times()
    frames, k = [], 0
    n_ctrl = max(1, int(round(P.CTRL_EVERY / m.opt.timestep)))
    step, t0 = 0, time.time()
    print(f"{tag}: dt={m.opt.timestep:.2e} s, {len(t_frames)} frames, simulating {T_END} s...", flush=True)
    while d.time < T_END:
        if d.time < T_REL:                                    # hold the cube until release
            d.qpos[qa:qa + 7] = cube_q; d.qvel[va:va + 6] = 0
        if step % n_ctrl == 0:
            d.ctrl[arm_act] = [np.interp(d.time, ts, qs[:, j]) for j in range(7)]
            d.ctrl[g8] = grip_ctrl(d.time, d.qpos[fj[0]])
        mujoco.mj_step(m, d); step += 1
        while k < len(t_frames) and d.time >= t_frames[k]:
            camera(d.time, cam)
            rend.update_scene(d, cam); frames.append(rend.render()); k += 1
    frames += [frames[-1]] * int(END_HOLD * FPS)
    out = os.path.join(P.OUT, f"{tag}.mp4")
    media.write_video(out, frames, fps=FPS, qp=16)
    print(f"wrote {out}: {len(frames)/FPS:.1f} s video, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    os.makedirs(P.OUT, exist_ok=True)
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 0.1)
