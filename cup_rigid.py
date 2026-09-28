"""Rigid cup dropped onto the ground.

The cup is one rigid body made of convex pieces: a cylinder for the bottom and n_panels
thin slanted boxes for the wall. (A single mesh geom would collide as its convex hull,
closing the cup's opening, so the pieces keep the inside open for a cube later.)

  python cup_rigid.py                                  # writes cup_rigid_drop.xml + .mp4
  python -m mujoco.viewer --mjcf=cup_rigid_drop.xml    # interactive
"""
import numpy as np, mujoco, mediapy as media

def cup_rigid_parts(r_top=0.040, r_bot=0.0275, h=0.090, t=0.001, n_panels=32, cup_mass=0.005):
    """Cup pieces as dicts (name, type, size, pos, quat, density), in the cup body frame."""
    alpha = np.arctan2(r_top - r_bot, h)              # outward lean of the wall
    L = np.hypot(r_top - r_bot, h)                    # slant height
    w = 2*np.pi*r_top/n_panels * 1.05                 # panel width, slight overlap at the rim
    r_mid = 0.5*(r_bot + r_top) + 0.5*t*np.cos(alpha) # panel centre, wall inner face on the cone
    vol = n_panels*w*L*t + np.pi*r_bot**2*t
    rho = cup_mass/vol
    parts = [dict(name="cup_bottom", type="cylinder", size=[r_bot, t/2, 0],
                  pos=[0, 0, t/2], quat=[1, 0, 0, 0], density=rho)]
    qy = np.array([np.cos(alpha/2), 0, np.sin(alpha/2), 0])
    for i in range(n_panels):
        th = 2*np.pi*i/n_panels
        qz = np.array([np.cos(th/2), 0, 0, np.sin(th/2)])
        q = np.empty(4); mujoco.mju_mulQuat(q, qz, qy)
        parts.append(dict(name=f"cup_wall{i}", type="box", size=[t/2, w/2, L/2],
                          pos=[r_mid*np.cos(th), r_mid*np.sin(th), h/2], quat=list(q), density=rho))
    return parts

def cup_rigid_geoms(rgba="0.3 0.6 0.9 1", **kw):
    f = lambda v: " ".join(f"{x:.6f}" for x in v)
    return "\n        ".join(
        f'<geom name="{p["name"]}" type="{p["type"]}" size="{f(p["size"][:2] if p["type"] == "cylinder" else p["size"])}" '
        f'pos="{f(p["pos"])}" quat="{f(p["quat"])}" density="{p["density"]:.1f}" rgba="{rgba}"/>'
        for p in cup_rigid_parts(**kw))

def build_xml(z0=0.2, mu=0.5, dt=5e-4):
    return f"""
<mujoco model="cup_rigid_drop">
  <option timestep="{dt}"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <default><geom friction="{mu} 0.005 0.0001"/></default>
  <asset>
    <texture name="g" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".7 .7 .7" width="512" height="512"/>
    <material name="g" texture="g" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 -0.3 0.8" dir="0 0.4 -1"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.01" material="g"/>
    <body name="cup" pos="0 0 {z0}">
      <freejoint/>
        {cup_rigid_geoms()}
    </body>
  </worldbody>
</mujoco>"""

if __name__ == "__main__":
    xml = build_xml()
    open("cup_rigid_drop.xml", "w").write(xml)
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    print(f"cup mass = {1e3*m.body('cup').mass[0]:.2f} g")

    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.1]
    cam.distance, cam.elevation, cam.azimuth = 0.5, -15, 90
    frames, fps, T = [], 30, 1.5
    while d.time < T:
        mujoco.mj_step(m, d)
        if len(frames) < d.time*fps:
            r.update_scene(d, cam); frames.append(r.render())
    media.write_video("cup_rigid_drop.mp4", frames, fps=fps)
    print(f"t={T}s: cup z = {1e3*d.body('cup').xpos[2]:.2f} mm  -> cup_rigid_drop.mp4")
