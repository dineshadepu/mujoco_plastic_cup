"""Rigid cup resting on the ground; a cube is released above it and falls inside.

  python cup_rigid_cube.py [cube_mass]                   # writes cup_rigid_cube.xml + .mp4
  python -m mujoco.viewer --mjcf=cup_rigid_cube.xml      # interactive
"""
import sys, mujoco, mediapy as media
from cup_rigid import cup_rigid_geoms

# solref: contact time constant 2 ms (default 20 ms let the cube sink ~12 mm into the
# 1 mm cup bottom before being pushed back out). Must stay >= 2*dt.
def build_xml(cube_mass=0.1, cube_half=0.0125, cube_z=0.15, mu=0.5, dt=5e-4, solref="0.002 1"):
    return f"""
<mujoco model="cup_rigid_cube">
  <option timestep="{dt}"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <default><geom friction="{mu} 0.005 0.0001" solref="{solref}"/></default>
  <asset>
    <texture name="g" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".7 .7 .7" width="512" height="512"/>
    <material name="g" texture="g" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 -0.3 0.8" dir="0 0.4 -1"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.01" material="g"/>
    <body name="cup" pos="0 0 0">
      <freejoint/>
        {cup_rigid_geoms(rgba="0.3 0.6 0.9 0.5")}
    </body>
    <body name="cube" pos="0 0 {cube_z}" euler="0 0 30">
      <freejoint/>
      <geom type="box" size="{cube_half} {cube_half} {cube_half}" mass="{cube_mass}" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>"""

if __name__ == "__main__":
    mass = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    xml = build_xml(cube_mass=mass)
    open("cup_rigid_cube.xml", "w").write(xml)
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)

    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.07]
    cam.distance, cam.elevation, cam.azimuth = 0.4, -20, 90
    frames, fps, T = [], 30, 1.5
    while d.time < T:
        mujoco.mj_step(m, d)
        if len(frames) < d.time*fps:
            r.update_scene(d, cam); frames.append(r.render())
    media.write_video("cup_rigid_cube.mp4", frames, fps=fps)
    print(f"t={T}s: cube z = {1e3*d.body('cube').xpos[2]:.2f} mm, "
          f"cup z = {1e3*d.body('cup').xpos[2]:.2f} mm  -> cup_rigid_cube.mp4")
