"""Minimal MuJoCo example: a cube dropped on the ground.

  mjpython cube_drop.py          # live interactive viewer (macOS needs mjpython)
  python   cube_drop.py --video  # headless: render cube_drop.mp4
"""
import sys, mujoco, mediapy as media

XML = """
<mujoco model="cube_drop">
  <option timestep="0.002"/>
  <asset>
    <texture name="g" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".7 .7 .7" width="512" height="512"/>
    <material name="g" texture="g" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 -0.3 1" dir="0 0.3 -1"/>
    <geom type="plane" size="1 1 0.01" material="g"/>
    <body name="cube" pos="0 0 0.5" euler="20 30 0">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.1" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(XML); d = mujoco.MjData(m)

if "--video" in sys.argv:
    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.1]
    cam.distance, cam.elevation, cam.azimuth = 1.2, -20, 90
    frames, fps = [], 30
    while d.time < 2.0:
        mujoco.mj_step(m, d)
        if len(frames) < d.time*fps:
            r.update_scene(d, cam); frames.append(r.render())
    media.write_video("cube_drop.mp4", frames, fps=fps)
    print("wrote cube_drop.mp4")
else:
    import time, mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(max(0, m.opt.timestep - (time.time() - t0)))   # ~real time
