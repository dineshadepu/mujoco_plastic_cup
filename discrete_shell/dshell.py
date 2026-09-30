"""Reduced-order discrete shell for MuJoCo: rigid panels joined by spring-damper joints.

Model
-----
The shell mid-surface is cut into rigid panels (thin convex slabs, one free MuJoCo body each).
Two panels that share an edge are joined by a spring-damper joint at the edge midpoint. The
joint frame is e1 (in-plane, across the edge), e2 (along the edge) and e3 (surface normal):

  rotation about e2      bending            k = D l / d
  rotation about e1      twist              k = 2 D (1 - nu) l / d
  rotation about e3      in-plane rotation  k = s_m E' t l^3 / (12 d)
  translation along e1   membrane stretch   k = s_m E' t l / d
  translation along e2   membrane shear     k = s_m G t l / d
  translation along e3   transverse shear   k = s_m (5/6) G t l / d

D = E t^3 / (12 (1 - nu^2)) is the thin-shell bending rigidity, E' = E / (1 - nu^2),
G = E / (2 (1 + nu)), l is the shared edge length and d the distance between the two panel
centroids. The two bending springs come from equating the joint energy with the plate bending
energy (D/2) kappa^2 over the area l*d, with kappa = theta / d.

The springs act on the change from the rest pose, so a curved rest shape (a cup) is kept exactly.
MuJoCo's 2D flex bending assumes a flat rest shape, which is why the flex cup unrolled.

The membrane of a thin shell is about (d/t)^2 times stiffer than its bending. s_m < 1 scales the
membrane springs down (a penalty) so the time step stays usable. The shell is still effectively
inextensible as long as the membrane springs remain much stiffer than the bending springs; check
this by showing results don't change with s_m.

Damping: c = 2 zeta sqrt(k m_eff) per spring, where m_eff is the reduced mass (or inertia) of the two
panels at the joint. An optional `relax` term damps absolute panel velocity; it is used for dynamic
relaxation to a static equilibrium and is off otherwise.

MuJoCo has no element that joins two bodies with an anisotropic 6-DOF spring across a closed loop
of bodies. So the joint forces are computed in a passive-force callback (a numba-compiled loop over
the joints) and applied through d.xfrc_applied. Contact, friction, gravity and integration are all MuJoCo's own.
"""
import numpy as np, mujoco, numba

PP = dict(E=1.5e9, nu=0.40, rho=900.0)          # polypropylene: E 1.3-1.8 GPa, rho 900-910 kg/m^3
# cup used by the robot scene (72 mm rim, 50 mm base, 90 mm tall); thin-walled PP cup
CUP = dict(r_bot=0.025, r_top=0.036, height=0.090, t=0.5e-3)
# full cup mesh: panels around, bottom rings (+ centre panel), heel rows, wall rows
CUP_MESH = dict(n_theta=24, n_bottom=3, fillet_r=0.003, n_fillet=1, n_z=6)
# robot trials: the cup stands on a foot ring with the bottom recessed (as real PP cups do)
FOOT = dict(foot_h=0.0015, foot_w=0.002)


def plate_D(E, t, nu):
    return E * t**3 / (12 * (1 - nu**2))


def _unit(v):
    return v / np.linalg.norm(v)


def _mat2quat(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel()); return q


class Shell:
    """Panels plus joints; builds the MJCF bodies and applies the joint forces."""

    def __init__(self, t, E=PP["E"], nu=PP["nu"], rho=PP["rho"], s_m=1.0, zeta=0.3, name="sh", mass_scale=1.0):
        self.t, self.E, self.nu, self.rho, self.s_m, self.zeta = t, E, nu, rho, s_m, zeta
        self.mass_scale = mass_scale
        self.name = name
        self.panels = []    # dict(name, c (3,), R (3,3) columns u v n, verts (k,3) in the panel frame)
        self.joints = []    # dict(a, b (body names), p (3,), F (3,3) columns e1 e2 e3, l, d)
        self.relax = 0.0
        self.energy = 0.0
        self.grid = None
        mujoco.set_mjcb_passive(None)   # one shell per process; drop a previous shell's callback

    @property
    def D(self):
        return plate_D(self.E, self.t, self.nu)

    # ---------------------------------------------------------------- geometry
    def add_panel(self, corners, name=None):
        """A planar panel from its mid-surface corners (k,3), ordered counter-clockwise seen
        from the +normal side. Returns the panel's body name."""
        corners = np.asarray(corners, float)
        c = corners.mean(0)
        k = len(corners)
        u = _unit(corners[1] - corners[0])
        n = _unit(np.cross(corners[1] - corners[0], corners[k - 1] - corners[0]))
        v = np.cross(n, u)
        R = np.column_stack([u, v, n])
        loc = (corners - c) @ R
        loc[:, 2] = 0
        verts = np.vstack([loc + [0, 0, -self.t / 2], loc + [0, 0, self.t / 2]])
        name = name or f"{self.name}{len(self.panels)}"
        self.panels.append(dict(name=name, c=c, R=R, verts=verts))
        return name

    def add_joint(self, a, b, p0, p1, ca, cb, na, nb=None):
        """Joint between bodies a and b along the shared edge p0-p1. ca, cb are the two panel
        centroids (ca is where the clamp face sits for a fixed body), na, nb their normals."""
        p0, p1, ca, cb = map(np.asarray, (p0, p1, ca, cb))
        e3 = _unit(na + (na if nb is None else nb))
        e1 = cb - ca
        e1 = _unit(e1 - e3 * (e1 @ e3))
        e2 = np.cross(e3, e1)
        self.joints.append(dict(a=a, b=b, p=0.5 * (p0 + p1), F=np.column_stack([e1, e2, e3]),
                                l=np.linalg.norm(p1 - p0), d=np.linalg.norm(cb - ca)))

    # ---------------------------------------------------------------- MJCF
    def assets_xml(self):
        out = []
        for p in self.panels:
            out.append(f'<mesh name="{p["name"]}_m" vertex="' +
                       " ".join(f"{x:.7g}" for x in p["verts"].ravel()) + '"/>')
        return "\n    ".join(out)

    def bodies_xml(self, rgba="0.3 0.6 0.9 1"):
        out = []
        for p in self.panels:
            rgba_p = p.get("rgba", rgba)
            pos = " ".join(f"{x:.7g}" for x in p["c"])
            quat = " ".join(f"{x:.7g}" for x in _mat2quat(p["R"]))
            out.append(f'<body name="{p["name"]}" pos="{pos}" quat="{quat}"><freejoint/>'
                       f'<geom type="mesh" mesh="{p["name"]}_m" density="{self.rho}" '
                       f'contype="2" conaffinity="1" solmix="1e-4" rgba="{rgba_p}"/></body>')
        return "\n    ".join(out)

    def translate(self, v):
        """Move the whole shell (panels and joint anchors) by v, before building the model."""
        v = np.asarray(v, float)
        for p in self.panels: p["c"] = p["c"] + v
        for j in self.joints: j["p"] = j["p"] + v
        return self

    def add_to_spec(self, spec, friction=(0.5, 0.005, 0.0001), rgba="0.3 0.6 0.9 1"):
        """Add the panels to an MjSpec scene (the robot pipeline) instead of generating XML.
        Same bodies, meshes and collision/contact settings as bodies_xml. Friction is set explicitly:
        MjSpec's default is 1, and MuJoCo uses the larger of the two geoms' coefficients."""
        for p in self.panels:
            spec.add_mesh(name=f'{p["name"]}_m', uservert=p["verts"].ravel().tolist())
            b = spec.worldbody.add_body(name=p["name"], pos=p["c"].tolist(), quat=_mat2quat(p["R"]).tolist())
            b.add_freejoint()
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'{p["name"]}_m', density=self.rho,
                       contype=2, conaffinity=1, solmix=1e-4, friction=list(friction),
                       rgba=[float(x) for x in p.get("rgba", rgba).split()])

    # ---------------------------------------------------------------- binding
    def bind(self, m, d):
        """Look up bodies, compute local anchors, stiffness, damping, and install the callback."""
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        bid = lambda s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s)
        J = self.joints
        self.ia = np.array([bid(j["a"]) for j in J]); self.ib = np.array([bid(j["b"]) for j in J])
        assert (self.ia >= 0).all() and (self.ib >= 0).all()
        R = d.xmat.reshape(-1, 3, 3).copy()
        P = np.array([j["p"] for j in J]); F = np.array([j["F"] for j in J])
        self.la = np.einsum("nji,nj->ni", R[self.ia], P - d.xpos[self.ia])
        self.lb = np.einsum("nji,nj->ni", R[self.ib], P - d.xpos[self.ib])
        self.Qa = np.swapaxes(R[self.ia], 1, 2) @ F
        self.Qb = np.swapaxes(R[self.ib], 1, 2) @ F

        E, nu, t, s = self.E, self.nu, self.t, self.s_m
        Ep, G, D = E / (1 - nu**2), E / (2 * (1 + nu)), self.D
        l = np.array([j["l"] for j in J]); dd = np.array([j["d"] for j in J])
        self.k_lin = np.column_stack([s * Ep * t * l / dd, s * G * t * l / dd, s * 5 / 6 * G * t * l / dd])
        self.k_rot = np.column_stack([2 * D * (1 - nu) * l / dd, D * l / dd, s * Ep * t * l**3 / (12 * dd)])

        # reduced mass / inertia of the two panels for each joint DOF (a fixed body has zero mobility)
        def mobility(ib):
            free = np.array([m.body_dofnum[b] > 0 for b in ib])
            mass = np.where(free, m.body_mass[ib], 1.0)                 # world/fixed: masked below
            inert = np.where(free[:, None], m.body_inertia[ib], 1.0)
            Ri = d.ximat.reshape(-1, 3, 3)[ib]
            Iinv = Ri @ (np.eye(3)[None] / inert[:, None, :]) @ np.swapaxes(Ri, 1, 2)
            r = P - d.xipos[ib]
            lin = np.empty((len(ib), 3)); rot = np.empty((len(ib), 3))
            for k in range(3):
                e = F[:, :, k]
                rxe = np.cross(r, e)
                lin[:, k] = 1 / mass + np.einsum("ni,nij,nj->n", rxe, Iinv, rxe)
                rot[:, k] = np.einsum("ni,nij,nj->n", e, Iinv, e)
            return lin * free[:, None], rot * free[:, None]
        la_, ra_ = mobility(self.ia); lb_, rb_ = mobility(self.ib)
        self.m_lin, self.m_rot = 1 / (la_ + lb_), 1 / (ra_ + rb_)
        if self.mass_scale != 1.0:
            # Selective mass scaling: raise every panel's inertia by mass_scale through MuJoCo's
            # armature (added to the mass matrix diagonal). Armature has no weight, so gravity loads
            # and static equilibria are unchanged; only the (very fast) panel vibrations slow down,
            # which raises the stable time step by sqrt(mass_scale).
            k = self.mass_scale - 1.0
            for b in np.unique(np.concatenate([self.ia, self.ib])):
                if m.body_dofnum[b] == 6:
                    a = m.body_dofadr[b]
                    m.dof_armature[a:a + 3] += k * m.body_mass[b]
                    m.dof_armature[a + 3:a + 6] += k * m.body_inertia[b].max()
            self.m_lin, self.m_rot = self.m_lin * self.mass_scale, self.m_rot * self.mass_scale
        self.c_lin = 2 * self.zeta * np.sqrt(self.k_lin * self.m_lin)
        self.c_rot = 2 * self.zeta * np.sqrt(self.k_rot * self.m_rot)

        # scatter: joint -> row of self.bodies
        self.bodies = np.unique(np.concatenate([self.ia, self.ib]))
        row = {b: i for i, b in enumerate(self.bodies)}
        self.row_a = np.array([row[b] for b in self.ia]); self.row_b = np.array([row[b] for b in self.ib])
        self.free = np.array([m.body_dofnum[b] > 0 for b in self.bodies])
        self.mass = m.body_mass[self.bodies]
        mujoco.set_mjcb_passive(self._passive)
        return self

    # ---------------------------------------------------------------- forces
    def _passive(self, m, d):
        frc, trq, self.energy = _joint_forces(
            d.xpos, d.xmat, d.cvel, d.subtree_com, m.body_rootid, d.xipos, self.ia, self.ib,
            self.la, self.lb, self.Qa, self.Qb, self.k_lin, self.c_lin, self.k_rot, self.c_rot,
            self.row_a, self.row_b, len(self.bodies))
        if self.relax:
            bd = self.bodies
            c = d.subtree_com[m.body_rootid[bd]]
            vc = d.cvel[bd, 3:] + np.cross(d.cvel[bd, :3], d.xipos[bd] - c)
            frc -= self.relax * self.mass[:, None] * vc
        d.xfrc_applied[self.bodies, :3] = frc
        d.xfrc_applied[self.bodies, 3:] = trq

    # ---------------------------------------------------------------- analysis
    def modes(self, m, d, eps=1e-7):
        """Linearised natural frequencies (rad/s, ascending) about the current pose, from a
        finite-difference stiffness matrix with gravity and velocity switched off."""
        g, relax = m.opt.gravity.copy(), self.relax
        m.opt.gravity[:] = 0; self.relax = 0
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        d.qvel[:] = 0
        K = np.zeros((m.nv, m.nv))
        for i in range(m.nv):
            col = []
            for sgn in (1, -1):
                dq = np.zeros(m.nv); dq[i] = sgn * eps
                d.qpos[:] = q0; mujoco.mj_integratePos(m, d.qpos, dq, 1)
                mujoco.mj_forward(m, d)
                qf = np.zeros(m.nv)
                for b in self.bodies:
                    mujoco.mj_applyFT(m, d, d.xfrc_applied[b, :3], d.xfrc_applied[b, 3:], d.xipos[b], b, qf)
                col.append(qf + d.qfrc_passive)
            K[:, i] = -(col[0] - col[1]) / (2 * eps)
        d.qpos[:] = q0; mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, M)
        d.qvel[:] = v0; m.opt.gravity[:] = g; self.relax = relax
        L = np.linalg.cholesky(M)
        Li = np.linalg.inv(L)
        A = Li @ (0.5 * (K + K.T)) @ Li.T
        lam = np.linalg.eigvalsh(A)
        return np.sqrt(np.clip(lam, 0, None))

    def stable_dt(self, m, d, safety=0.5):
        """Largest time step for explicit (semi-implicit Euler) stability, times a safety factor.
        Undamped stability needs w_max dt < 2; the damping lowers that, hence the factor."""
        w = self.modes(m, d)
        return safety * 2 / w[-1], w

    def panel_centers(self, d):
        return d.xipos[self.bodies[self.free]].copy()


@numba.njit(cache=True)
def _joint_forces(xpos, xmat, cvel, subtree_com, rootid, xipos, ia, ib, la, lb, Qa, Qb,
                  k_lin, c_lin, k_rot, c_rot, row_a, row_b, nb):
    """Spring-damper joint wrenches, accumulated per shell body (rows of Shell.bodies).
    Joint frame F = R_a Q_a; displacement dlt = F^T (p_b - p_a); rotation phi = log(F_a^T F_b);
    rates from the body velocities; force/torque on b, equal and opposite on a, applied at the
    midpoint of the two anchors so linear and angular momentum are conserved."""
    frc = np.zeros((nb, 3)); trq = np.zeros((nb, 3)); energy = 0.0
    Ra = np.empty((3, 3)); Rb = np.empty((3, 3)); Fa = np.empty((3, 3)); Fb = np.empty((3, 3))
    Rr = np.empty((3, 3))
    pa = np.empty(3); pb = np.empty(3); va = np.empty(3); vb = np.empty(3)
    fl = np.empty(3); tl = np.empty(3); f = np.empty(3); tau = np.empty(3)
    for j in range(ia.shape[0]):
        a = ia[j]; b = ib[j]
        for r in range(3):
            for c in range(3):
                Ra[r, c] = xmat[a, 3 * r + c]; Rb[r, c] = xmat[b, 3 * r + c]
        for r in range(3):
            pa[r] = xpos[a, r] + Ra[r, 0] * la[j, 0] + Ra[r, 1] * la[j, 1] + Ra[r, 2] * la[j, 2]
            pb[r] = xpos[b, r] + Rb[r, 0] * lb[j, 0] + Rb[r, 1] * lb[j, 1] + Rb[r, 2] * lb[j, 2]
            for c in range(3):
                Fa[r, c] = Ra[r, 0] * Qa[j, 0, c] + Ra[r, 1] * Qa[j, 1, c] + Ra[r, 2] * Qa[j, 2, c]
                Fb[r, c] = Rb[r, 0] * Qb[j, 0, c] + Rb[r, 1] * Qb[j, 1, c] + Rb[r, 2] * Qb[j, 2, c]
        for r in range(3):
            for c in range(3):
                Rr[r, c] = Fa[0, r] * Fb[0, c] + Fa[1, r] * Fb[1, c] + Fa[2, r] * Fb[2, c]
        # rotation vector of Rr
        w0 = Rr[2, 1] - Rr[1, 2]; w1 = Rr[0, 2] - Rr[2, 0]; w2 = Rr[1, 0] - Rr[0, 1]
        s = 0.5 * np.sqrt(w0 * w0 + w1 * w1 + w2 * w2)
        co = 0.5 * (Rr[0, 0] + Rr[1, 1] + Rr[2, 2] - 1.0)
        fac = 0.5 if s < 1e-12 else np.arctan2(s, co) / (2.0 * s)
        phi0 = w0 * fac; phi1 = w1 * fac; phi2 = w2 * fac
        # point velocities (cvel: [angular, linear at subtree com of the root])
        ca = subtree_com[rootid[a]]; cb = subtree_com[rootid[b]]
        ra0 = pa[0] - ca[0]; ra1 = pa[1] - ca[1]; ra2 = pa[2] - ca[2]
        rb0 = pb[0] - cb[0]; rb1 = pb[1] - cb[1]; rb2 = pb[2] - cb[2]
        va[0] = cvel[a, 3] + cvel[a, 1] * ra2 - cvel[a, 2] * ra1
        va[1] = cvel[a, 4] + cvel[a, 2] * ra0 - cvel[a, 0] * ra2
        va[2] = cvel[a, 5] + cvel[a, 0] * ra1 - cvel[a, 1] * ra0
        vb[0] = cvel[b, 3] + cvel[b, 1] * rb2 - cvel[b, 2] * rb1
        vb[1] = cvel[b, 4] + cvel[b, 2] * rb0 - cvel[b, 0] * rb2
        vb[2] = cvel[b, 5] + cvel[b, 0] * rb1 - cvel[b, 1] * rb0
        phi = (phi0, phi1, phi2)
        for k in range(3):
            dl = Fa[0, k] * (pb[0] - pa[0]) + Fa[1, k] * (pb[1] - pa[1]) + Fa[2, k] * (pb[2] - pa[2])
            dd = Fa[0, k] * (vb[0] - va[0]) + Fa[1, k] * (vb[1] - va[1]) + Fa[2, k] * (vb[2] - va[2])
            pd = (Fa[0, k] * (cvel[b, 0] - cvel[a, 0]) + Fa[1, k] * (cvel[b, 1] - cvel[a, 1])
                  + Fa[2, k] * (cvel[b, 2] - cvel[a, 2]))
            fl[k] = -(k_lin[j, k] * dl + c_lin[j, k] * dd)
            tl[k] = -(k_rot[j, k] * phi[k] + c_rot[j, k] * pd)
            energy += 0.5 * (k_lin[j, k] * dl * dl + k_rot[j, k] * phi[k] * phi[k])
        for r in range(3):
            f[r] = Fa[r, 0] * fl[0] + Fa[r, 1] * fl[1] + Fa[r, 2] * fl[2]
            tau[r] = Fa[r, 0] * tl[0] + Fa[r, 1] * tl[1] + Fa[r, 2] * tl[2]
        m0 = 0.5 * (pa[0] + pb[0]); m1 = 0.5 * (pa[1] + pb[1]); m2 = 0.5 * (pa[2] + pb[2])
        # b gets (f, tau + (mid - com_b) x f); a gets (-f, -tau - (mid - com_a) x f)
        x0 = m0 - xipos[b, 0]; x1 = m1 - xipos[b, 1]; x2 = m2 - xipos[b, 2]
        y0 = m0 - xipos[a, 0]; y1 = m1 - xipos[a, 1]; y2 = m2 - xipos[a, 2]
        qa = row_a[j]; qb = row_b[j]
        for r in range(3):
            frc[qb, r] += f[r]; frc[qa, r] -= f[r]
        trq[qb, 0] += tau[0] + x1 * f[2] - x2 * f[1]
        trq[qb, 1] += tau[1] + x2 * f[0] - x0 * f[2]
        trq[qb, 2] += tau[2] + x0 * f[1] - x1 * f[0]
        trq[qa, 0] -= tau[0] + y1 * f[2] - y2 * f[1]
        trq[qa, 1] -= tau[1] + y2 * f[0] - y0 * f[2]
        trq[qa, 2] -= tau[2] + y0 * f[1] - y1 * f[0]
    return frc, trq, energy


# -------------------------------------------------------------------- shapes
def beam(L, b, t, n, z=0.1, **kw):
    """Cantilever strip along +x, clamped at x=0 to a fixed body 'clamp', n panels."""
    sh = Shell(t, name="p", **kw)
    h = L / n
    nrm = np.array([0, 0, 1.0])
    for i in range(n):
        x0, x1 = i * h, (i + 1) * h
        sh.add_panel([[x0, -b / 2, z], [x1, -b / 2, z], [x1, b / 2, z], [x0, b / 2, z]])
    sh.add_joint("clamp", "p0", [0, -b / 2, z], [0, b / 2, z], [0, 0, z], [h / 2, 0, z], nrm)
    for i in range(n - 1):
        x = (i + 1) * h
        sh.add_joint(f"p{i}", f"p{i+1}", [x, -b / 2, z], [x, b / 2, z],
                     [x - h / 2, 0, z], [x + h / 2, 0, z], nrm, nrm)
    return sh


def revolved(profile, t, n_theta, z0=0.0, center=False, clamp_top=False, theta0=0.0, **kw):
    """Shell of revolution from a profile polyline [(r, z), ...] ordered bottom to top.
    Ring j (between profile points j and j+1) has n_theta planar trapezoid panels named w{i}_{j}.
    center=True closes the bottom with one flat n_theta-gon panel 'c' inside the first profile
    point. clamp_top=True clamps the last profile edge (the rim) to the world with joints of arm d
    = edge-to-centroid distance, like the beam clamp. Adjacent panels may meet at any angle: the
    joints hold the rest angle, so a sharp wall-bottom corner needs no transition panels.
    theta0 rotates the mesh about the axis (panel column i spans theta0 + [i, i+1] * 2 pi / n_theta)."""
    sh = Shell(t, name="w", **kw)
    prof = np.asarray(profile, float)
    nr = len(prof) - 1
    th = theta0 + 2 * np.pi * np.arange(n_theta + 1) / n_theta
    P = np.array([[[r * np.cos(a), r * np.sin(a), z0 + z] for r, z in prof] for a in th])  # [i][j]
    C, N = {}, {}
    for j in range(nr):
        for i in range(n_theta):
            cs = [P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]]   # CCW seen from outside
            sh.add_panel(cs, name=f"w{i}_{j}")
            C[i, j] = sh.panels[-1]["c"]; N[i, j] = sh.panels[-1]["R"][:, 2]
    if center:
        sh.add_panel(P[n_theta - 1::-1, 0], name="c")                    # reversed: normal points down
        cc, nc = sh.panels[-1]["c"], sh.panels[-1]["R"][:, 2]
    for i in range(n_theta):
        ip = (i + 1) % n_theta
        for j in range(nr):
            sh.add_joint(f"w{i}_{j}", f"w{ip}_{j}", P[i + 1, j], P[i + 1, j + 1],
                         C[i, j], C[ip, j], N[i, j], N[ip, j])          # bending about the meridian
            if j + 1 < nr:
                sh.add_joint(f"w{i}_{j}", f"w{i}_{j+1}", P[i, j + 1], P[i + 1, j + 1],
                             C[i, j], C[i, j + 1], N[i, j], N[i, j + 1])  # bending about the parallel
        if center:
            sh.add_joint("c", f"w{i}_0", P[i, 0], P[i + 1, 0], cc, C[i, 0], nc, N[i, 0])
        if clamp_top:
            j = nr - 1
            sh.add_joint("world", f"w{i}_{j}", P[i, nr], P[i + 1, nr],
                         0.5 * (P[i, nr] + P[i + 1, nr]), C[i, j], N[i, j])
    sh.grid = (n_theta, nr)
    sh.P = P
    return sh


def cone_ring(r_bot, r_top, height, t, n_theta, n_z, z0=0.0, **kw):
    """Bottomless truncated cone (cup wall) of n_theta x n_z panels; w{i}_{j}, j up the wall."""
    prof = np.column_stack([np.linspace(r_bot, r_top, n_z + 1), np.linspace(0, height, n_z + 1)])
    return revolved(prof, t, n_theta, z0=z0, **kw)


def cup_profile(r_bot, r_top, height, n_bottom, n_z, fillet_r=0.0, n_fillet=0, center_frac=None,
                foot_h=0.0, foot_w=0.0, **_):
    """Cup profile (r, z) from the bottom centre-panel edge to the rim. The bottom is flat (z = 0)
    with n_bottom rings plus the centre panel of radius center_frac * (flat bottom radius)
    (default: the same width as a ring). n_fillet > 0 rounds the wall-bottom corner with an arc
    of radius fillet_r (tangent to bottom and wall) cut into n_fillet rows of intermediate angle;
    n_fillet = 0 gives a sharp corner. The wall is n_z rows up to the rim.
    foot_h > 0 recesses the bottom by foot_h, with one extra row stepping down over foot_w to a
    standing ring at the start of the heel (real cups stand on such a foot ring, not on the whole bottom)."""
    phi = np.arctan2(height, r_top - r_bot)                 # wall angle from horizontal
    tau = fillet_r * np.tan(phi / 2) if n_fillet else 0.0   # corner-to-tangent-point distance
    wdir = np.array([np.cos(phi), np.sin(phi)])
    t1 = np.array([r_bot - tau, 0.0])                       # end of the flat bottom
    t2 = np.array([r_bot, 0.0]) + tau * wdir                # start of the straight wall
    rb = t1[0] - (foot_w if foot_h else 0.0)                # outer radius of the flat bottom
    r0 = rb * (center_frac or 1 / (n_bottom + 1))            # centre panel radius
    pts = [[r, foot_h] for r in np.linspace(r0, rb, n_bottom + 1)]
    if foot_h:
        pts.append([t1[0], 0.0])                             # the standing ring
    if n_fillet:
        cen = t1 + [0, fillet_r]
        for a in np.linspace(0, phi, n_fillet + 1)[1:]:
            pts.append(list(cen + fillet_r * np.array([np.sin(a), -np.cos(a)])))
    top = np.array([r_top, height])
    pts += [list(t2 + (top - t2) * s) for s in np.linspace(0, 1, n_z + 1)[1:]]
    return np.array(pts)


# -------------------------------------------------------------------- scene / utilities
# Contact. MuJoCo's soft contact is mass-normalised and a panel weighs ~50 ug, so under a heavy object
# (the cube, gripper pads) a 2 ms time constant lets it sink ~1 mm into the panels (steady penetration
# ~ (1-d)/d * f * tc^2 / m_eff). CONTACT_HARD (tc = 0.1 ms, d = 0.99) keeps that to a few um. The floor
# only carries the cup's own weight and keeps the softer default, which also damps a landing (hard floor
# contact made the dropped wall bounce and chatter). Panels get solmix ~ 0, so the contact parameters
# are always those of the other geom (MuJoCo averages solref/solimp weighted by solmix).
CONTACT_SOFT = 'solref="0.002 1"'
CONTACT_HARD = 'solref="0.0001 1" solimp="0.99 0.999 0.001"'


def scene_xml(shell, dt, gravity=True, floor=True, mu=0.5, extra_body="", extra_asset="",
              integrator="Euler"):
    g = "0 0 -9.81" if gravity else "0 0 0"
    fl = '<geom name="floor" type="plane" size="0.5 0.5 0.01" material="g"/>' if floor else ""
    return f"""
<mujoco model="discrete_shell">
  <option timestep="{dt:.3g}" gravity="{g}" integrator="{integrator}" cone="elliptic" impratio="100"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <default><geom friction="{mu} 0.005 0.0001" {CONTACT_SOFT}/></default>
  <asset>
    <texture name="g" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".7 .7 .7" width="512" height="512"/>
    <material name="g" texture="g" texrepeat="8 8"/>
    {shell.assets_xml()}
    {extra_asset}
  </asset>
  <worldbody>
    <light pos="0 -0.3 0.8" dir="0 0.4 -1"/>
    {fl}
    {extra_body}
    {shell.bodies_xml()}
  </worldbody>
</mujoco>"""


def kabsch_residual(X, Y):
    """RMS and max distance between point sets X and Y after the best rigid fit of X onto Y."""
    xc, yc = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - xc).T @ (Y - yc))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    Rm = U @ S @ Vt
    r = np.linalg.norm((X - xc) @ Rm - (Y - yc), axis=1)
    return np.sqrt(np.mean(r**2)), r.max()


def kinetic_energy(m, d):
    return 0.5 * d.qvel @ _Mv(m, d)


def _Mv(m, d):
    out = np.zeros(m.nv); mujoco.mj_mulM(m, d, out, d.qvel); return out
