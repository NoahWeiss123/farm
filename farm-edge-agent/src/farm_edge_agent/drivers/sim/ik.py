"""Damped-least-squares IK with multi-start + joint-limit avoidance.

mj_inverse is for forward-dynamics inversion, not joint IK. We use a small DLS
loop with mid-range nullspace regulation: solve the position+orientation error
in the operational space, project a secondary objective into the nullspace that
keeps joints away from limits, and retry with perturbed initial conditions if
the primary task can't reach tolerance from the current pose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    qpos: np.ndarray
    pos_err: float
    rot_err: float
    iterations: int
    converged: bool
    restarts: int


def _site_quat(data: mujoco.MjData, site_id: int) -> np.ndarray:
    mat = data.site_xmat[site_id].reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


def _quat_err_vec(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    err_quat = np.zeros(4)
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, current_quat)
    mujoco.mju_mulQuat(err_quat, target_quat, neg)
    rot_err = np.zeros(3)
    mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
    return rot_err


def _midrange_bias(model: mujoco.MjModel, arm_joint_ids: Sequence[int], qpos: np.ndarray) -> np.ndarray:
    """Gradient pulling each joint toward the middle of its range."""
    bias = np.zeros(len(arm_joint_ids))
    for k, j in enumerate(arm_joint_ids):
        if not model.jnt_limited[j]:
            continue
        lo, hi = model.jnt_range[j]
        mid = 0.5 * (lo + hi)
        bias[k] = (mid - qpos[k]) * 0.05
    return bias


def _solve_once(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None,
    arm_joint_ids: Sequence[int],
    qpos_addr: Sequence[int],
    dof_addr: Sequence[int],
    max_iter: int,
    pos_tol: float,
    rot_tol: float,
    damping: float,
    step_scale: float,
    *,
    axis_world: np.ndarray | None = None,
) -> tuple[float, float, int, bool]:
    """One IK descent. Three modes:

    * full quaternion (``target_quat`` set, ``axis_world`` None) — 3 pos + 3 rot.
    * axis-only (``axis_world`` set; ``target_quat`` ignored) — 3 pos + 2
      orientation, leaving a 1-DoF nullspace (free rotation about the
      target axis). Right mode for "gripper points down with free yaw".
    * position-only (``target_quat`` None, ``axis_world`` None) — 3 pos,
      3-DoF nullspace.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    pos_err_norm = float("inf")
    rot_err_norm = float("inf")
    use_full_rot = target_quat is not None and axis_world is None
    axis_only = axis_world is not None
    n = len(arm_joint_ids)
    last_iter = 0
    for it in range(max_iter):
        last_iter = it + 1
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[site_id].copy()
        pos_err = target_pos - current_pos
        pos_err_norm = float(np.linalg.norm(pos_err))
        if use_full_rot:
            current_quat = _site_quat(data, site_id)
            rot_err = _quat_err_vec(target_quat, current_quat)
            rot_err_norm = float(np.linalg.norm(rot_err))
            err = np.concatenate([pos_err, rot_err])
        elif axis_only:
            mat = data.site_xmat[site_id].reshape(3, 3)
            current_axis = mat @ np.array([0.0, 0.0, 1.0])
            # cross product = sin(θ)·axis-of-rotation between current and
            # target — this is the operational-space rotational residual that
            # is naturally zero only when current_axis == target_axis.
            axis_err = np.cross(current_axis, axis_world)
            rot_err_norm = float(np.linalg.norm(axis_err))
            err = np.concatenate([pos_err, axis_err])
        else:
            rot_err_norm = 0.0
            err = pos_err
        if pos_err_norm < pos_tol and rot_err_norm < rot_tol:
            return pos_err_norm, rot_err_norm, last_iter, True

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        if use_full_rot:
            J_full = np.vstack([jacp, jacr])
        elif axis_only:
            mat = data.site_xmat[site_id].reshape(3, 3)
            current_axis = mat @ np.array([0.0, 0.0, 1.0])
            cx, cy, cz = current_axis
            # d(axis)/dω = -[axis]× (right-multiply by jacr gives Jacobian
            # of `current_axis` with respect to the operational angular vel).
            skew = np.array([[0, -cz, cy], [cz, 0, -cx], [-cy, cx, 0]])
            axis_jac = -skew @ jacr
            J_full = np.vstack([jacp, axis_jac])
        else:
            J_full = jacp
        J = J_full[:, dof_addr]
        m = J.shape[0]
        JJt = J @ J.T + (damping**2) * np.eye(m)
        dq_primary = J.T @ np.linalg.solve(JJt, err)
        current_q = np.array([data.qpos[a] for a in qpos_addr])
        N = np.eye(n) - np.linalg.pinv(J) @ J
        dq_null = N @ _midrange_bias(model, arm_joint_ids, current_q)
        dq = step_scale * dq_primary + 0.3 * dq_null

        max_dq = 0.2
        norm = np.linalg.norm(dq)
        if norm > max_dq:
            dq *= max_dq / norm

        for k, q_idx in enumerate(qpos_addr):
            j_id = arm_joint_ids[k]
            new_val = data.qpos[q_idx] + float(dq[k])
            if model.jnt_limited[j_id]:
                lo, hi = model.jnt_range[j_id]
                margin = 1e-3
                new_val = max(lo + margin, min(hi - margin, new_val))
            data.qpos[q_idx] = new_val
    return pos_err_norm, rot_err_norm, last_iter, False


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None = None,
    arm_joint_ids: Sequence[int] | None = None,
    max_iter: int = 120,
    pos_tol: float = 1e-3,
    rot_tol: float = 2e-2,
    damping: float = 1e-2,
    step_scale: float = 0.5,
    seed_attempts: int = 4,
    *,
    axis_world: np.ndarray | None = None,
) -> IKResult:
    """Iterative damped-least-squares IK with multi-start.

    ``axis_world`` (when set) puts the IK in **axis-only** mode: instead of
    matching the full target quaternion, the chain aligns the site's local
    +Z to the supplied world-frame vector, leaving rotation about that axis
    unconstrained. Used for "gripper-down with free yaw" — the UF850's
    wrist hits its joint5 limit at most floor-level workspaces when fully
    constrained, so the extra DoF is what makes ground grasping reachable.
    """
    if arm_joint_ids is None:
        arm_joint_ids = [
            i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
        ]
    qpos_addr = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]
    dof_addr = [int(model.jnt_dofadr[j]) for j in arm_joint_ids]

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    if target_quat is not None:
        target_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)
        target_quat = target_quat / max(np.linalg.norm(target_quat), 1e-12)
    if axis_world is not None:
        axis_world = np.asarray(axis_world, dtype=np.float64).reshape(3)
        axis_world = axis_world / max(np.linalg.norm(axis_world), 1e-12)

    rng = np.random.default_rng(seed=42)
    initial_q = np.array([data.qpos[a] for a in qpos_addr])
    best: tuple[float, float, np.ndarray, int, bool] | None = None
    for attempt in range(seed_attempts):
        if attempt > 0:
            perturb = rng.uniform(-0.8, 0.8, size=len(qpos_addr))
            for k, q_idx in enumerate(qpos_addr):
                j_id = arm_joint_ids[k]
                lo, hi = model.jnt_range[j_id]
                val = initial_q[k] + perturb[k]
                if model.jnt_limited[j_id]:
                    val = max(lo + 1e-3, min(hi - 1e-3, val))
                data.qpos[q_idx] = val
        else:
            for k, q_idx in enumerate(qpos_addr):
                data.qpos[q_idx] = initial_q[k]
        # When constraining an axis, first burn a small pos-only IK pass to
        # snap into a reachable basin, then refine with axis alignment. The
        # two-stage flow is much more reliable than asking pos+axis together
        # from a singular start: position alone has a 3-DoF nullspace so it
        # almost always converges, and the axis stage only needs a tiny
        # rotation tweak from there.
        if axis_world is not None:
            _solve_once(
                model, data, site_id, target_pos, None,
                arm_joint_ids, qpos_addr, dof_addr,
                max_iter=max(20, max_iter // 3),
                pos_tol=pos_tol, rot_tol=rot_tol,
                damping=damping, step_scale=step_scale,
                axis_world=None,
            )
        pos_err, rot_err, iters, converged = _solve_once(
            model, data, site_id, target_pos, target_quat,
            arm_joint_ids, qpos_addr, dof_addr,
            max_iter, pos_tol, rot_tol, damping, step_scale,
            axis_world=axis_world,
        )
        cur_q = np.array([data.qpos[a] for a in qpos_addr])
        score = pos_err + rot_err
        if best is None or score < (best[0] + best[1]):
            best = (pos_err, rot_err, cur_q.copy(), iters, converged)
        if converged:
            break
    assert best is not None
    pos_err, rot_err, cur_q, iters, converged = best
    for k, q_idx in enumerate(qpos_addr):
        data.qpos[q_idx] = float(cur_q[k])
    mujoco.mj_forward(model, data)
    return IKResult(
        qpos=cur_q,
        pos_err=pos_err,
        rot_err=rot_err,
        iterations=iters,
        converged=converged,
        restarts=attempt,
    )
