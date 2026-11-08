# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim 5.1 and 6.0 compatibility for the two Isaac Sim utilities this package uses (terraforge, 2026-11-08).

On Isaac Sim 5.1 (this workshop's own image, Isaac Lab 2.3.2) both names come from Isaac Sim itself, unchanged.
Isaac Sim 6.0 (Isaac Lab 3.0) dropped `isaacsim.core.utils.rotations` and `isaacsim.core.prims`; there the
fallbacks below stand in, covering only what this package calls:
  - euler_angles_to_quat(euler, degrees=False, extrinsic=True) -> (w, x, y, z), as Isaac Sim's;
  - XFormPrim(prim_paths_expr=...) with .prims, set_local_poses(orientations=...) and get_local_poses().
Needed to run the twin on r2d2, whose driver (595.91) Isaac Sim 5.1's RTX renderer crashes on.
"""
from __future__ import annotations

try:
    from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: F401
except ImportError:
    import numpy as np
    from scipy.spatial.transform import Rotation

    def euler_angles_to_quat(euler_angles, degrees: bool = False, extrinsic: bool = True):
        x, y, z, w = Rotation.from_euler("xyz" if extrinsic else "XYZ", np.asarray(euler_angles),
                                         degrees=degrees).as_quat()
        return np.array([w, x, y, z])

try:
    from isaacsim.core.prims import XFormPrim  # noqa: F401
except ImportError:
    import re

    import torch
    from pxr import Gf, UsdGeom

    import isaaclab.sim as sim_utils

    class XFormPrim:
        """The subset of Isaac Sim 5.1's XFormPrim this package uses, on plain USD."""

        def __init__(self, prim_paths_expr: str):
            stage = sim_utils.get_current_stage()
            rx = re.compile("^" + prim_paths_expr.replace("*", ".*") + "$") if "*" in prim_paths_expr else None
            if rx is None:
                p = stage.GetPrimAtPath(prim_paths_expr)
                self.prims = [p] if p.IsValid() else []
            else:
                self.prims = [p for p in stage.Traverse() if rx.match(p.GetPath().pathString)]

        @staticmethod
        def _orient_op(xf: UsdGeom.Xformable):
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    return op
            return xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

        def set_local_poses(self, orientations=None, translations=None):
            for i, p in enumerate(self.prims):
                xf = UsdGeom.Xformable(p)
                if orientations is not None:
                    w, x, y, z = (float(v) for v in torch.as_tensor(orientations)[i].tolist())
                    op = self._orient_op(xf)
                    q = Gf.Quatf(w, x, y, z) if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat \
                        else Gf.Quatd(w, x, y, z)
                    op.Set(q)
                if translations is not None:
                    UsdGeom.XformCommonAPI(xf).SetTranslate(Gf.Vec3d(*torch.as_tensor(translations)[i].tolist()))

        def get_local_poses(self):
            pos, quat = [], []
            for p in self.prims:
                m = UsdGeom.Xformable(p).GetLocalTransformation()
                t = m.ExtractTranslation()
                r = m.ExtractRotationQuat()
                pos.append([t[0], t[1], t[2]])
                quat.append([r.GetReal(), *r.GetImaginary()])
            return torch.tensor(pos, dtype=torch.float32), torch.tensor(quat, dtype=torch.float32)
