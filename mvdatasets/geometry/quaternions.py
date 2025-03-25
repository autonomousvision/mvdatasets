import torch
import torch.nn.functional as F
import numpy as np
from typing import Union


def quats_rotate_vectors(q, v):
    """
    Rotate 3D vector v by quaternion q.
    q in (x, y, z, w).
    """

    # assert v and q are of the same type
    is_numpy = isinstance(q, np.ndarray)
    is_torch = isinstance(q, torch.Tensor)
    assert is_numpy == isinstance(v, np.ndarray), "v and q must be of the same type"
    assert is_torch == isinstance(v, torch.Tensor), "v and q must be of the same type"

    # Convert v into a 'pure quaternion' (vx, vy, vz, 0)
    if is_numpy:
        v_quat = np.concatenate([v, np.zeros_like(v[..., :1])], axis=-1)  # (..., 4)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    elif is_torch:
        v_quat = torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)  # (..., 4)
        q = F.normalize(q, p=2, dim=-1)
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")

    # rotated v = q * v_quat * q_inv
    q_inv = quats_invert(q)  # (..., 4)
    tmp = quats_multiply(q, v_quat)  # (..., 4)
    tmp = quats_multiply(tmp, q_inv)  # (..., 4)

    return tmp[..., :3]


def quats_multiply(q1, q2):
    """
    Multiply two quaternions q1, q2 (in [x,y,z,w] order) of shape (..., 4).

    Both q1 and q2 can be either:
    - torch.Tensor of shape (..., 4), or
    - np.ndarray of shape (..., 4).

    By default, each input quaternion is assumed to be normalized.
    """

    # Optional shape checks:
    if q1.shape != q2.shape:
        raise ValueError("q1 and q2 dimentions should match.")
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError("q1 and q2 last dimention is not 4.")

    is_numpy = isinstance(q1, np.ndarray)

    if is_numpy:

        # Ensure float32 for safety
        q1 = q1.astype(np.float32)
        q2 = q2.astype(np.float32)

        # Extract components (x,y,z,w)
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        # Hamilton product in (x,y,z,w) layout
        x_out = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y_out = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z_out = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        w_out = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        out = np.stack([x_out, y_out, z_out, w_out], axis=-1)

    else:

        # Extract components (x,y,z,w)
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        # Hamilton product in (x,y,z,w) layout
        x_out = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y_out = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z_out = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        w_out = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        out = torch.stack([x_out, y_out, z_out, w_out], dim=-1)

    # verify output shape
    assert (
        out.shape == q1.shape
    ), f"Output shape {out.shape} does not match input shape {q1.shape}"

    return out


def quats_invert(q: Union[torch.tensor, np.ndarray]) -> Union[torch.tensor, np.ndarray]:
    """
    Inverts a batch of quaternions q.
    Supports both PyTorch tensors and NumPy arrays.

    q: (N, 4) batch of quaternions (torch.Tensor or np.ndarray)
    Returns: (N, 4) batch of inverted quaternion (same type as input)
    """

    unsqueezed = False
    if q.ndim == 1:
        unsqueezed = True

    if isinstance(q, np.ndarray):

        if unsqueezed:
            q = q[np.newaxis, :]

        q = q.astype(np.float32)
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        q = q / norm

        # conjugate
        x = -q[..., 0]
        y = -q[..., 1]
        z = -q[..., 2]
        w = q[..., 3]
        inv_q = np.stack([x, y, z, w], axis=-1)

        inv_q = inv_q / np.linalg.norm(inv_q, axis=-1, keepdims=True)

    elif isinstance(q, torch.Tensor):

        if unsqueezed:
            q = q.unsqueeze(0)

        q = q / torch.linalg.norm(q, dim=-1, keepdim=True)

        x = -q[..., 0]
        y = -q[..., 1]
        z = -q[..., 2]
        w = q[..., 3]
        inv_q = torch.stack([x, y, z, w], dim=-1)

        inv_q = inv_q / torch.linalg.norm(inv_q, dim=-1, keepdim=True)

    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")

    if unsqueezed:
        inv_q = inv_q.squeeze(0)

    return inv_q


def quats_from_degs(deg_x, deg_y, deg_z):

    all_torch = all(isinstance(a, torch.Tensor) for a in [deg_x, deg_y, deg_z])
    all_numpy = all(isinstance(a, np.ndarray) for a in [deg_x, deg_y, deg_z])

    if all_torch:
        return quats_from_rads(
            torch.radians(deg_x), torch.radians(deg_y), torch.radians(deg_z)
        )
    elif all_numpy:
        return quats_from_rads(np.radians(deg_x), np.radians(deg_y), np.radians(deg_z))
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")


# def invert_rigid_transformation(quats, transls):
#     quats_inv = quats_invert(quats)  # (G, K, 4)
#     transls_padded = torch.cat([transls, torch.zeros_like(transls[..., :1])], dim=-1)
#     tmp = quats_multiply(quats_inv, transls_padded)
#     tmp = quats_multiply(tmp, quats)
#     transls_inv = -tmp[..., :3]

#     return quats_inv, transls_inv


# def quats_from_rads(rad_x, rad_y, rad_z, dtype=torch.float32, device="cpu"):
#     q = torch.tensor(
#         [
#             np.cos(rad_x / 2) * np.cos(rad_y / 2) * np.cos(rad_z / 2)
#             + np.sin(rad_x / 2) * np.sin(rad_y / 2) * np.sin(rad_z / 2),
#             np.sin(rad_x / 2) * np.cos(rad_y / 2) * np.cos(rad_z / 2)
#             - np.cos(rad_x / 2) * np.sin(rad_y / 2) * np.sin(rad_z / 2),
#             np.cos(rad_x / 2) * np.sin(rad_y / 2) * np.cos(rad_z / 2)
#             + np.sin(rad_x / 2) * np.cos(rad_y / 2) * np.sin(rad_z / 2),
#             np.cos(rad_x / 2) * np.cos(rad_y / 2) * np.sin(rad_z / 2)
#             - np.sin(rad_x / 2) * np.sin(rad_y / 2) * np.cos(rad_z / 2),
#         ],
#         dtype=dtype,
#         device=device,
#     )
#     # Reorder from [w, x, y, z] to [x, y, z, w]
#     q = q[..., [1, 2, 3, 0]]
#     # Normalize the quaternion
#     q = q / torch.linalg.norm(q)
#     return q


def quats_from_rads(rad_x, rad_y, rad_z):
    """
    Compute a quaternion [x, y, z, w] given three rotation angles in radians.

    The function supports two modes of operation:
      - If rad_x, rad_y, rad_z are all torch.Tensors, it uses PyTorch ops.
      - If rad_x, rad_y, rad_z are all NumPy arrays, it uses NumPy ops.

    Returns a quaternion in the same type (torch.Tensor or np.ndarray) as the inputs:
      - If all inputs are torch.Tensors, returns a torch.Tensor on the same device as rad_x.
      - If all inputs are np.ndarray, returns a np.ndarray.

    Args:
        rad_x, rad_y, rad_z:
            Rotation angles (either all torch.Tensor or all np.ndarray).
            Each must be either scalar or matching shapes if you want a batch dimension.
        dtype (torch.dtype, optional):
            Default dtype used if no type can be inferred from the inputs (only used
            if the inputs are Tensors but do not have a dtype, e.g., integer Tensors).
        device (str, optional):
            Default device if not inferred from the inputs (only used if the inputs
            are Tensors but do not have a known device).
    """

    # --- 1) Identify and check input types ---
    all_torch = all(isinstance(a, torch.Tensor) for a in [rad_x, rad_y, rad_z])
    all_numpy = all(isinstance(a, np.ndarray) for a in [rad_x, rad_y, rad_z])

    if not (all_torch or all_numpy):
        raise TypeError(
            "quats_from_rads requires that all three inputs are either torch.Tensors "
            "or all three are np.ndarrays. Mixed or unsupported types are not allowed."
        )

    # --- 2) If all inputs are Tensors, do everything in PyTorch ---
    if all_torch:
        # Move everything to the same device/dtype as rad_x
        # (Assumes rad_x, rad_y, rad_z are consistently typed.)
        dev = rad_x.device
        dt = rad_x.dtype

        # Convert them to float if needed
        rx = rad_x.to(dt, copy=False).to(dev)
        ry = rad_y.to(dt, copy=False).to(dev)
        rz = rad_z.to(dt, copy=False).to(dev)

        # We want to compute w, x, y, z in the [w, x, y, z] format
        # using torch.cos and torch.sin
        half_rx = rx / 2.0
        half_ry = ry / 2.0
        half_rz = rz / 2.0

        cosx = torch.cos(half_rx)
        cosy = torch.cos(half_ry)
        cosz = torch.cos(half_rz)
        sinx = torch.sin(half_rx)
        siny = torch.sin(half_ry)
        sinz = torch.sin(half_rz)

        # old w, x, y, z format
        w = cosx * cosy * cosz + sinx * siny * sinz
        x = sinx * cosy * cosz - cosx * siny * sinz
        y = cosx * siny * cosz + sinx * cosy * sinz
        z = cosx * cosy * sinz - sinx * siny * cosz

        # stack into shape (..., 4). If scalars, shape is (4,).
        # You can handle batch dims by broadcasting or ensuring rad_x, rad_y, rad_z have the same shape.
        q_wxyz = torch.stack([w, x, y, z], dim=-1)  # shape (...,4)

        # Reorder [w, x, y, z] -> [x, y, z, w]
        # This will keep the last dimension in the correct order
        # We'll gather along the last dimension
        q_xyzw = q_wxyz[..., [1, 2, 3, 0]]

        # Normalize
        # shape matching: norm over last dimension
        norms = torch.linalg.norm(q_xyzw, dim=-1, keepdim=True)
        q_xyzw = q_xyzw / norms

        return q_xyzw  # shape (...,4), torch.Tensor on same device/dtype as inputs

    # --- 3) If all inputs are NumPy, do everything in NumPy ---
    elif all_numpy:
        # Convert to float32 if needed
        rx = rad_x.astype(np.float32, copy=False)
        ry = rad_y.astype(np.float32, copy=False)
        rz = rad_z.astype(np.float32, copy=False)

        half_rx = rx / 2.0
        half_ry = ry / 2.0
        half_rz = rz / 2.0

        cosx = np.cos(half_rx)
        cosy = np.cos(half_ry)
        cosz = np.cos(half_rz)
        sinx = np.sin(half_rx)
        siny = np.sin(half_ry)
        sinz = np.sin(half_rz)

        w = cosx * cosy * cosz + sinx * siny * sinz
        x = sinx * cosy * cosz - cosx * siny * sinz
        y = cosx * siny * cosz + sinx * cosy * sinz
        z = cosx * cosy * sinz - sinx * siny * cosz

        # Stack into (N,4) if rad_x, rad_y, rad_z have shape (N,)
        # or just shape (4,) if they're scalars
        q_wxyz = np.stack([w, x, y, z], axis=-1)

        # Reorder [w, x, y, z] -> [x, y, z, w]
        q_xyzw = q_wxyz[..., [1, 2, 3, 0]]

        # Normalize over last dimension
        norms = np.linalg.norm(q_xyzw, axis=-1, keepdims=True)
        q_xyzw = q_xyzw / norms

        return q_xyzw  # shape (...,4), np.ndarray

    # If none of these branches matched (shouldn't happen due to the earlier check),
    # we raise an error:
    raise TypeError("Inputs must be either all torch.Tensors or all NumPy arrays.")


# def rots_to_quats(rots: torch.tensor) -> torch.tensor:
#     """
#     Convert rotations given as rotation matrices to quaternions.

#     Args:
#         matrix: Rotation matrices as tensor of shape (..., 3, 3).

#     Returns:
#         quaternions with real part first, as tensor of shape (..., 4).
#     """
#     if rots.size(-1) != 3 or rots.size(-2) != 3:
#         raise ValueError(f"Invalid rotation matrix shape {rots.shape}.")

#     batch_dim = rots.shape[:-2]
#     m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
#         rots.reshape(batch_dim + (9,)), dim=-1
#     )

#     def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
#         """
#         Returns torch.sqrt(torch.max(0, x))
#         but with a zero subgradient where x is 0.
#         """
#         ret = torch.zeros_like(x)
#         positive_mask = x > 0
#         ret[positive_mask] = torch.sqrt(x[positive_mask])
#         return ret

#     q_abs = _sqrt_positive_part(
#         torch.stack(
#             [
#                 1.0 + m00 + m11 + m22,
#                 1.0 + m00 - m11 - m22,
#                 1.0 - m00 + m11 - m22,
#                 1.0 - m00 - m11 + m22,
#             ],
#             dim=-1,
#         )
#     )

#     # we produce the desired quaternion multiplied by each of r, i, j, k
#     quat_by_rijk = torch.stack(
#         [
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
#         ],
#         dim=-2,
#     )

#     # We floor here at 0.1 but the exact level is not important; if q_abs is small,
#     # the candidate won't be picked.
#     flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
#     quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

#     # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
#     # forall i; we pick the best-conditioned one (with the largest denominator)

#     quats = quat_candidates[
#         torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
#     ].reshape(batch_dim + (4,))

#     # Reorder from [w, x, y, z] to [x, y, z, w]
#     quats = quats[..., [1, 2, 3, 0]]
#     return quats


def rots_to_quats(rots):
    """
    Convert rotation matrices to quaternions, supporting both Torch and NumPy inputs.

    If `rots` is a torch.Tensor of shape (..., 3, 3), returns a torch.Tensor of shape (..., 4),
    containing quaternions in [x, y, z, w] format.

    If `rots` is a np.ndarray of shape (..., 3, 3), returns a np.ndarray of shape (..., 4),
    also containing quaternions in [x, y, z, w] format.

    Raises:
        ValueError: if the last two dimensions are not (3, 3)
        TypeError:  if `rots` is neither a torch.Tensor nor a np.ndarray
    """

    # --- BRANCH A: TORCH INPUT ---
    if isinstance(rots, torch.Tensor):
        # Check shape
        if rots.size(-1) != 3 or rots.size(-2) != 3:
            raise ValueError(
                f"Invalid rotation matrix shape {rots.shape} for Torch tensor."
            )

        # Flatten the last two dimensions (3,3) into one dimension of size 9
        # and unbind to get each element.
        batch_dim = rots.shape[:-2]
        m = rots.reshape(*batch_dim, 9)
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(m, dim=-1)

        def _sqrt_positive_part_torch(x: torch.Tensor) -> torch.Tensor:
            """
            Returns sqrt of the positive part of x:
                result = torch.sqrt(torch.clamp(x, min=0))
            but done in a way that yields zero for negative or zero parts.
            """
            ret = torch.zeros_like(x)
            mask = x > 0
            ret[mask] = torch.sqrt(x[mask])
            return ret

        # q_abs holds the absolute value of each candidate’s w-component
        # (or x-, y-, z-component) so we can pick the best-conditioned candidate.
        q_abs = _sqrt_positive_part_torch(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,  # candidate: w
                    1.0 + m00 - m11 - m22,  # candidate: x
                    1.0 - m00 + m11 - m22,  # candidate: y
                    1.0 - m00 - m11 + m22,  # candidate: z
                ],
                dim=-1,
            )
        )

        # Build candidate quaternions (in [w, x, y, z]) for each “pivot” (r, i, j, k)
        quat_by_rijk = torch.stack(
            [
                torch.stack(
                    [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
                ),
                torch.stack(
                    [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
                ),
                torch.stack(
                    [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
                ),
                torch.stack(
                    [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
                ),
            ],
            dim=-2,
        )  # shape (..., 4, 4)

        # Prevent division by zero for tiny denominators
        flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
        denom = 2.0 * torch.max(q_abs[..., None], flr)  # shape (..., 4, 1)
        quat_candidates = quat_by_rijk / denom  # shape (..., 4, 4)

        # For each batch, pick whichever candidate had the largest q_abs
        # (i.e., w/x/y/z pivot). We use one_hot to pick that row out of the 4.
        pivot_idx = q_abs.argmax(dim=-1)  # shape (...,)
        pivot_mask = torch.nn.functional.one_hot(pivot_idx, num_classes=4) > 0.5
        # pivot_mask has shape (..., 4). We broadcast along the last dimension of size 4.
        quats = quat_candidates[pivot_mask]  # shape (... * 4,)
        quats = quats.reshape(*batch_dim, 4)  # shape (..., 4)

        # Reorder from [w, x, y, z] -> [x, y, z, w]
        quats = quats[..., [1, 2, 3, 0]]
        return quats

    # --- BRANCH B: NUMPY INPUT ---
    elif isinstance(rots, np.ndarray):
        if rots.shape[-1] != 3 or rots.shape[-2] != 3:
            raise ValueError(
                f"Invalid rotation matrix shape {rots.shape} for NumPy array."
            )

        # Flatten the last two dimensions (3,3) into one dimension of size 9
        # and split to get each element.
        batch_dim = rots.shape[:-2]
        reshaped = rots.reshape(*batch_dim, 9)
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = [
            reshaped[..., i] for i in range(9)
        ]

        def _sqrt_positive_part_np(x: np.ndarray) -> np.ndarray:
            """
            Returns sqrt of the positive part of x.
            Negative or zero values become 0.
            """
            ret = np.zeros_like(x)
            mask = x > 0
            ret[mask] = np.sqrt(x[mask])
            return ret

        q_abs = _sqrt_positive_part_np(
            np.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                axis=-1,
            )
        )

        # Build candidate quaternions in [w, x, y, z]
        quat_by_rijk = np.stack(
            [
                np.stack(
                    [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1
                ),
                np.stack(
                    [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1
                ),
                np.stack(
                    [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1
                ),
                np.stack(
                    [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1
                ),
            ],
            axis=-2,
        )  # shape (..., 4, 4)

        flr = 0.1
        # shape of q_abs is (..., 4), so q_abs[..., None] is (..., 4, 1).
        denom = 2.0 * np.maximum(q_abs[..., None], flr)  # shape (..., 4, 1)
        quat_candidates = quat_by_rijk / denom  # shape (..., 4, 4)

        # pick the best pivot index
        pivot_idx = np.argmax(q_abs, axis=-1)  # shape (...,)

        # gather along the penultimate dimension (which is size 4).
        # Need to expand pivot_idx to match shape.
        # shape(...,) -> shape(...,1) -> shape(...,1,1) -> broadcast to (...,1,4)
        pivot_idx_expanded = np.expand_dims(pivot_idx, axis=-1)  # (...,1)
        pivot_idx_expanded = np.expand_dims(pivot_idx_expanded, axis=-1)  # (...,1,1)
        pivot_idx_expanded = np.broadcast_to(
            pivot_idx_expanded, quat_candidates.shape[:-2] + (1, 4)
        )

        quats = np.take_along_axis(quat_candidates, pivot_idx_expanded, axis=-2)
        # now shape is (...,1,4). Squeeze out the penultimate dimension
        quats = np.squeeze(quats, axis=-2)  # shape (...,4)

        # Reorder from [w, x, y, z] -> [x, y, z, w]
        quats = quats[..., [1, 2, 3, 0]]
        return quats

    # --- ERROR FOR OTHER TYPES ---
    else:
        raise TypeError("rots_to_quats only supports torch.Tensor or np.ndarray.")


# def quats_angular_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
#     """compute row-wise angular distance between two batches of quaternions

#     Args:
#         q1 (torch.tensor): (N, 4) batch of quaternions
#         q2 (torch.tensor): (N, 4) batch of quaternions
#     Output:
#         theta (torch.tensor): (N,) batch of angular distances
#     """

#     # Ensure quaternions are normalized
#     q1 = q1 / torch.linalg.norm(q1, dim=1, keepdim=True)
#     q2 = q2 / torch.linalg.norm(q2, dim=1, keepdim=True)

#     # theta = 2 * arccos(|dot(q1, q2)|)
#     dot_product = torch.sum(q1 * q2, dim=-1)
#     # theta = 2 * torch.acos(torch.abs(dot_product))
#     # simplify to improve efficiency
#     theta = 1 - torch.abs(dot_product)

#     return theta


def quats_angular_distance(q1, q2):
    """
    Compute row-wise angular distance between two batches of quaternions.

    The function supports either torch.Tensors or NumPy arrays.
    - If both inputs are torch.Tensors, a torch.Tensor is returned.
    - If both inputs are NumPy arrays, a NumPy array is returned.
    - If the inputs are mixed types or some other type, a TypeError is raised.

    Args:
        q1: (N, 4) batch of quaternions (torch.Tensor or np.ndarray)
        q2: (N, 4) batch of quaternions (torch.Tensor or np.ndarray)

    Returns:
        theta: (N,) array/tensor of angular distances in the same type as inputs.
    """

    # Branch: both are PyTorch tensors
    if isinstance(q1, torch.Tensor) and isinstance(q2, torch.Tensor):
        # Ensure float dtype
        q1 = q1.float()
        q2 = q2.float()

        # Normalize quaternions
        q1 = q1 / torch.linalg.norm(q1, dim=1, keepdim=True)
        q2 = q2 / torch.linalg.norm(q2, dim=1, keepdim=True)

        # Compute 1 - |dot(q1, q2)|
        dot_product = torch.sum(q1 * q2, dim=-1)
        theta = 1 - torch.abs(dot_product)
        return theta

    # Branch: both are NumPy arrays
    elif isinstance(q1, np.ndarray) and isinstance(q2, np.ndarray):
        # Ensure float dtype
        q1 = q1.astype(np.float32)
        q2 = q2.astype(np.float32)

        # Normalize quaternions
        norms_q1 = np.linalg.norm(q1, axis=1, keepdims=True)
        norms_q2 = np.linalg.norm(q2, axis=1, keepdims=True)
        q1 /= norms_q1
        q2 /= norms_q2

        # Compute 1 - |dot(q1, q2)|
        dot_product = np.sum(q1 * q2, axis=-1)
        theta = 1 - np.abs(dot_product)
        return theta

    # If types do not match or are not recognized
    else:
        raise TypeError(
            "quats_angular_distance only supports either both inputs as "
            "torch.Tensors or both as NumPy arrays."
        )


def quats_to_rots(
    quats: Union[torch.tensor, np.ndarray]
) -> Union[torch.tensor, np.ndarray]:

    unsqueezed = False
    if quats.ndim == 1:
        unsqueezed = True

    # Determine whether the input is a torch tensor or numpy array
    if isinstance(quats, torch.Tensor):

        if unsqueezed:
            quats = quats.unsqueeze(0)

        # Reorder from [x, y, z, w] to [w, x, y, z]
        quats = quats[..., [3, 0, 1, 2]]
        r, i, j, k = torch.unbind(quats, -1)
        two_s = 2.0 / (quats * quats).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        rots = o.view(quats.shape[:-1] + (3, 3))

    elif isinstance(quats, np.ndarray):

        if unsqueezed:
            quats = quats[np.newaxis, :]

        # Reorder from [x, y, z, w] to [w, x, y, z]
        quats = quats[..., [3, 0, 1, 2]]
        r, i, j, k = np.split(quats, 4, axis=-1)
        two_s = 2.0 / (quats * quats).sum(axis=-1, keepdims=True)
        o = np.concatenate(
            [
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ],
            axis=-1,
        )
        shape = quats.shape[:-1] + (3, 3)
        rots = o.reshape(shape)
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")

    if unsqueezed:
        rots = rots.squeeze(0)

    return rots
