import unittest
import numpy as np
import torch

from mvdatasets.geometry.quaternions import (
    quats_multiply,
    quats_invert,
    quats_from_degs,
    quats_from_rads,
    rots_to_quats,
    quats_angular_distance,
    quats_to_rots,
)


def to_numpy(x):
    """Utility to convert either a torch.Tensor or np.ndarray to np.ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


class TestQuaternionFunctions(unittest.TestCase):

    def test_quats_multiply(self):
        for arr_fn, arr_name in [
            (torch.tensor, "torch.tensor"),
            (np.array, "np.array"),
        ]:
            with self.subTest(msg=f"Testing quats_multiply with {arr_name}"):
                # Simple test
                a = arr_fn([[0.0, 0.0, 0.0, 1.0]], dtype=float)
                b = arr_fn([[1.0, 0.0, 0.0, 0.0]], dtype=float)
                result = quats_multiply(a, b)
                expected = arr_fn([[1.0, 0.0, 0.0, 0.0]], dtype=float)

                self.assertTrue(
                    np.allclose(to_numpy(result), to_numpy(expected), atol=1e-4),
                    f"quats_multiply({arr_name}): {result} vs {expected}",
                )

                # Associativity check
                c = arr_fn([[0.0, 1.0, 0.0, 0.0]], dtype=float)
                result_abc = quats_multiply(quats_multiply(a, b), c)
                result_cba = quats_multiply(a, quats_multiply(b, c))

                self.assertTrue(
                    np.allclose(to_numpy(result_abc), to_numpy(result_cba), atol=1e-4),
                    "Quaternion multiplication should be associative",
                )

    def test_quats_invert(self):
        for arr_fn, arr_name in [
            (torch.tensor, "torch.tensor"),
            (np.array, "np.array"),
        ]:
            with self.subTest(msg=f"Testing quats_invert with {arr_name}"):
                q = arr_fn([[0.7071, 0.0, 0.0, 0.7071]], dtype=float)
                result = quats_invert(q)
                expected = arr_fn([[-0.7071, 0.0, 0.0, 0.7071]], dtype=float)

                self.assertTrue(
                    np.allclose(to_numpy(result), to_numpy(expected), atol=1e-4),
                    f"Inversion mismatch with {arr_name}",
                )

                # Double-inversion returns original
                double_inverted = quats_invert(result)
                self.assertTrue(
                    np.allclose(to_numpy(double_inverted), to_numpy(q), atol=1e-4),
                    "Double inversion should yield the original quaternion",
                )

                # (Optional) Check q * invert(q) = identity
                identity = arr_fn([[0.0, 0.0, 0.0, 1.0]], dtype=float)
                q_inv_mult = quats_multiply(q, quats_invert(q))
                self.assertTrue(
                    np.allclose(to_numpy(q_inv_mult), to_numpy(identity), atol=1e-4),
                    "q * invert(q) should yield identity quaternion",
                )

    def test_rots_to_quats(self):
        for arr_fn, arr_name in [
            (torch.tensor, "torch.tensor"),
            (np.array, "np.array"),
        ]:
            with self.subTest(msg=f"Testing rots_to_quats with {arr_name}"):
                # Identity rotation
                eye_3x3 = np.eye(3)[None]  # shape (1,3,3)
                rots = arr_fn(eye_3x3, dtype=float)
                result = rots_to_quats(rots)
                expected = arr_fn([[0.0, 0.0, 0.0, 1.0]], dtype=float)
                print(result, expected)
                self.assertTrue(
                    np.allclose(to_numpy(result), to_numpy(expected), atol=1e-4),
                    f"Identity rotation mismatch with {arr_name}",
                )

                # 90° around Z
                rot_z = np.array(
                    [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=float
                )
                rots = arr_fn(rot_z, dtype=float)
                result = rots_to_quats(rots)
                expected = arr_fn([[0.0, 0.0, 0.7071, 0.7071]], dtype=float)
                self.assertTrue(
                    np.allclose(to_numpy(result), to_numpy(expected), atol=1e-4),
                    f"Rotation around Z mismatch with {arr_name}",
                )

    def test_quats_angular_distance(self):
        for arr_fn, arr_name in [
            (torch.tensor, "torch.tensor"),
            (np.array, "np.array"),
        ]:
            with self.subTest(msg=f"Testing quats_angular_distance with {arr_name}"):
                # q1 = identity
                q1 = arr_fn([[0.0, 0.0, 0.0, 1.0]], dtype=float)
                # q2 = 90 deg about X
                q2 = arr_fn([[0.7071, 0.0, 0.0, 0.7071]], dtype=float)
                result = quats_angular_distance(q1, q2)
                expected = arr_fn([0.2929], dtype=float)
                self.assertTrue(
                    np.allclose(to_numpy(result), to_numpy(expected), atol=1e-4),
                    f"Angular distance mismatch with {arr_name}",
                )

                # Angular distance between identical quaternions -> 0.0
                result = quats_angular_distance(q1, q1)
                expected_zero = arr_fn([0.0], dtype=float)
                self.assertTrue(
                    np.allclose(to_numpy(result), to_numpy(expected_zero), atol=1e-4),
                    f"Distance of identical quaternions should be 0 with {arr_name}",
                )

    def test_quats_to_rots(self):
        for arr_fn, arr_name in [
            (torch.tensor, "torch.tensor"),
            (np.array, "np.array"),
        ]:
            with self.subTest(msg=f"Testing quats_to_rots with {arr_name}"):
                # Identity quaternion
                quats = arr_fn([[0.0, 0.0, 0.0, 1.0]], dtype=float)
                result = quats_to_rots(quats)
                expected = arr_fn(np.eye(3)[None], dtype=float)
                print(result, expected)
                self.assertTrue(
                    np.allclose(to_numpy(result), to_numpy(expected), atol=1e-4),
                    f"Identity quaternion -> rotation mismatch with {arr_name}",
                )

                # 90° rotation around Y
                quats = arr_fn([[0.0, 0.7071, 0.0, 0.7071]], dtype=float)
                result = quats_to_rots(quats)
                expected = np.array(
                    [[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=float
                )
                self.assertTrue(
                    np.allclose(to_numpy(result), expected, atol=1e-4),
                    f"90° rotation around Y mismatch with {arr_name}",
                )


if __name__ == "__main__":
    unittest.main()
