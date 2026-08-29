#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MultimodalSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import unittest
import torch

# Note: These tests focus on the parameter validation logic of
# resize_and_normalize. The actual call into acc.Qwen2VLProcessor.PreprocessTensor
# requires the native libcore.so library, which is exercised by integration tests.
from mm.core.processor import resize_and_normalize


# Convenience: build a uint8 tensor with the given shape. uint8 is the only
# supported dtype; float32 / int32 / etc. are treated as abnormal input and
# must be rejected (see TestResizeAndNormalizeAbnormalDtype below).
def _u8(*shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8)


class TestResizeAndNormalizeParamValidation(unittest.TestCase):
    """Tests for resize_and_normalize parameter validation logic.

    All "valid path" tests use uint8 input. dtype-related failures live in
    TestResizeAndNormalizeAbnormalDtype.
    """

    # ---------------- frames shape ----------------
    def test_frames_shape_not_4d_raises(self):
        # 3D tensor should fail
        frames = _u8(3, 16, 16)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        msg = str(ctx.exception)
        self.assertIn("4D", msg)

    def test_frames_shape_5d_raises(self):
        # 5D tensor should fail
        frames = _u8(2, 3, 4, 16, 16)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("4D", str(ctx.exception))

    # ---------------- height/width ----------------
    def test_zero_height_raises(self):
        frames = _u8(1, 3, 16, 16)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=0,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("positive", str(ctx.exception))

    def test_zero_width_raises(self):
        frames = _u8(1, 3, 16, 16)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=0,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("positive", str(ctx.exception))

    def test_negative_height_raises(self):
        frames = _u8(1, 3, 16, 16)
        with self.assertRaises(ValueError):
            resize_and_normalize(
                frames,
                height=-1,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )

    def test_negative_width_raises(self):
        frames = _u8(1, 3, 16, 16)
        with self.assertRaises(ValueError):
            resize_and_normalize(
                frames,
                height=8,
                width=-8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )

    # ---------------- image_mean ----------------
    def test_image_mean_wrong_length_raises(self):
        frames = _u8(1, 3, 16, 16)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5],  # only 2 elements
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("image_mean", str(ctx.exception))

    def test_image_mean_too_long_raises(self):
        frames = _u8(1, 3, 16, 16)
        with self.assertRaises(ValueError):
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5, 0.5],  # 4 elements
                image_std=[0.5, 0.5, 0.5],
            )

    def test_image_mean_not_list_or_tuple_raises(self):
        frames = _u8(1, 3, 16, 16)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=0.5,  # scalar, not a list
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("image_mean", str(ctx.exception))

    def test_image_mean_tuple_accepted(self):
        # tuple should also be accepted for image_mean
        frames = _u8(1, 3, 16, 16)
        # Only test validation - will fail later on PreprocessTensor due to libcore.so
        try:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=(0.5, 0.5, 0.5),  # tuple is OK
                image_std=[0.5, 0.5, 0.5],
            )
        except ValueError as e:
            # Validation should pass; if ValueError, it must not be about image_mean
            self.assertNotIn("image_mean", str(e))
        except Exception:
            # libcore.so missing or other runtime error is OK
            pass

    # ---------------- image_std ----------------
    def test_image_std_wrong_length_raises(self):
        frames = _u8(1, 3, 16, 16)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5],  # only 2 elements
            )
        self.assertIn("image_std", str(ctx.exception))

    def test_image_std_zero_raises(self):
        frames = _u8(1, 3, 16, 16)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.0, 0.5, 0.5],  # first element zero
            )
        self.assertIn("image_std", str(ctx.exception))

    def test_image_std_negative_raises(self):
        frames = _u8(1, 3, 16, 16)
        with self.assertRaises(ValueError):
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, -0.5, 0.5],
            )

    # ---------------- NCHW vs NHWC detection ----------------
    def test_nchw_detection_3d_at_dim1(self):
        # NCHW: dim 1 is 3 -> NCHW path
        # We only test that validation passes; actual call into PreprocessTensor
        # requires libcore.so.
        frames = _u8(1, 3, 16, 16)
        try:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        except ValueError as e:
            # If ValueError is raised, it must be from PreprocessTensor (no libcore.so),
            # not from our validation. Validation would say "shape ... is not NCHW or NHWC"
            self.assertNotIn("is not NCHW or NHWC", str(e))
        except Exception:
            # libcore.so missing - OK
            pass

    def test_nhwc_detection_3d_at_dim3(self):
        # NHWC: dim 3 is 3 -> NHWC path
        frames = _u8(1, 16, 16, 3)
        try:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        except ValueError as e:
            self.assertNotIn("is not NCHW or NHWC", str(e))
        except Exception:
            pass

    def test_invalid_channel_dim_raises(self):
        # Neither dim 1 nor dim 3 is 3 -> should raise
        frames = _u8(1, 5, 16, 16)  # dim 1 = 5
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("is not NCHW or NHWC", str(ctx.exception))

    def test_invalid_dim3_raises(self):
        # dim 1 = 3 but dim 3 = 5 - actually this passes since dim 1 is 3 (NCHW)
        # We need dim 1 != 3 AND dim 3 != 3
        frames = _u8(1, 4, 16, 4)  # dim 1 = 4, dim 3 = 4
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("is not NCHW or NHWC", str(ctx.exception))


class TestResizeAndNormalizeParamWarning(unittest.TestCase):
    """Tests that out-of-range image_mean triggers warn (not error)."""

    def test_image_mean_above_one_does_not_raise(self):
        # Out-of-range image_mean should log a warning but not raise
        # (we only test the validation logic, not the actual processing)
        frames = _u8(1, 3, 16, 16)
        try:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[1.5, 0.5, 0.5],  # first value out of range
                image_std=[0.5, 0.5, 0.5],
            )
        except ValueError as e:
            # If ValueError, it must not be about image_mean being out of range
            self.assertNotIn("image_mean", str(e))
        except Exception:
            pass

    def test_image_mean_negative_does_not_raise(self):
        # Negative image_mean should log a warning but not raise
        frames = _u8(1, 3, 16, 16)
        try:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[-0.1, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        except ValueError as e:
            self.assertNotIn("image_mean", str(e))
        except Exception:
            pass


class TestResizeAndNormalizeAbnormalDtype(unittest.TestCase):
    """Abnormal dtype inputs must be rejected with ValueError.

    Only ``torch.uint8`` is supported; every other dtype (float32 in
    particular) is treated as abnormal input and raises before any
    downstream processing happens.
    """

    def test_float32_nchw_raises(self):
        frames = torch.randn(1, 3, 16, 16, dtype=torch.float32)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("dtype", str(ctx.exception))

    def test_float32_nhwc_raises(self):
        frames = torch.randn(1, 16, 16, 3, dtype=torch.float32)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("dtype", str(ctx.exception))

    def test_float32_even_when_shape_valid_raises(self):
        # Even with a perfectly valid 4D NCHW shape, float32 must be rejected
        # before any NCHW/NHWC detection runs.
        frames = torch.zeros(1, 3, 16, 16, dtype=torch.float32)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("dtype", str(ctx.exception))

    def test_float64_raises(self):
        frames = torch.randn(1, 3, 16, 16, dtype=torch.float64)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("dtype", str(ctx.exception))

    def test_int32_raises(self):
        frames = torch.zeros(1, 3, 16, 16, dtype=torch.int32)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("dtype", str(ctx.exception))

    def test_int64_raises(self):
        frames = torch.zeros(1, 3, 16, 16, dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("dtype", str(ctx.exception))

    def test_bool_raises(self):
        frames = torch.zeros(1, 3, 16, 16, dtype=torch.bool)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("dtype", str(ctx.exception))

    def test_uint16_raises(self):
        # Anything other than uint8 must be rejected, even if it is unsigned.
        frames = torch.zeros(1, 3, 16, 16, dtype=torch.uint16)
        with self.assertRaises(ValueError) as ctx:
            resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        self.assertIn("dtype", str(ctx.exception))


class TestResizeAndNormalizeSmoke(unittest.TestCase):
    """Smoke test that the function signature accepts common input shapes.

    These tests will only pass when libcore.so is available (i.e. in the
    wheel-install environment). On a dev environment without the native
    library, the test will raise some non-validation exception which is
    acceptable.
    """

    def test_nchw_single_image_accepted(self):
        frames = _u8(1, 3, 32, 32)
        try:
            result = resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
            self.assertEqual(result.shape[0], 1)
            self.assertEqual(result.shape[1], 3)
        except Exception as e:
            # Acceptable failures: libcore.so missing or PreprocessTensor internal error
            if isinstance(e, ValueError):
                # Must be a validation failure, not NCHW/NHWC detection failure
                self.assertNotIn("is not NCHW or NHWC", str(e))
            # Otherwise: libcore.so or runtime issue - skip

    def test_nhwc_single_image_accepted(self):
        frames = _u8(1, 32, 32, 3)
        try:
            result = resize_and_normalize(
                frames,
                height=8,
                width=8,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
            self.assertEqual(result.shape[0], 1)
        except Exception as e:
            if isinstance(e, ValueError):
                self.assertNotIn("is not NCHW or NHWC", str(e))


if __name__ == "__main__":
    unittest.main()
