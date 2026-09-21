#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
import numpy as np
import os
from PIL import Image as PILImage

# adapter仅在transformers 4.x（对应vLLM 4以前的版本）可用，见 mm/adapter/__init__.py。
# 不可用时其内部模块不会被引入，用AVAILABLE守卫条件import，避免collection阶段报错；
# else分支兜底赋None，防止pylint报possibly-used-before-assignment（此时用例整体skip，不会真正执行）。
from mm import adapter as _mm_adapter

if _mm_adapter.AVAILABLE:
    from mm import Image, ImageFormat, InternVL2PreProcessor
else:
    Image = ImageFormat = InternVL2PreProcessor = None


VALID_IMAGE_PATH = "./test/assets/dog_1920_1080.jpg"
DEVICE_CPU = "cpu"
RESIZE_SIZE = 448
MIN_RATIO_NUM = 1
MAX_RATIO_NUM = 12


@unittest.skipIf(
    not _mm_adapter.AVAILABLE,
    "adapter仅适用于vLLM 4以前的版本（transformers 4.x），当前环境不可用",
)
class Test_InternVL2_Preprocess(unittest.TestCase):
    def test_valid_image(self):
        os.chmod(VALID_IMAGE_PATH, 0o640)
        image = Image.open(VALID_IMAGE_PATH, DEVICE_CPU)
        internVL2PreProcessor = InternVL2PreProcessor()
        result = internVL2PreProcessor.preprocess_image(image, 448, MIN_RATIO_NUM, MAX_RATIO_NUM, True)
        self.assertEqual(result.shape, (9, 3, 448, 448))

    def test_invalid_min_max_num(self):
        os.chmod(VALID_IMAGE_PATH, 0o640)
        image = Image.open(VALID_IMAGE_PATH, DEVICE_CPU)
        internVL2PreProcessor = InternVL2PreProcessor()
        with self.assertRaises(ValueError):
            internVL2PreProcessor.preprocess_image(image, 448, MAX_RATIO_NUM, MIN_RATIO_NUM, True)

    def test_invalid_image(self):
        img = np.random.randint(0, 256, size=(8193, 8193, 3), dtype=np.uint8)
        pil_img = PILImage.fromarray(img)
        internVL2PreProcessor = InternVL2PreProcessor()
        with self.assertRaises(RuntimeError):
            internVL2PreProcessor.preprocess_image(pil_img, 448, MIN_RATIO_NUM, MAX_RATIO_NUM, True)


if __name__ == "__main__":
    unittest.main()
