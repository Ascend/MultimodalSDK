#!/usr/bin/python3
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
import pytest

from accdata.pipeline import Pipeline
from accdata import ops


@pytest.mark.smoke
class OpsExternalSourceTest(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)

    def test_str_name(self):
        with self.pipe:
            external_data = ops.external_source("external_data")
        self.pipe.build([external_data.spec], [external_data.output])

    def test_not_str_name(self):
        with self.assertRaises(TypeError):
            with self.pipe:
                external_data = ops.external_source(0)
            self.pipe.build([external_data.spec], [external_data.output])


if __name__ == "__main__":
    unittest.main()
