#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
"""
Description: python log test.
Author: ACC SDK
Create: 2025
History: NA
"""
import ctypes
import logging
import os
import sys
import tempfile
import types
from accsdk_pytest import BaseTestCase
from acc import LogCallBacker, Image, LogLevel_INFO, LogLevel_FATAL


logger = logging.getLogger("test_logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

call_count = 0


class LogCallBackerAux(LogCallBacker):
    def __init__(self):
        super().__init__()

    def log(self, level, file, func, line, msg):
        return


def custom_log(this, level, msg, file, line, func):
    global call_count
    call_count += 1
    print("custom log:", level, file.decode("utf-8"),
          func.decode("utf-8"), line, msg.decode("utf-8"))
    logger.log(logging.ERROR, msg)


class TestPyLog(BaseTestCase):

    def setUp(self):
        self._cb = LogCallBackerAux()
        self._orig_stdout_fd = os.dup(sys.stdout.fileno())
        self._orig_stderr_fd = os.dup(sys.stderr.fileno())
        self._tmpfile = tempfile.TemporaryFile(mode="w+")
        os.dup2(self._tmpfile.fileno(), sys.stdout.fileno())
        os.dup2(self._tmpfile.fileno(), sys.stderr.fileno())

    def tearDown(self):
        os.dup2(self._orig_stdout_fd, sys.stdout.fileno())
        os.dup2(self._orig_stderr_fd, sys.stderr.fileno())
        os.close(self._orig_stdout_fd)
        os.close(self._orig_stderr_fd)
        print("Captured C++ output:\n", self._captured_output)

    def test_log_register_success_with_info_level(self):
        global call_count
        call_count = 0
        self._cb.log = types.MethodType(custom_log, self._cb)
        LogCallBacker.register_log_conf(LogLevel_INFO, self._cb)
        with self.assertRaises(Exception):
            image = Image.open(b"", b"cpu")
        self.assertEqual(call_count, 5)
        self._tmpfile.seek(0)
        self._captured_output = self._tmpfile.read()
        self._tmpfile.close()
        self.assertIn('custom log:', self._captured_output)

    def test_log_register_success_with_fatal_level(self):
        global call_count
        call_count = 0
        self._cb = LogCallBackerAux()
        self._cb.log = types.MethodType(custom_log, self._cb)
        LogCallBacker.register_log_conf(LogLevel_FATAL, self._cb)
        with self.assertRaises(Exception):
            image = Image.open(b"", b"cpu")
        self.assertEqual(call_count, 0)
        self._tmpfile.seek(0)
        self._captured_output = self._tmpfile.read()
        self._tmpfile.close()
        self.assertEqual(self._captured_output, "")

    def test_log_default_with_info_level(self):
        LogCallBacker.register_log_conf(LogLevel_INFO, None)
        with self.assertRaises(Exception):
            image = Image.open(b"", b"cpu")
        # need to flush c stdout
        libc = ctypes.CDLL(None)
        libc.fflush(None)
        self._tmpfile.seek(0)
        self._captured_output = self._tmpfile.read()
        self._tmpfile.close()
        self.assertNotIn('custom log:', self._captured_output)
        self.assertIn('Check file path failed. The path is empty.',
                      self._captured_output)

    def test_log_default_with_fatal_level(self):
        LogCallBacker.register_log_conf(LogLevel_FATAL, None)
        with self.assertRaises(Exception):
            image = Image.open(b"", b"cpu")
        libc = ctypes.CDLL(None)
        libc.fflush(None)
        self._tmpfile.seek(0)
        self._captured_output = self._tmpfile.read()
        self._tmpfile.close()
        self.assertNotIn('custom log:', self._captured_output)
        self.assertNotIn(
            'Check file path failed. The path is empty.', self._captured_output)


if __name__ == '__main__':
    failed = TestPyLog.run_tests()
    sys.exit(1 if failed > 0 else 0)
