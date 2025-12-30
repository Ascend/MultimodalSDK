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
import ctypes
import os
import sys
import tempfile
import unittest
from mm import register_log_conf, LogLevel
import mm


def custom_log_handler(level: LogLevel, file: str, func: str, line: int, msg: str) -> None:
    print(f"[custom][{level}] {file}:{line} ({func}) - {msg}")


class TestLog(unittest.TestCase):
    def setUp(self):
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

    def read_output(self):
        sys.stdout.flush()
        self._tmpfile.flush()
        self._tmpfile.seek(0)
        self._captured_output = self._tmpfile.read()
        self._tmpfile.close()

    def test_log_default_success_with_info_level(self):
        mm.comm.log._Logger.error("ERROR log")
        self.read_output()
        self.assertIn('ERROR log', self._captured_output)

    def test_log_default_filter_invalid_char(self):
        mm.comm.log._Logger.error(
            "Hello\n\tWorld\b!\r\nThis\f is\v a test\u0007.")
        self.read_output()
        self.assertIn('Hello World ! This  is  a test .',
                      self._captured_output)

    def test_log_default_filter_with_lower_level(self):
        register_log_conf(LogLevel.ERROR, None)
        mm.comm.log._Logger.error("ERROR log")
        mm.comm.log._Logger.warn("WARN log")
        mm.comm.log._Logger.info("INFO log")
        mm.comm.log._Logger.debug("DEBUG log")
        mm.comm.log._Logger.fatal("FATAL log")
        self.read_output()
        self.assertIn('ERROR log', self._captured_output)
        self.assertIn('FATAL log', self._captured_output)
        self.assertNotIn('INFO log', self._captured_output)
        self.assertNotIn('WARN log', self._captured_output)
        self.assertNotIn('DEBUG log', self._captured_output)

    def test_register_log_success_with_lower_level(self):
        register_log_conf(LogLevel.ERROR, custom_log_handler)
        mm.comm.log._Logger.error("ERROR log")
        mm.comm.log._Logger.warn("WARN log")
        mm.comm.log._Logger.info("INFO log")
        mm.comm.log._Logger.debug("DEBUG log")
        mm.comm.log._Logger.fatal("FATAL log")
        self.read_output()
        self.assertIn('ERROR log', self._captured_output)
        self.assertIn('FATAL log', self._captured_output)
        self.assertNotIn('INFO log', self._captured_output)
        self.assertNotIn('WARN log', self._captured_output)
        self.assertNotIn('DEBUG log', self._captured_output)
        self.assertEqual(self._captured_output.count("[custom]"), 2)

    def test_defaule_log_success_with_acc_error(self):
        register_log_conf(LogLevel.INFO, None)
        with self.assertRaises(Exception):
            image = mm.Image.open(b"", b"cpu")
        libc = ctypes.CDLL(None)
        libc.fflush(None)
        self.read_output()
        self.assertNotIn('[custom]', self._captured_output)
        self.assertIn('Check file path failed. The path is empty.',
                      self._captured_output)

    def test_register_log_success_with_acc_error(self):
        register_log_conf(LogLevel.INFO, custom_log_handler)
        with self.assertRaises(Exception):
            image = mm.Image.open(b"", b"cpu")
        libc = ctypes.CDLL(None)
        libc.fflush(None)
        self.read_output()
        self.assertEqual(self._captured_output.count("[custom]"), 5)
        self.assertIn('Check file path failed. The path is empty.',
                      self._captured_output)

    def test_register_log_fail_with_none_level(self):
        with self.assertRaises(Exception):
            register_log_conf(None, custom_log_handler)
        self.read_output()

    def test_register_log_fail_with_none_level(self):
        with self.assertRaises(Exception):
            register_log_conf(1, custom_log_handler)
        self.read_output()

if __name__ == '__main__':
    unittest.main()
