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
Description: MultimodalSDK Python Unit Test Base Class
Author: MultimodalSDK
Create: 2025
History: NA
"""

import inspect
import os
import time
import datetime
import unittest
import xml.etree.ElementTree as ET
from xml.dom import minidom


class GTestLikeResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_cases = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_cases.append((test, "success", None))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.test_cases.append((test, "failure", self._exc_info_to_string(err, test)))

    def addError(self, test, err):
        super().addError(test, err)
        self.test_cases.append((test, "error", self._exc_info_to_string(err, test)))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_cases.append((test, "skipped", reason))


class BaseTestCase(unittest.TestCase):

    @classmethod
    def run_tests(cls):
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()

        suite.addTests(loader.loadTestsFromTestCase(cls))
        runner = unittest.TextTestRunner(resultclass=GTestLikeResult, verbosity=1)
        start_time = time.time()
        result = runner.run(suite)
        duration = time.time() - start_time
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec='seconds')
        return cls._generate_xml(result, duration, timestamp)

    @classmethod
    def _generate_xml(cls, result, duration, timestamp):
        file_path = inspect.getfile(cls)
        file_name = os.path.basename(file_path).replace(".py", "")
        output_dir = os.path.join(os.path.dirname(__file__), "test-reports")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{file_name}.xml")
        testsuites_elem = ET.Element("testsuites", {
            "tests": str(result.testsRun),
            "failures": str(len(result.failures)),
            "disabled": "0",
            "errors": str(len(result.errors)),
            "time": "%.3f" % duration,
            "timestamp": timestamp,
            "name": "AllTests"
        })
        testsuite_elem = ET.Element("testsuite", {
            "name": file_name,
            "tests": str(result.testsRun),
            "failures": str(len(result.failures)),
            "disabled": "0",
            "errors": str(len(result.errors)),
            "time": "%.3f" % duration,
            "timestamp": timestamp
        })
        for test, status, message in result.test_cases:
            test_id = test.id()
            parts = test_id.split('.')
            method_name = parts[-1] if len(parts) >= 1 else ""
            testcase_elem = ET.Element("testcase", {
                "classname": test.__class__.__name__,
                "name": method_name,
                "status": "run",
                "result": "completed" if status == "success" else status,
                "time": "0",
                "timestamp": timestamp
            })
            if status == "failure":
                failure_elem = ET.SubElement(testcase_elem, "failure", {"message": "failure"})
                failure_elem.text = message
            elif status == "error":
                error_elem = ET.SubElement(testcase_elem, "error", {"message": "error"})
                error_elem.text = message
            elif status == "skipped":
                ET.SubElement(testcase_elem, "skipped", {"message": message})
            testsuite_elem.append(testcase_elem)
        testsuites_elem.append(testsuite_elem)
        cls._write_pretty_xml(testsuites_elem, output_file)
        return len(result.errors) + len(result.failures)

    @classmethod
    def _write_pretty_xml(cls, elem, output_file):
        xml_bytes = ET.tostring(elem, encoding="utf-8", xml_declaration=True)
        parsed = minidom.parseString(xml_bytes)
        pretty_xml = parsed.toprettyxml(indent="  ")
        with open(output_file, "w", encoding="utf-8") as f:
            for line in pretty_xml.splitlines():
                if line.strip():
                    f.write(f"{line}\n")

