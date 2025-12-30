#!/usr/bin/env python3
# coding=utf-8

"""
-------------------------------------------------------------------------
 This file is part of the MultimodalSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MultimodalSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

          http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
Description: Generating Log Files.
Author: MindX SDK
Create: 2025
History: NA
"""

import os
import sys
import time
from xml.etree import ElementTree


def combine_xml(xml_path, output_path):
    total_tests = 0
    total_failures = 0
    total_disabled = 0
    total_errors = 0
    total_time = 0.0
    total_skips = 0

    test_suites = ElementTree.Element('testsuites')
    test_suites.attrib.update({
        "tests": str(total_tests),
        "failures": str(total_failures),
        "disabled": str(total_disabled),
        "errors": str(total_errors),
        "timestamp": "",
        "time": str(round(total_time, 4)),
        "name": "AllTests",
        "skipped": str(total_skips)
    })

    for path, _, file_list in os.walk(xml_path):
        for file in file_list:
            if not file.endswith(".xml"):
                continue
            file_path = os.path.join(path, file)
            try:
                temp_tree = ElementTree.parse(file_path)
                test_suite = temp_tree.getroot()
            except ElementTree.ParseError as e:
                print(f"[WARN] Failed to parse XML file: {file_path}, error: {e}")
                continue
            except Exception as e:
                print(f"[ERROR] Unexpected error while parsing {file_path}: {e}")
                continue

            try:
                if file.endswith("test_report.xml"):
                    total_time += float(test_suite.attrib.get("time", "0")) / 1000
                else:
                    total_time += float(test_suite.attrib.get("time", "0"))
                total_tests += int(test_suite.attrib.get("tests", "0"))
                total_disabled += int(test_suite.attrib.get("disabled", "0"))
                total_failures += int(test_suite.attrib.get("failures", "0"))
                total_errors += int(test_suite.attrib.get("errors", "0"))

                for test_case in test_suite.findall("testcase"):
                    if isinstance(test_case.find("skipped"), ElementTree.Element):
                        total_skips += 1

                for temp_node in ["system-out", "system-err"]:
                    for temp_test_suite_node in test_suite.findall(temp_node):
                        if len(str(temp_test_suite_node.text)) <= len(str(None)):
                            test_suite.remove(temp_test_suite_node)

                test_suites.append(test_suite)
            except Exception as e:
                print(f"[ERROR] Failed to process content of {file_path}: {e}")
                continue

    test_suites.set("time", str(round(total_time, 4)))
    test_suites.set("tests", str(total_tests))
    test_suites.set("disabled", str(total_disabled))
    test_suites.set("failures", str(total_failures))
    test_suites.set("errors", str(total_errors))
    test_suites.set("skipped", str(total_skips))
    test_suites.set("timestamp", str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    output_file = os.path.join(output_path, "test_detail.xml")
    try:
        tree = ElementTree.ElementTree(test_suites)
        tree.write(output_file, encoding='UTF-8', xml_declaration=True)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "generate output file:", output_file)
    except Exception as e:
        print(f"[ERROR] Failed to write output XML file: {e}")


if __name__ == "__main__":
    combine_xml(len(sys.argv) > 1 and sys.argv[1] or ".", len(sys.argv) > 2 and sys.argv[2] or ".")
