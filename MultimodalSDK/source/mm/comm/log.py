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
from enum import IntEnum
import inspect
import re
import sys
import types
from datetime import datetime
from typing import Callable, Optional
from ..acc._impl import acc as _acc


class LogLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    FATAL = 4


class _Logger:
    _min_level: LogLevel = LogLevel.INFO
    _TIMESTAMP_WIDTH = 3

    @staticmethod
    def _filter_invalid_chars(s: str) -> str:
        """_summary_ filter invalid chars in original log message

        Args:
            s (str): original log message

        Returns:
            str: _description_
        """
        invalid_chars = [
            '\n', '\f', '\r', '\b', '\t', '\v',
            '\u000D', '\u000A', '\u000C', '\u000B',
            '\u0009', '\u0008', '\u0007'
        ]
        pattern = '[' + re.escape(''.join(invalid_chars)) + ']+'
        return re.sub(pattern, ' ', s)

    @staticmethod
    def _default_log(level: LogLevel, message: str, file: str, line: int, function: str):
        """_summary_ default log output function

        Args:
            level (LogLevel): _description_ log level
            message (str): _description_ original log message
            file (str): _description_ log generation file
            line (int): _description_ log generation line
            function (str): _description_ log generation function
        """
        filtered_message = _Logger._filter_invalid_chars(message)
        # format default log
        now = datetime.utcnow()
        millis = int(now.microsecond / 1000)
        micros = now.microsecond % 1000
        millis_str = f"{millis:0{_Logger._TIMESTAMP_WIDTH}d}"
        micros_str = f"{micros:0{_Logger._TIMESTAMP_WIDTH}d}"
        log_str = (
            f"[{level.name}] "
            f"{now.strftime('%Y-%m-%d-%H:%M:%S')}.{millis_str}.{micros_str} "
            f"[{file}:{line}] {function}: {filtered_message}"
        )
        print(log_str)

    # current log callback
    _log_callback: Callable[[LogLevel, str, str, int, str], None] = _default_log

    @staticmethod
    def _acc_log_callback(level: int, message: bytes, file: bytes, line: int, function: bytes):
        """_summary_ AccSdk log callback

        Args:
            level (int): _description_ log level
            message (bytes): _description_ original log message
            file (bytes): _description_ log generation file
            line (int): _description_ log generation line
            function (bytes): _description_ log generation function
        """
        _Logger._log_callback(LogLevel(level), message.decode("utf-8"),
                              file.decode("utf-8"), line, function.decode("utf-8"))

    @staticmethod
    def _log(level: 'LogLevel', message: str):
        """_summary_ inner log output function

        Args:
            level (LogLevel): _description_ log level
            message (str): _description_ original log message
        """
        if level < _Logger._min_level:
            return
        frame = inspect.currentframe().f_back.f_back
        file = frame.f_code.co_filename
        line = frame.f_lineno
        func = frame.f_code.co_name
        _Logger._log_callback(level, message, file, line, func)

    @staticmethod
    def set_logger(callback: Callable[[LogLevel, str, str, int, str], None], min_level: 'LogLevel'):
        """_summary_ set log callback and min log level

        Args:
            callback (Callable[[LogLevel, str, str, int, str], None]): _description_ callback function
            min_level (LogLevel): _description_ min log level
        """
        _Logger._log_callback = callback or _Logger._default_log
        _Logger._min_level = min_level

    @staticmethod
    def debug(message: str):
        """_summary_ inner debug log function

        Args:
            message (str): _description_ original log message
        """
        _Logger._log(LogLevel.DEBUG, message)

    @staticmethod
    def info(message: str):
        """_summary_ inner info log function

        Args:
            message (str): _description_ original log message
        """
        _Logger._log(LogLevel.INFO, message)

    @staticmethod
    def warn(message: str):
        """_summary_ inner warn log function

        Args:
            message (str): _description_ original log message
        """
        _Logger._log(LogLevel.WARN, message)

    @staticmethod
    def error(message: str):
        """_summary_ inner error log function

        Args:
            message (str): _description_ original log message
        """
        _Logger._log(LogLevel.ERROR, message)

    @staticmethod
    def fatal(message: str):
        """_summary_ inner fatal log function

        Args:
            message (str): _description_ original log message
        """
        _Logger._log(LogLevel.FATAL, message)


class _LogCallBackerAux(_acc.LogCallBacker):
    """_summary_ acc log auxiliary callback class

    """
    def __init__(self):
        super().__init__()

    def log(self, level, msg, file, line, func):
        return


# global callback instance to ensure the lifetime
_acc_log_cb_instance_global = _LogCallBackerAux()


def register_log_conf(min_level: LogLevel, callback: Callable[[LogLevel, str, str, int, str], None]):
    """_summary_ register log callback and min log level

    Args:
        min_level (LogLevel): _description_ min log level, logs whose levels are higher than this value are output
        callback (Callable[[LogLevel, str, str, int, str], None], optional): _description_. call back function.
                                                                            Defaults to None.
    """
    if min_level is None:
        raise ValueError("min_level cannot be None")
    if not isinstance(min_level, LogLevel):
        raise TypeError("The parameter 'min_level' must be of LogLevel type.")
    _Logger.set_logger(callback, min_level)
    _acc_log_cb_instance_global.log = _Logger._acc_log_callback
    _acc.LogCallBacker.register_log_conf(min_level.value, _acc_log_cb_instance_global)
