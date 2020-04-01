#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def trains_stop_stdout_monitor():
    global TRAINS_STD_LOGGER
    try:
        from trains.logger import StdStreamPatch
        if "TRAINS_STD_LOGGER" not in globals() and StdStreamPatch._stdout_proxy is not None:
            TRAINS_STD_LOGGER = StdStreamPatch._stdout_proxy._log
        StdStreamPatch.remove_std_logger()
    except ImportError:
        pass


def trains_restore_stdout_monitor():
    if "TRAINS_STD_LOGGER" in globals():
        try:
            from trains.logger import StdStreamPatch
            StdStreamPatch.patch_std_streams(TRAINS_STD_LOGGER)
        except ImportError:
            pass

def trains_log_text(text):
    logger = None
    if "TRAINS_STD_LOGGER" in globals():
        logger = TRAINS_STD_LOGGER
    else:
        try:
            from trains import Task
            logger = Task.current_task().get_logger()
        except:
            pass
    if logger is None:
        return
    logger.report_text(text)
    logger.flush()

