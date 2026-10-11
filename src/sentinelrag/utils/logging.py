#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging utilities.
"""

import logging


class Log:
    """Wrapper for logging module"""

    def __init__(self, log_file, file_level=logging.DEBUG, console_level=logging.INFO):
        self.log_file = log_file
        self.file_level = file_level
        self.console_level = console_level

    def get(self, name: str):
        """Get a configured logger"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()
        logger.propagate = False

        file_handler = logging.FileHandler(self.log_file, 'a')
        file_handler.setLevel(self.file_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)

        date_fmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(
            '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt=date_fmt
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
