"""
BraiNets
========

Python codes for causal relationships using Gaussian copula and information
theory based tools.
"""
import logging

from .syslog import set_log_level

# Set 'info' as the default logging level
logger = logging.getLogger('brainets')
set_log_level('info')

__version__ = "0.0.0"
