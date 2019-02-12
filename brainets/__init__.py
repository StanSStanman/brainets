"""
BraiNets
========

Python codes for causal relationships using Gaussian copula and information
theory based tools.
"""
import logging

from brainets import (behavior, gcmi, infodyn, spectral, stats, syslog, utils,  # noqa
                      preprocessing)

# Set 'info' as the default logging level
logger = logging.getLogger('brainets')
syslog.set_log_level('info')

__version__ = "0.0.0"
