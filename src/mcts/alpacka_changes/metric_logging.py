"""Metric logging."""

import numpy as np
from mrunner.helpers.client_helper import logger as neptune_logger


class StdoutLogger:
    """Logs to standard output."""

    @staticmethod
    def log_scalar(name, step, value):
        """Logs a scalar to stdout."""
        # Format:
        #      1 | accuracy:                   0.789
        #   1234 | loss:                      12.345
        print('{:>6} | {:32}{:>9.3f}'.format(step, name + ':', value))
        neptune_logger(name, value)

    @staticmethod
    def log_property(name, value):
        # Not supported in this logger.
        pass


_loggers = [StdoutLogger]


def register_logger(logger):
    """Adds a logger to log to."""
    _loggers.append(logger)


def log_scalar(name, step, value):
    """Logs a scalar to the loggers."""
    for logger in _loggers:
        logger.log_scalar(name, step, value)


def log_property(name, value):
    """Logs a property to the loggers."""
    for logger in _loggers:
        logger.log_property(name, value)


def log_scalar_metrics(prefix, step, metrics):
    for (name, value) in metrics.items():
        log_scalar(prefix + '/' + name, step, value)


def compute_scalar_statistics(x, prefix=None, with_min_and_max=False):
    """Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x (np.ndarray): Samples of the scalar to produce statistics for.
        prefix (str): Prefix to put before a statistic name, separated with
            an underscore.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.

    Returns:
        Dictionary with statistic names as keys (can be prefixed, see the prefix
        argument) and statistic values.
    """
    prefix = prefix + '_' if prefix else ''
    stats = {}

    stats[prefix + 'mean'] = np.mean(x)
    stats[prefix + 'std'] = np.std(x)
    if with_min_and_max:
        stats[prefix + 'min'] = np.min(x)
        stats[prefix + 'max'] = np.max(x)

    return stats
