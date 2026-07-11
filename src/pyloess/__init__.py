"""
pyloess - A Python re-implementation of R's loess() local regression smoother.
"""

from .loess import Loess, loess

__all__ = ["Loess", "loess"]
__version__ = "0.1.0"
