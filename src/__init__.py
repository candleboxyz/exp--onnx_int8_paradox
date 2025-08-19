"""
Edge AI Optimization Toolkit
============================
A comprehensive toolkit for optimizing deep learning models for edge deployment.
"""

from .converter import ModelConverter
from .quantizer import ModelQuantizer
from .benchmarker import Benchmarker
from .analyzer import ModelAnalyzer
from .demo import EdgeYOLODemo

__version__ = "1.0.0"
__all__ = [
    "ModelConverter",
    "ModelQuantizer",
    "Benchmarker",
    "ModelAnalyzer",
    "EdgeYOLODemo",
]
