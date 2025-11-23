"""
Models module para Opción 1: Fine-tuning Incremental Simple
"""

from .text_encoder import TextEncoder
from .multimodal_fusion import MultimodalFusionModule, SimpleFusionModule
from .deimv2_multimodal import DEIMv2Multimodal, build_multimodal_model

__all__ = [
    'TextEncoder',
    'MultimodalFusionModule',
    'SimpleFusionModule',
    'DEIMv2Multimodal',
    'build_multimodal_model'
]