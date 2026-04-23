"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


from .deim import DEIM

from .matcher import HungarianMatcher

from .dfine_decoder import DFINETransformer

from .postprocessor import PostProcessor, DQPostProcessor
from .deim_criterion import DEIMCriterion

from .deimv2_decoder import DEIMV2Transformer
from .hybrid_encoder_deimv2 import HybridEncoderV2

from .dinov3_teacher import DINOv3TeacherModel