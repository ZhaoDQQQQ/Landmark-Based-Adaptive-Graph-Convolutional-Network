# Copyright (c) OpenMMLab. All rights reserved.
from .recognizergcn import RecognizerGCN
from .recognizer_gcn_multiflow import RecognizerGCNMultiflow
from .recognizer_gcn_single_flow import RecognizerGCNSingleFlow

__all__ = ['RecognizerGCN', 'RecognizerGCNMultiflow', "RecognizerGCNSingleFlow"]
