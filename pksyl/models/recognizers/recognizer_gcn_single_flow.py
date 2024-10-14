import numpy as np
import torch

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class RecognizerGCNSingleFlow(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition. """
    def __init__(self, backbone, cls_head=None, softmax=False, train_cfg=dict(), test_cfg=dict()):
        super().__init__(backbone, cls_head, train_cfg, test_cfg)
        self.softmax = softmax
        if softmax:
            self.softmax_layer = torch.nn.Softmax(dim=1)

    def forward_train(self, keypoint, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]

        # losses = dict()
        x = self.extract_feat(keypoint)
        cls_score = self.cls_head(x)
        # gt_label = label.squeeze(-1)
        # loss = self.cls_head.loss(cls_score, gt_label)
        # losses.update(loss)
        if self.softmax:
            cls_score = self.softmax_layer(cls_score)

        return cls_score

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert self.with_cls_head or self.feat_ext
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])

        x = self.extract_feat(keypoint)
        feat_ext = self.test_cfg.get('feat_ext', False)
        pool_opt = self.test_cfg.get('pool_opt', 'all')
        score_ext = self.test_cfg.get('score_ext', False)
        if feat_ext or score_ext:
            assert bs == 1
            assert isinstance(pool_opt, str)
            dim_idx = dict(n=0, m=1, t=3, v=4)

            if pool_opt == 'all':
                pool_opt == 'nmtv'
            if pool_opt != 'none':
                for digit in pool_opt:
                    assert digit in dim_idx

            if isinstance(x, tuple) or isinstance(x, list):
                x = torch.cat(x, dim=2)
            assert len(x.shape) == 5, 'The shape is N, M, C, T, V'
            if pool_opt != 'none':
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)

            if score_ext:
                w = self.cls_head.fc_cls.weight
                b = self.cls_head.fc_cls.bias
                x = torch.einsum('nmctv,oc->nmotv', x, w)
                if b is not None:
                    x = x + b[..., None, None]
                x = x[None]
            if self.softmax:
                x = self.softmax_layer(x)
            # return x
            return x.data.cpu().numpy().astype(np.float16)

        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        # harmless patch
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'

        cls_score = self.average_clip(cls_score)
        if isinstance(cls_score, tuple) or isinstance(cls_score, list):
            cls_score = [x.data.cpu().numpy() for x in cls_score]
            return [[x[i] for x in cls_score] for i in range(bs)]

        if self.softmax:
            cls_score = self.softmax_layer(cls_score)

        # return cls_score
        return cls_score.data.cpu().numpy()

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        # if keypoint.is_cuda is False:
        #     keypoint = keypoint.cuda(0)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            # if label.is_cuda is False:
            #     label = label.cuda(0)
            return self.forward_train(keypoint, label, **kwargs)

        return self.forward_test(keypoint, **kwargs)

    def extract_feat(self, keypoint):
        """Extract features through a backbone.

        Args:
            keypoint (torch.Tensor): The input keypoints.

        Returns:
            torch.tensor: The extracted features.
        """
        return self.backbone(keypoint)
