import torch
from collections import OrderedDict
import torch.distributed as dist
import torch.nn as nn
from ..builder import RECOGNIZERS
from ..builder import build_loss
from ...core import top_k_accuracy
from .. import builder
import torch.nn.functional as F
import copy


@RECOGNIZERS.register_module()
class RecognizerGCNMultiflow(nn.Module):
    def __init__(self, which_flow, flow_weight, num_classes,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0, **kwargs):

        # super().__init__()
        super(RecognizerGCNMultiflow, self).__init__()
        self.which_flow = which_flow
        self.flow_weight = flow_weight
        self.num_classes = num_classes
        if len(self.which_flow) >= 2:
            self.flag = True
        elif len(self.which_flow) == 1:
            self.flag = False
        else:
            raise "input which_flow error"

        sum_weight = 0.0
        for flow in self.which_flow:
            sum_weight += self.flow_weight[flow]
        self.sum_weight = sum_weight

        for flow in self.which_flow:
            self.flow_weight[flow] = self.flow_weight[flow]/self.sum_weight

        self.stream = nn.ModuleDict()

        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

    def add_flow(self, model, model_name):
        self.stream.add_module(model_name, model)

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        #  N, C, T, V, M = keypoint.size()
        keypoint_total = {}

        for i in range(len(self.which_flow)):
            keypoint_total[self.which_flow[i]] = keypoint[..., 3*i:3*(i+1)]

        if return_loss:
            losses = dict()
            cls_score = {}

            for flow in self.which_flow:
                cls_score[flow] = self.stream[flow](keypoint_total[flow], label, return_loss, **kwargs)

            total_cls_score = cls_score[self.which_flow[0]] * self.flow_weight[self.which_flow[0]]
            if self.flag:
                for flow in self.which_flow[1:]:
                    total_cls_score = torch.add(total_cls_score, cls_score[flow] * self.flow_weight[flow])

            gt_label = label.squeeze(-1)
            loss = self.loss(total_cls_score, gt_label)

            losses.update(loss)

            return losses
        else:
            cls_score = {}
            for flow in self.which_flow:
                cls_score[flow] = self.stream[flow](keypoint_total[flow], label, return_loss, **kwargs)

            total_cls_score = cls_score[self.which_flow[0]] * self.flow_weight[self.which_flow[0]]

            if self.flag:
                for flow in self.which_flow[1:]:
                    total_cls_score += cls_score[flow] * self.flow_weight[flow]

            return total_cls_score

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data_batch, optimizer, **kwargs):
        losses = self(**data_batch, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def loss(self, cls_score, label, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``label``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            label (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        # loss_cls = self.loss_cls(cls_score, label, **kwargs)
        # 对预测值取log
        log_cls_score = torch.log(cls_score)
        # 计算最终的结果
        loss_cls = F.nll_loss(log_cls_score, label)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses





