import paddle
import paddle.nn as nn
from ..registry import LOSSES
from .base import BaseWeightedLoss
from ...utils.manet_utils import float_, long_


@LOSSES.register()
class Added_BCEWithLogitsLoss(BaseWeightedLoss):
    def __init__(self,
                 top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(Added_BCEWithLogitsLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert (top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step
        if self.top_k_percent_pixels == None:
            self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.bceloss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, dic_tmp, label_tmp, step, obj_ids, seq_):

        final_loss = 0
        for seq_name in dic_tmp.keys():
            label_ = label_tmp[seq_name].transpose([1, 2, 0])
            label_ = (float_(label_) == float_(obj_ids[seq_name]))
            label_ = label_.unsqueeze(-1).transpose([3, 2, 0, 1])
            pred_logits = dic_tmp[seq_name]
            gts = float_(label_)
            if self.top_k_percent_pixels == None:
                final_loss += self.bceloss(pred_logits, gts)
            else:
                # Only compute the loss for top k percent pixels.
                # First, compute the loss for all pixels. Note we do not put the loss
                # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.shape[2] * pred_logits.shape[3])
                pred_logits = pred_logits.view(
                    -1, pred_logits.shape[1],
                    pred_logits.shape[2] * pred_logits.shape[3])
                gts = gts.view(-1, gts.shape[1], gts.shape[2] * gts.shape[3])
                pixel_losses = self.bceloss(pred_logits, gts)
                if self.hard_example_mining_step == 0:
                    top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
                else:
                    ratio = min(1.0,
                                step / float(self.hard_example_mining_step))
                    top_k_pixels = int((ratio * self.top_k_percent_pixels +
                                        (1.0 - ratio)) * num_pixels)
                _, top_k_indices = paddle.topk(pixel_losses,
                                               k=top_k_pixels,
                                               axis=2)

                final_loss += nn.BCEWithLogitsLoss(weight=top_k_indices,
                                                   reduction='mean')(
                                                       pred_logits, gts)
        return final_loss


@LOSSES.register()
class Added_CrossEntropyLoss(BaseWeightedLoss):
    def __init__(self,
                 top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(Added_CrossEntropyLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert (top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step
        if self.top_k_percent_pixels == None:
            self.celoss = nn.CrossEntropyLoss(ignore_index=255,
                                              reduction='mean')
        else:
            self.celoss = nn.CrossEntropyLoss(ignore_index=255,
                                              reduction='none')

    def forward(self, dic_tmp, label_tmp, step, seq_, **kwargs):

        final_loss = 0
        for seq_name in dic_tmp.keys():
            pred_logits = dic_tmp[seq_name]
            gts = long_(label_tmp[seq_name])
            if self.top_k_percent_pixels == None:
                final_loss += self.celoss(pred_logits, gts)
            else:
                # Only compute the loss for top k percent pixels.
                # First, compute the loss for all pixels. Note we do not put the loss
                # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.shape[2] * pred_logits.shape[3])
                pred_logits = pred_logits.reshape([
                    pred_logits.shape[1],
                    pred_logits.shape[2] * pred_logits.shape[3]
                ]).transpose([1, 0])
                gts = gts.reshape([gts.shape[1] * gts.shape[2]])
                pixel_losses = self.celoss(pred_logits, gts).reshape([1, -1])
                if self.hard_example_mining_step == 0:
                    top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
                else:
                    ratio = min(1.0,
                                step / float(self.hard_example_mining_step))
                    top_k_pixels = int((ratio * self.top_k_percent_pixels +
                                        (1.0 - ratio)) * num_pixels)
                top_k_loss, top_k_indices = paddle.topk(pixel_losses,
                                                        k=top_k_pixels,
                                                        axis=1)

                final_loss += paddle.mean(top_k_loss)
        return final_loss
