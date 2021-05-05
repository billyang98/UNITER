"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VQA model
"""
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU, BertOnlyMLMHead
from .model import UniterPreTrainedModel, UniterModel


class UniterForVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        self.apply(self.init_weights)
        # added MLM task stuff
        self.cls = BertOnlyMLMHead(
            config, self.uniter.embeddings.word_embeddings.weight)

    def forward(self, batch, compute_loss=True, task='vqa', text_only=False):
        assert task == 'vqa' or task =='mlm', "Invalid task {}".format(task)
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        if task == 'vqa':
            sequence_output = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attn_masks, gather_index,
                                          output_all_encoded_layers=False)
            pooled_output = self.uniter.pooler(sequence_output)
            answer_scores = self.vqa_output(pooled_output)

            if compute_loss:
                targets = batch['targets']
                vqa_loss = F.binary_cross_entropy_with_logits(
                    answer_scores, targets, reduction='none')
                return vqa_loss
            else:
                return answer_scores
        elif task == 'mlm':
            txt_labels = batch['masked_txt_labels']
            input_ids = batch['masked_input_ids']
            if text_only:
                img_feat = None 
            sequence_output = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attn_masks, gather_index,
                                          output_all_encoded_layers=False)
            # get only the text part
            sequence_output = sequence_output[:, :input_ids.size(1), :]
            # only compute masked tokens for better efficiency
            masked_output = self._compute_masked_hidden(sequence_output,
                                                        txt_labels != -1)
            prediction_scores = self.cls(masked_output)

            if compute_loss:
                masked_lm_loss = F.cross_entropy(prediction_scores,
                                                txt_labels[txt_labels != -1],
                                                reduction='none')
                return masked_lm_loss
            else:
                return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked
