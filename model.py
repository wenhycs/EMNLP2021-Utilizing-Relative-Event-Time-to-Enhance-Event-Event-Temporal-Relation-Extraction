import logging
import warnings

from os import path

import torch
import torch.nn as nn

from copy import deepcopy
from transformers import BertPreTrainedModel
from transformers.file_utils import add_start_docstrings
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_roberta import ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING,\
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, RobertaModel, RobertaClassificationHead

from utils import get_label_type


@add_start_docstrings(
    "A relational classifier and relative time anchoring \
        on top of RoBERTa Model transformer.",
    ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING
)
class TemporalRelationClassificationAndRelativeAnchor(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, dataset=None, cascade=True):
        super(TemporalRelationClassificationAndRelativeAnchor, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.cascade = cascade
        
        config_for_classification_head = deepcopy(config)
        config_for_classification_head.num_labels = len(get_label_type(dataset))
        config_for_classification_head.hidden_size *= 2
        if self.cascade:
            config_for_classification_head.hidden_size += 2

        self.classifier = RobertaClassificationHead(config_for_classification_head)

        config_for_time_anchor = deepcopy(config)
        config_for_time_anchor.num_labels = 1

        self.time_anchor = RobertaClassificationHead(config_for_time_anchor)

        self.init_weights()

    def forward(self, input_ids, attention_mask, event_ix, labels=None, output_time=False):

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        batch_size, sequence_length, hidden_size = sequence_output.size()

        event_1_ix, event_2_ix = event_ix.split(1, dim=-1)
        event_1 = torch.gather(sequence_output, dim=1, 
                               index=event_1_ix.expand(batch_size, hidden_size).unsqueeze(dim=1))
        event_2 = torch.gather(sequence_output, dim=1,
                               index=event_2_ix.expand(batch_size, hidden_size).unsqueeze(dim=1))
        event_pair = torch.cat([event_1.squeeze(dim=1), event_2.squeeze(dim=1)], dim=1)

        event_1_time = torch.tanh(self.time_anchor(event_1).squeeze())
        event_2_time = torch.tanh(self.time_anchor(event_2).squeeze())

        if self.cascade:
            augmented_repr = torch.cat([event_1_time.unsqueeze(dim=-1),
                                        event_2_time.unsqueeze(dim=-1)], dim=-1).view(-1 ,2)
            event_pair = torch.cat([event_pair, augmented_repr], dim=-1)

        logits = self.classifier(event_pair.unsqueeze(dim=1))

        loss = 0.
        relative = event_1_time - event_2_time
        mask_before = (labels == 0).float()
        relative_sum_before = ((1 + relative) > 0).float() * (1 + relative)
        loss += torch.sum(relative_sum_before * mask_before)
        mask_after = (labels == 1).float()
        relative_sum_after = ((1 - relative) > 0).float() * (1 - relative)
        loss += torch.sum(relative_sum_after * mask_after)
        mask_equal = (labels == 2).float()
        loss += torch.sum(torch.abs(relative * mask_equal))
        loss /= batch_size

        final_outputs = []
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss + loss_fct(logits, labels)
            final_outputs.append(loss)
            final_outputs.append(logits)
        else:
            final_outputs.append(logits)
        if output_time:
            final_outputs.append([event_1_time, event_2_time])
        return final_outputs

def prepare_model(language_model_name, model_type, device, model_weights_path=None, dataset='matres'):
    logging.info(f"***** Loading Model {model_type} *****\n")
    logging.info(f"device = {device}")

    if model_weights_path is None:
        logging.info(f"Loading default model weights from {language_model_name}")
    elif path.isdir(model_weights_path):
        logging.info(f"Loading model weights from {model_weights_path}")
        language_model_name = model_weights_path
    else:
        raise ValueError("Invalid value for model weights")
    
    if model_type == 'time_anchor':
        model = TemporalRelationClassificationAndRelativeAnchor.from_pretrained(language_model_name, dataset=dataset)
    else:
        ValueError(f"Invalid model_type {model_type}")
    
    model.to(device)

    logging.info(f"Initialized model {model}")
    return model


def prepare_model_eval(language_model_name, model_type, device, model_weights_path=None, dataset='matres'):
    logging.info(f"***** Loading Model {model_type} *****\n")
    logging.info(f"device = {device}")

    if model_weights_path is None:
        logging.info(f"Loading default model weights from {language_model_name}")
    elif path.isdir(model_weights_path):
        logging.info(f"Loading model weights from {model_weights_path}")
        language_model_name = model_weights_path
    else:
        raise ValueError("Invalid value for model weights")

    if model_type == 'time_anchor':
        model = TemporalRelationClassificationAndRelativeAnchor.from_pretrained(language_model_name, dataset=dataset)
    else:
        ValueError(f"Invalid model_type {model_type}")

    model.to(device)

    return model
