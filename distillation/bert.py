from transformers import BertModel, BertForPreTraining
import torch
import torch.nn as nn


class Bert(nn.Module):
    def __init__(self, weights='../weights/saved-bert-base-uncased'):
        super(Bert, self).__init__()
        self.bert = BertForPreTraining.from_pretrained(weights)
        self.vocab_size = self.bert.config.vocab_size
        self.freeze([self.bert])

    def freeze(self, layers):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
    def forward(self, tokens, attention_mask):
        """
        :param tokens
        :param attention_mask

        shape of output:  (13, bach_size, len_seq, len_hidden_state) 13 = 12 hidden states + output of the embeddings
        """
        outputs = self.bert(input_ids=tokens, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['hidden_states']
        prediction_logits, seq_relationship_logits = outputs['prediction_logits'],  outputs['seq_relationship_logits']
        hidden_states = [h.unsqueeze(0) for h in hidden_states]
        hidden_states = torch.cat(hidden_states, 0)
        return prediction_logits, seq_relationship_logits
