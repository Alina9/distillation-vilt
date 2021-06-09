import torch
import torch.nn as nn
from vit.VIT import ImgBERT
from transformers import BertTokenizer


class Vit(nn.Module):
    def __init__(self, weights='vit-model_16', freezed=False):
        super(Vit, self).__init__()
        if weights is None:
            vit_model_1 = ImgBERT.from_pretrained('../weights/bert_vit-model_16')
            self.vit_model = ImgBERT(vit_model_1.config)
        else:
            self.vit_model = ImgBERT.from_pretrained(weights)
        self.tokenizer = BertTokenizer.from_pretrained('../weights/saved-bert-base-uncased')
        if freezed:
            self.freeze([self.vit_model])

    def freeze(self, layers):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, image, tokens=None, attention_mask=None, hidden_states=None, vit_layers=[], a=0):
        """
        :param image
        :param tokens

        :returns hidden_states
        shape of output:  (13, bach_size, len_seq, len_hidden_state) 13 = 12 hidden states + output of the embeddings
        """
        if tokens is None:
            batch_size = image.shape[0]
            cls = torch.tensor(self.tokenizer.cls_token_id).long().unsqueeze(0)
            tokens = cls.repeat(batch_size, 1).to(image.device)
        logits, outputs, _ = self.vit_model(input_ids=tokens, input_image=image, attention_mask=attention_mask,
                                            correct_vit_hidden_states=hidden_states, output_hidden_states=True,
                                            vit_layers=vit_layers, a=a)
        hidden_states = outputs[2]
        hidden_states = [h.unsqueeze(0) for h in hidden_states]
        if self.vit_model.vit:
            # hidden_states = [h[:, 1:].unsqueeze(0) for h in hidden_states]  ## removed the output of <CLS>
            return logits, torch.cat(hidden_states, 0)
        # else:

        return logits, outputs[:2], torch.cat(hidden_states, 0)
