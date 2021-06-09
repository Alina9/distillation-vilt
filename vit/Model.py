import torch.nn as nn
import torch
import numpy as np
from VIT import ImgBERT, PredictionHead, ImgPredictionHead


class Model(nn.Module):
    def __init__(self, model_weight):
        super(Model, self).__init__()
        self.bert = ImgBERT.from_pretrained(model_weight)
        self.head_text = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.head_image = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.vocab_size = self.bert.config.vocab_size
        #self.freeze([self.bert, self.img_prediction_head])

    def get_embeddings(self, text=None, img=None):
        text_embed, img_embed = self.bert.get_embeddings(text, img)
        return text_embed, img_embed

    def freeze(self, layers):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, text, image, image_mask, attention_mask, alpha):
        _, hidden_states, img_embedding = self.bert(input_ids=text, input_image=image, image_mask=image_mask,
                                                 attention_mask=attention_mask)
        txt_len = text.shape[1]
        sequence_output = hidden_states[0]

        out_text = (1 - alpha) * sequence_output[:, 0] + alpha * sequence_output[:, 1:txt_len].mean(dim=1)
        out_image = (1 - alpha) * sequence_output[:, txt_len] + alpha * sequence_output[:, txt_len + 1:].mean(dim=1)
        out_text = self.head_text(out_text)
        out_image = self.head_image(out_image)
        out = out_text @ out_image.transpose(0, 1)
        return out

