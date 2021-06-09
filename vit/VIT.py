import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertAttention, BertLayer, BertOutput, \
    BertEncoder, BertSelfOutput, BertIntermediate, BertEmbeddings, BaseModelOutputWithPastAndCrossAttentions
from vit.ImageEmbedding import Embedding


class BertEmbeddings_without_LayerNorm(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class BertOutput_without_LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class BertAttention_with_LayerNorm(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output = BertSelfOutput_without_LayerNorm(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        norm_hidden_states = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            norm_hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayer(BertLayer):
    def __init__(self, config, bert_layer):
        super().__init__(config)
        if bert_layer == 0:
            self.attention = BertAttention_with_LayerNorm(config)
            self.output = BertOutput_without_LayerNorm(config)
            self.intermediate = BertIntermediate_with_LayerNorm(config)


class BertEncoder(BertEncoder):
    def __init__(self, config, bert_layers):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, bert_layers[i]) for i in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            correct_vit_hidden_states=None,
            vit_layers=[],
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if (correct_vit_hidden_states is not None) and (i in vit_layers):
                vit_len = correct_vit_hidden_states.shape[2]
                vit_hidden_states = correct_vit_hidden_states[i]
                bert_hidden_states = hidden_states[:, :-vit_len]
                hidden_states = torch.cat([bert_hidden_states, vit_hidden_states], 1)
            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertSelfOutput_without_LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class BertIntermediate_with_LayerNorm(BertIntermediate):
    def __init__(self, config):
        super().__init__(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(self.LayerNorm(hidden_states))
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ImgBERT(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.patches_size = config.patches_size
        self.cls_img_embedding = nn.Embedding(1, config.hidden_size)
        self.img_embedding = Embedding(config.hidden_size, patches_size=self.patches_size)
        self.tokenizer = BertTokenizer.from_pretrained('../weights/saved-bert-base-uncased')
        self.head = nn.Linear(config.hidden_size, 1000)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.bert_layers = config.bert_layers
        self.vit = config.vit
        if self.bert_layers[0] == 0:
            self.embeddings = BertEmbeddings_without_LayerNorm(config)
            self.vit_embeddings = BertEmbeddings_without_LayerNorm(config)
        else:
            self.embeddings = BertEmbeddings(config)
            self.vit_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, self.bert_layers)
        # freeze_layers = [self.embeddings.position_embeddings, self.embeddings.token_type_embeddings]
        # self.freeze(freeze_layers)

    def get_embeddings(self, text=None, img=None):
        if text is not None:
            text = self.embeddings.word_embeddings(text)
        if img is not None:
            img = self.img_embedding(img)
        return text, img

    def freeze(self, layers):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def mask_image(self, image_embedding, image_mask):
        if image_mask is None:
            return image_embedding
        idx = np.where(image_mask == 1)
        image_embedding[idx] = 0
        return image_embedding

    def forward(
            self,
            input_ids=None,
            input_image=None,
            image_mask=None,
            attention_mask=None,
            correct_vit_hidden_states=None,
            vit_layers=[],
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            a=0
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None and input_image is not None:
            input_shape = input_ids.size()
            if self.vit:
                img_embedding_len = int(input_image.shape[2] * input_image.shape[3] / self.patches_size ** 2)
            else:
                img_embedding_len = 1 + int(input_image.shape[2] * input_image.shape[3] / self.patches_size ** 2)
            input_att_shape = torch.Size([input_shape[0], input_shape[1] + img_embedding_len])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds and input_image")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_att_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_att_shape,
                                                                                 device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        img_cls_token = torch.tensor([0], dtype=torch.long, device=device)
        cls_img_emb = self.cls_img_embedding(img_cls_token)
        cls_img_emb = cls_img_emb[None, :, :].repeat(input_shape[0], 1, 1)
        img_embedding = self.img_embedding(input_image)
        masked_img_embedding = self.mask_image(img_embedding, image_mask)
        if not self.vit:
            masked_img_embedding = torch.cat([cls_img_emb, masked_img_embedding], 1)
        img_type_ids = torch.ones(masked_img_embedding.shape[:2], dtype=torch.long, device=device)

        if self.vit:
            img_position_ids = torch.tensor(np.arange(input_shape[1],
                                                      input_shape[
                                                          1] + img_embedding_len)).long()  # сквозная нумерация картинки и <CLS>
        else:
            img_position_ids = torch.tensor(
                np.arange(img_embedding_len)).long()  # отдельная нумерация картинки и текста

        img_pos_ids = img_position_ids.to(device)

        img_embedding_output = self.vit_embeddings(input_ids=None,
                                                   position_ids=img_pos_ids,
                                                   token_type_ids=img_type_ids,
                                                   inputs_embeds=masked_img_embedding)
        embedding_output = torch.cat([embedding_output, img_embedding_output], 1)

        encoder_outputs = self.encoder(
            embedding_output,
            correct_vit_hidden_states=correct_vit_hidden_states,
            vit_layers=vit_layers,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if self.bert_layers[-1] == 0:
            sequence_output = self.LayerNorm(encoder_outputs[0])
        else:
            sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:
                                                      ]  # add hidden_states and attentions if they are here
        if self.vit:
            head_input = sequence_output[:, 0]
            logits = self.head(head_input)
        else:
            cls_idx = input_shape[1]
            head_input = (1 - a) * sequence_output[:, cls_idx] + a * sequence_output[:, cls_idx:].mean(dim=1)
            logits = self.head(head_input)

        return logits, outputs, img_embedding  # sequence_output, pooled_output, (hidden_states), (attentions)


class PredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls = nn.Linear(config.hidden_size, 2)

        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, cls_hidden_states, hidden_states):
        cls = self.cls(cls_hidden_states)
        hidden_states = self.linear(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        outputs = self.decoder(hidden_states)
        return cls, outputs


class ImgPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        outputs = self.decoder(hidden_states)
        return outputs
