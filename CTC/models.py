import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput
from transformers.models.bert.modeling_bert import BertLayer
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()

        if num_layers == 1:
            self.seq = nn.Sequential(nn.Linear(input_dim, output_dim, bias=True))
        elif num_layers == 2:
            self.seq = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True), nn.GELU(), nn.Linear(hidden_dim, output_dim, bias=True))
        elif num_layers == 3:
            self.seq = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim, bias=True), nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim, bias=True), nn.GELU(),
                    nn.Linear(hidden_dim, output_dim, bias=True))

            raise NotImplementedError(f"MLP layer number = {num_layers} is not implemented!")
        
        self.seq.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        return self.seq(x)


class CellTypeCLSModel(nn.Module):
    """
    The model for AxCell style inputs
    """
    def __init__(self, config):
    # def __init__(self, pretrained: str, classification_head, num_labels, hidden_dim=32, num_MLP_layers=2):

        super().__init__()

        self.num_labels = config.num_labels

        self.text_encoder = AutoModel.from_pretrained(config.pretrained)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = MLPHead(output_dim=config.num_labels, input_dim=768, hidden_dim=32, num_layers=config.num_MLP_layers)
    
    def forward(self, 
        input_ids: torch.LongTensor,  # (BS, seq_length)
        attention_mask: torch.LongTensor,
        has_reference_ids: Optional[torch.LongTensor] = None,
        input_type_ids: Optional[torch.LongTensor] = None,
        labels=None
    ):
    
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(outputs[1])  # BaseModelOutputWithPoolingAndCrossAttentions.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SciBertWithAdditionalFeatures(nn.Module):
    """
    The model for AxCell style inputs, with additional features.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.text_encoder = AutoModel.from_pretrained(config.pretrained)
        self.input_type_embedding = nn.Embedding(len(config.input_cols), 768)
        self._init_weights(self.input_type_embedding)

        # self.row_id_embedding = nn.Embedding(57, 192)
        # self.col_id_embedding = nn.Embedding(25, 192)
        # self.reverse_row_id_embedding = nn.Embedding(57, 192)
        # self.reverse_col_id_embedding = nn.Embedding(25, 192)
        # self.region_type_embedding = nn.Embedding(4, 768)
        # for i in ['row_id_embedding', 'col_id_embedding', 'reverse_row_id_embedding', 'reverse_col_id_embedding', 'region_type_embedding', 'input_type_embedding']:
        #     self._init_weights(getattr(self, i))
    
        self.dropout = nn.Dropout(p=0.1, inplace=False)

        if hasattr(config, 'num_labels'):
            self.num_labels = config.num_labels
            self.classifier = MLPHead(output_dim=config.num_labels, input_dim=768, hidden_dim=32, num_layers=config.num_MLP_layers)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, 
        input_ids: torch.LongTensor,  # (BS, seq_length)
        attention_mask: torch.LongTensor,
        # row_id: Optional[torch.LongTensor] = None,  # (BS, 1)
        # col_id: Optional[torch.LongTensor] = None,
        # reverse_row_id: Optional[torch.LongTensor] = None,
        # reverse_col_id: Optional[torch.LongTensor] = None,
        input_type_ids: Optional[torch.LongTensor] = None,
        # region_type: Optional[torch.LongTensor] = None,
        labels = None
    ):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        position_ids = self.text_encoder.embeddings.position_ids[:,  : seq_length]
        extended_attention_mask: torch.Tensor = attention_mask[:, None, None, :]
        input_type_embeddings = self.input_type_embedding(input_type_ids)
        
        word_embeddings = self.text_encoder.embeddings.word_embeddings(input_ids)
        position_embeddings = self.text_encoder.embeddings.position_embeddings(position_ids)

        embeddings = word_embeddings + input_type_embeddings + position_embeddings
        embeddings = self.text_encoder.embeddings.LayerNorm(embeddings)
        embedding_output = self.text_encoder.embeddings.dropout(embeddings)

        encoder_outputs = self.text_encoder.encoder(hidden_states=embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]  # last_hidden_state


        logits = None
        loss = None
        if labels is not None:
            pooled_output = self.text_encoder.pooler(sequence_output)  # (BS, 768)
            ## Add additional numerical and categorical features
            # for i, layer in zip([row_id, col_id, reverse_row_id, reverse_col_id, region_type], (self.row_id_embedding, self.col_id_embedding, self.reverse_row_id_embedding, self.reverse_col_id_embedding, self.region_type_embedding)):
            #     if i is not None:
            #         pooled_output = torch.cat((pooled_output, layer(i)), dim=1)
            ############################
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if labels is not None:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
        else:
            return BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
            ) 


class TableEmbedding(nn.Module):
    """
    Embeds a flattened table into hidden states by a pretrained model.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.has_reference_embeddings = nn.Embedding(2, 768)
        self.is_empty_embeddings = nn.Embedding(2, 768)
        self.row_position_embeddings = nn.Embedding(512, 768)
        self.col_position_embeddings = nn.Embedding(512, 768)
        self.layer_norm = nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(p=0.1, inplace=False)

        self.combiner = nn.Sequential(nn.Linear(768 * 2, 768, bias=True), nn.GELU(), nn.LayerNorm(768, eps=1e-12))

        self.combiner.apply(self._init_weights)
        self.init_weights()

        # grab the last BertLayer from SciBert
        # self.seq_encoder = AutoModel.from_pretrained(config.pretrained).encoder
        self.seq_encoder = AutoModel.from_pretrained(config.pretrained).encoder.layer[-2]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        for module in self.modules():
            self._init_weights(module)
    
    def combine(self, in1, in2):
        assert(in1.shape == in2.shape)
        return torch.mean(torch.stack([in1, in2]), dim=0)
        # return self.combiner(torch.cat([in1, in2], dim=-1))

    def forward(self, 
        has_reference_ids: torch.LongTensor,
        is_empty_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        input_embeds1: Optional[torch.FloatTensor] = None,
        row_position_ids: Optional[torch.LongTensor] = None,
        col_position_ids: Optional[torch.LongTensor] = None,
    )-> BaseModelOutput:

        input_shape = input_embeds.size()[:-1]

        batch_size, seq_length = input_shape
        device = input_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        assert(attention_mask.dim() == 2)
        extended_attention_mask: torch.Tensor = attention_mask[:, None, None, :]

        dummy = torch.zeros(((batch_size, seq_length, 768)), device=device)

        is_empty_embeddings = dummy
        if self.config.use_is_empty:
            is_empty_embeddings = self.is_empty_embeddings(is_empty_ids)
            
        has_reference_embeddings = dummy
        if self.config.use_has_reference:
            has_reference_embeddings = self.has_reference_embeddings(has_reference_ids)
            
        row_position_embeddings = self.row_position_embeddings(row_position_ids)
        col_position_embeddings = self.col_position_embeddings(col_position_ids)

        assert(input_embeds is not None)
        if input_embeds is not None and input_embeds1 is not None:
            input_embeds = self.combine(input_embeds, input_embeds1)

        embeddings = has_reference_embeddings + row_position_embeddings + col_position_embeddings + input_embeds + is_empty_embeddings

        embeddings = self.layer_norm(embeddings)
        embedding_output = self.dropout(embeddings)

        ## Use the pre-trained encoder

        if isinstance(self.seq_encoder, nn.ModuleList):
            for i, layer_module in enumerate(self.seq_encoder):
                layer_outputs = layer_module(
                    embedding_output,
                    attention_mask,
                )
                embedding_output = layer_outputs[0]
            encoder_output = layer_outputs
        else:
            encoder_output = self.seq_encoder(
                hidden_states=embedding_output,
                attention_mask=extended_attention_mask,
            )

        last_hidden_state = encoder_output[0]  # (batch_size, sequence_length, hidden_size)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class TableEmbeddingCTCHeadModel(nn.Module):
    def __init__(self, config):
        """
        Unified means that each cell produces one token embedding
        """
        super().__init__()
        
        self.num_labels = config.num_labels
        self.unified = config.unified
        self.tab_emb = TableEmbedding(config)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = MLPHead(output_dim=config.num_labels, input_dim=768+1, hidden_dim=config.hidden_dim, num_layers=config.num_MLP_layers)

    def forward(self,
        num_rows: int,
        num_cols: int,
        has_reference_ids: torch.LongTensor,
        is_empty_ids: torch.LongTensor,
        spans_or_locations,
        labels,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids1: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        input_embeds1: Optional[torch.FloatTensor] = None,
        row_position_ids: Optional[torch.LongTensor] = None,
        col_position_ids: Optional[torch.LongTensor] = None
    ):
        """
            spans: (batch_size, 2)
            labels: (batch_size, num_labels)
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = input_embeds.size()[:-1]
                
        batch_size, seq_length = input_shape

        table_embedding_output = self.tab_emb(
            input_ids=input_ids,
            input_embeds=input_embeds,
            input_embeds1=input_embeds1,
            is_empty_ids=is_empty_ids,
            has_reference_ids=has_reference_ids,
            attention_mask=attention_mask,
            row_position_ids=row_position_ids,
            col_position_ids=col_position_ids
        )

        table_embedding = table_embedding_output.last_hidden_state

        if self.unified is False:
            padded = torch.nn.functional.pad(table_embedding.cumsum(dim=1), (0, 0, 1, 0))
            # (batch_size, hidden_size)
            cell_embedding = (padded[range(batch_size), spans_or_locations[range(batch_size), 1]] - padded[range(batch_size), spans_or_locations[range(batch_size), 0]]) / torch.diff(spans_or_locations, dim=1)
        else:
            # assume col-wise flattened
            index = (spans_or_locations[:, 0] + spans_or_locations[:, 1] * num_rows)  # (batch_size), +1 to consider [CLS]
            cell_embedding = table_embedding[torch.arange(batch_size), index]  # (batch_size, hidden_size)

            cell_embedding = self.dropout(cell_embedding)
            has_reference_cell = has_reference_ids[torch.arange(batch_size), index][:, None]
            cell_embedding = torch.cat([cell_embedding, has_reference_cell], dim=1)        

        logits = self.classifier(cell_embedding)
        # logits = self.classifier(self.dropout(cell_embedding))

        loss = None
        if labels is not None:
            # single_label_classification
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


class TableEmbeddingCTCSeqHeadModel(nn.Module):
    """Produce the labels for cells in a table all at once."""
    def __init__(self, config):
        super().__init__()
        
        self.num_labels = config.num_labels
        self.tab_emb = TableEmbedding(config)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.transform = nn.Sequential(
                nn.Linear(768, 768, bias=True), nn.GELU(),  nn.LayerNorm(768, eps=1e-12))
        self.classifier = MLPHead(output_dim=config.num_labels, input_dim=768, hidden_dim=config.hidden_dim, num_layers=config.num_MLP_layers)

        self.transform.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self,
        has_reference_ids: torch.LongTensor,
        is_empty_ids: torch.LongTensor,
        labels,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        input_embeds1: Optional[torch.FloatTensor] = None,
        row_position_ids: Optional[torch.LongTensor] = None,
        col_position_ids: Optional[torch.LongTensor] = None
    ):

        table_embedding_output = self.tab_emb(
            input_ids=input_ids,
            input_embeds=input_embeds,
            input_embeds1=input_embeds1,
            is_empty_ids=is_empty_ids,
            has_reference_ids=has_reference_ids,
            attention_mask=attention_mask,
            row_position_ids=row_position_ids,
            col_position_ids=col_position_ids
        )

        table_embedding = table_embedding_output.last_hidden_state  # (BS, seq_length, 768)
        prediction_scores = self.classifier(self.transform(table_embedding))

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')
            loss = loss_fct(prediction_scores.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=prediction_scores.view(-1, self.num_labels)
        )


class TableEmbeddingWithCellsSentences(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.transform = nn.Sequential(
                nn.Linear(768, 768, bias=True), nn.GELU(), nn.LayerNorm(768, eps=1e-12))
        self.classifier = MLPHead(output_dim=config.num_labels, input_dim=768, hidden_dim=config.hidden_dim, num_layers=config.num_MLP_layers)
        self.transform.apply(self._init_weights)
        self.tab_emb = TableEmbedding(config)
        self.custom_transformer = None
       

        # self.text_sentence_encoder = AutoModel.from_pretrained(config.pretrained)
        self.cell_content_encoder = AutoModel.from_pretrained(config.pretrained)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def pool_seq(self, input, mtd, 
        attn_mask: Optional[torch.LongTensor] = None
    ):
        if mtd == 'pooler_output':
            return self.dropout(input[1])
        elif mtd == 'avg':
            hidden_state = input.hidden_states[0]
            mask = torch.unsqueeze(attn_mask, dim=-1).expand(hidden_state.size())
            embed = torch.sum(hidden_state * mask, dim=1) / torch.sum(mask, dim=1)
            return torch.nan_to_num(embed)
        elif mtd == 'cls':
            return self.dropout(input.last_hidden_state[:, 0])
        else:
            raise ValueError(f"{mtd}")

    def forward(self,
        input_cell_content_ids,  # (BS, cell_count, seq_len1)
        input_cell_content_ids_attn_masks,  # (BS, cell_count, seq_len1)
        has_reference_ids: torch.LongTensor,
        is_empty_ids: torch.LongTensor,
        attention_mask,  # The attention mask along the cell_count dim
        input_text_sentence_embeds: Optional[torch.FloatTensor] = None,  # (BS, cell_count, 768)
        input_text_sentence_ids: Optional[torch.FloatTensor] = None,  # (BS, cell_count, seq_len2)
        input_text_sentence_ids_attn_masks: Optional[torch.LongTensor] = None,  # (BS, cell_count, seq_len1)
        labels: Optional[torch.LongTensor] = None,
        row_position_ids: Optional[torch.LongTensor] = None,
        col_position_ids: Optional[torch.LongTensor] = None,
    ):

        input_shape = input_cell_content_ids.size()
        BS, cell_count, seq_len1 = input_shape

        # input_shape = input_text_sentence_ids.size()
        # BS, cell_count, seq_len2 = input_shape

        device = input_cell_content_ids.device 


        ########## Get the embedding for each cell ####################

        # cell_contents_embeds = []
        # # text_sentence_embeds = []

        # for idx in range(cell_count):
        #     input_cell_content_id = input_cell_content_ids[:, idx, :]

        #     cell_content_outputs = self.cell_content_encoder(input_ids=input_cell_content_id, attention_mask=input_cell_content_ids_attn_masks[:, idx])  # (BS, seq_len1, 768)
        #     cell_content = self.pool_seq(cell_content_outputs)  # (BS, 768)
        #     cell_contents_embeds.append(cell_content[:, None, :].cpu())
        
        # cell_contents_embeds = torch.cat(cell_contents_embeds, dim=1).to(device)  # (BS, cell_count, 768)

        ### cell_contents

        cell_content_outputs = self.cell_content_encoder(input_ids=input_cell_content_ids.view(-1, seq_len1), attention_mask=input_cell_content_ids_attn_masks.view(-1, seq_len1), output_hidden_states=True)  # (BS * cell_count, seq_len1, 768)

        cell_contents_embeds = self.pool_seq(cell_content_outputs, mtd='pooler_output')  # (BS * cell_count, seq_len1, 768)
        # cell_contents_embeds = self.pool_seq(cell_content_outputs, mtd='avg', attn_mask=input_cell_content_ids_attn_masks.view(-1, seq_len1))

        cell_contents_embeds = cell_contents_embeds.view(BS, cell_count, 768)
        ###

        ### text_sentences

        # text_sentence_outputs = self.cell_content_encoder(input_ids=input_text_sentence_ids.view(-1, seq_len2), attention_mask=input_text_sentence_ids_attn_masks.view(-1, seq_len2))  # (BS * cell_count, seq_len1, 768)

        # text_sentence_embeds = self.pool_seq(text_sentence_outputs, mtd='pooler_output')  # (BS * cell_count, seq_len2, 768)
        # # cell_contents_embeds = self.pool_seq(cell_content_outputs, mtd='avg', attn_mask=input_cell_content_ids_attn_masks.view(-1, seq_len2))

        # text_sentence_embeds = text_sentence_embeds.view(BS, cell_count, 768)
        ###

        ############################################################

        table_embedding_output = self.tab_emb(
            input_embeds=cell_contents_embeds,
            input_embeds1=input_text_sentence_embeds,
            is_empty_ids=is_empty_ids,
            has_reference_ids=has_reference_ids,
            attention_mask=attention_mask,
            row_position_ids=row_position_ids,
            col_position_ids=col_position_ids
        )

        table_embedding = table_embedding_output.last_hidden_state
        prediction_scores = self.classifier(self.transform(table_embedding))

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prediction_scores.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=prediction_scores.view(-1, self.num_labels)
        )

