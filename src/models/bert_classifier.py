from __future__ import annotations
import torch
from torch import nn

from models.common import classification_output

class BertClassifier(nn.Module):

    def __init__(
        self,
        num_labels: int,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        try:
            from transformers import BertModel
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError("Install `transformers` to use BertClassifier.") from exc

        self.bert = BertModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout) # During training 10%(0.1) of the neurons will be randomly dropped out to prevent overfitting

        self.classifier = nn.Linear(
            hidden_size,
            num_labels,
        ) # The linear layer takes the hidden size -> number of labels (classes) as input and output dimensions -> output: logits for each class

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor, # which tokens are real and which are padding tokens (1 for real, 0 for padding), so the model doesn't attend to padding tokens
        labels: torch.Tensor | None = None, #used for calculating the loss during training, not needed during inference
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ) # text -> tokenizer -> input_ids and attention_mask -> BERT model -> outputs (last hidden state, pooler output, etc.)

        embeddings = outputs.pooler_output

        return classification_output(embeddings, self.dropout, self.classifier, labels)
    '''
    Document Text
      ↓
Tokenizer
      ↓
input_ids
      ↓
BERT
      ↓
768-d embedding
      ↓
Dropout
      ↓
Linear Layer
      ↓
Logits
      ↓
CrossEntropyLoss
    '''
