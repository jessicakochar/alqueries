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
        cache_dir: str | None = None,
    ) -> None:

        super().__init__()

        try:
            #from transformers import BertModel
            from transformers import AutoModelForSequenceClassification
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError("Install `transformers` to use BertClassifier.") from exc

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
        )

        # self.bert = BertModel.from_pretrained(model_name, cache_dir=cache_dir)

        # hidden_size = self.bert.config.hidden_size

        # self.dropout = nn.Dropout(dropout) # During training 10%(0.1) of the neurons will be randomly dropped out to prevent overfitting

        # self.classifier = nn.Linear(
        #     hidden_size,
        #     num_labels,
        # ) # The linear layer takes the hidden size -> number of labels (classes) as input and output dimensions -> output: logits for each class

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        embeddings = outputs.hidden_states[-1][:, 0, :]  # Get the [CLS] token embedding from the last hidden state
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "embeddings": embeddings,
        }
        # outputs = self.bert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        # )

        # embeddings = outputs.pooler_output

        # return classification_output(
        #     embeddings,
        #     self.dropout,
        #     self.classifier,
        #     labels,
        # )

    #     return self.model(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     token_type_ids=token_type_ids,
    #     labels=labels,
    # )