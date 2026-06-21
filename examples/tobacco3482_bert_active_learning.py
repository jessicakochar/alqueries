from __future__ import annotations

from functools import partial

from transformers import BertTokenizer

from alqueries.huggingface import (
    TextClassificationDataset,
    load_tobacco3482_ocr,
    predict_hf_text_classifier,
    train_hf_text_classifier,
)
from alqueries.training import ActiveLearningLoop, ActiveLearningLoopConfig
from models import BertClassifier


def main() -> None:
    data = load_tobacco3482_ocr(split="train", limit=500)
    dataset = TextClassificationDataset(data.texts, data.labels)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    config = ActiveLearningLoopConfig(
        initial_size=50,
        query_size=50,
        rounds=3,
        strategy_name="entropy_sampling",
        seed=7,
    )

    loop = ActiveLearningLoop(
        dataset,
        model_builder=lambda: BertClassifier(num_labels=len(data.label_names)),
        train_fn=partial(
            train_hf_text_classifier,
            tokenizer=tokenizer,
            batch_size=8,
            epochs=1,
            lr=2e-5,
        ),
        predict_fn=partial(
            predict_hf_text_classifier,
            tokenizer=tokenizer,
            batch_size=16,
            mc_dropout_runs=10,
        ),
        config=config,
    )

    for result in loop.run():
        print(
            {
                "round": result.round_index,
                "labeled_size": len(result.labeled_indices),
                "selected": result.selected_indices.tolist(),
                **dict(result.train_metrics),
            }
        )


if __name__ == "__main__":
    main()
