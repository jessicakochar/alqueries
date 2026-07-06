from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tobacco3482Data:
    texts: list[str]
    labels: list[int]
    label_names: list[str]


def load_tobacco3482_ocr(
    *,
    split: str = "train",
    dataset_name: str = "anirudh1112/corrected-tobacco-dataset-with-ocr",
    text_column: str = "text",
    label_column: str = "label",
    limit: int | None = None,
    cache_dir: str | None = None,
) -> Tobacco3482Data:
    """
    Load OCR text + labels for Tobacco3482 from Hugging Face.

    This is the easiest first dataset for BERT because it exposes document OCR
    text directly, while still belonging to the document-understanding setting.
    """

    try:
        from datasets import ClassLabel, load_dataset
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("Install `datasets` to load Tobacco3482.") from exc

    print(f"[Dataset] Loading {dataset_name} split={split}")
    if cache_dir is not None:
        print(f"[Dataset] Cache dir: {cache_dir}")

    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    print(f"[Dataset] Full size loaded: {len(dataset)} samples")

    if limit is not None:
        old_size = len(dataset)
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"[Dataset] Reduced dataset: {old_size} → {len(dataset)} samples")

    label_feature = dataset.features.get(label_column)
    if isinstance(label_feature, ClassLabel):
        label_names = list(label_feature.names)
        labels = [int(value) for value in dataset[label_column]]
    else:
        raw_labels = list(dataset[label_column])
        label_names = sorted({str(value) for value in raw_labels})
        label_to_id = {label: index for index, label in enumerate(label_names)}
        labels = [label_to_id[str(value)] for value in raw_labels]

    texts = [str(value or "") for value in dataset[text_column]]
    return Tobacco3482Data(texts=texts, labels=labels, label_names=label_names)
