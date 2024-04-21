import logging
from datasets import DatasetDict, load_dataset, ClassLabel, Sequence

logger = logging.getLogger(__name__)


def get_dataset(dataset_name: str, expected_labels: list[str]) -> DatasetDict:
    id2label = dict(enumerate(expected_labels))
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    dataset = load_dataset(dataset_name, trust_remote_code=True)

    # Map PLOD-CW labels 'B-O', etc => 0,1,2,3, etc
    if dataset_name == "surrey-nlp/PLOD-CW":
        logger.debug("PLOD-CW dataset detected... mapping labels to ids...")

        def labelname_to_id(examples, column_name: str, map_dict: dict):
            examples[column_name] = [
                [map_dict[val] for val in seq] for seq in examples[column_name]
            ]
            return examples

        dataset = dataset.map(
            labelname_to_id,
            batched=True,
            fn_kwargs={"column_name": "ner_tags", "map_dict": label2id},
        )

    ner_tags_feature = ClassLabel(
        num_classes=len(id2label), names=list(id2label.values())
    )
    dataset = dataset.cast_column("ner_tags", feature=Sequence(ner_tags_feature))
    dataset = dataset.remove_columns(["pos_tags"])

    label_list = dataset["train"].features["ner_tags"].feature.names
    logger.debug("NER labels: %", label_list)

    return dataset, id2label, label2id, label_list, num_labels
