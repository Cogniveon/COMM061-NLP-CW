import os
from typing import Any
import numpy as np
import evaluate
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import DatasetDict


def init_trainer(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    label_list: list[str],
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    batch_size: int,
    num_epochs: int,
    model_name_or_path: str,
    dataset: DatasetDict,
    **trainer_kwargs: dict[str, Any],
):
    metric = evaluate.load("seqeval")
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model_name = model_name_or_path.split("/")[-1]
    run_name = f"nlpcw_{model_name}-abbr"
    args = TrainingArguments(
        output_dir=os.path.join("experiments", run_name),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.001,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        **trainer_kwargs,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    return trainer
