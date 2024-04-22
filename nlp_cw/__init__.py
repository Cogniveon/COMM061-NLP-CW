import logging.config
from simple_parsing import parse, field, choice
from dataclasses import dataclass

from nlp_cw.get_dataset import get_dataset
from nlp_cw.tokenize_dataset import tokenize_dataset
from nlp_cw.trainer import init_trainer


@dataclass
class TrainerConfig:
    batch_size: int = field(default=1)
    num_epochs: int = field(default=5)
    labels: list[str] = field(
        default_factory=["B-O", "B-AC", "I-AC", "B-LF", "I-LF"].copy
    )
    dataset: str = choice(
        [
            "surrey-nlp/PLOD-filtered",
            "surrey-nlp/PLOD-unfiltered",
            "surrey-nlp/PLOD-CW",
        ],
        default="surrey-nlp/PLOD-CW",
    )  # Expects a dataset with train, val, test splits of examples in format ('tokens': Seq(int), 'ner_tags': Seq(int))
    model: str = field(default="romainlhardy/roberta-large-finetuned-ner")

    @property
    def loggingConfig(self):
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {
                    "format": "%(levelname)s %(pathname)s:%(lineno)d %(asctime)s: %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                },
            },
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "level": "WARNING",
                    "formatter": "simple",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "simple",
                    "filename": "nlp_cw.log",
                    "maxBytes": 10000,
                    "backupCount": 3,
                },
            },
            "loggers": {
                "nlp_cw": {
                    "level": "DEBUG",
                    "handlers": ["stderr", "file"],
                },
            },
        }


def main_cli():
    config = parse(TrainerConfig)

    logging.config.dictConfig(config.loggingConfig)

    dataset, id2label, label2id, label_list, num_labels = get_dataset(
        config.dataset, config.labels
    )

    tokenizer, tokenized_dataset = tokenize_dataset(dataset, config.model)

    trainer = init_trainer(
        tokenizer,
        label_list,
        num_labels,
        id2label,
        label2id,
        config.batch_size,
        config.num_epochs,
        model_name_or_path=config.model,
        dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.evaluate()
