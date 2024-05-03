import torch

from transformers import BertConfig, PreTrainedModel, PreTrainedTokenizerBase, Trainer

from tfbnlp.data import load_mnli_dataset, prepare_dataset


def run_trainer(
    config: BertConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(...)

    # TODO: setup

    trainer.train()

    # TODO: evaluation
