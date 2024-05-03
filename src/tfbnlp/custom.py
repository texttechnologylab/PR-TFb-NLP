import torch

from datasets import DatasetDict
from torch.nn.utils.rnn import pad_sequence  # hint
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import BertConfig, PreTrainedModel, PreTrainedTokenizerBase

# TODO: choose and import the necessary modules
from torch.nn import *  # TODO
from torch.optim import *  # TODO

from tfbnlp.data import load_mnli_dataset, prepare_dataset


def custom_collate(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = ...
    label = ...
    return input_ids, label


def run_custom(
    config: BertConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode_pt(batch: dict):
        return tokenizer(
            batch["input"],
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )

    pt_dataset: DatasetDict = ...  # TODO

    criterion = ...  # TODO
    optimizer = ...  # TODO
    num_epochs = ...  # TODO
    batch_size = ...  # TODO

    train_dataloader = DataLoader(
        pt_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )
    dev_dataloader = DataLoader(
        pt_dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
    )

    model.to(device)
    for epoch in trange(num_epochs, position=0):
        model.train()
        for batch in tqdm(train_dataloader, position=1, leave=False):
            ...  # TODO

        model.eval()
        for batch in tqdm(dev_dataloader, position=1, leave=False):
            ...  # TODO

