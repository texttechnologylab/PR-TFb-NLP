from transformers import AutoModel  # TODO: which AutoModelFor...?
from transformers import AutoConfig, AutoTokenizer

from tfbnlp.custom import run_custom
from tfbnlp.trainer import run_trainer


if __name__ == "__main__":
    config = ...  # TODO
    model = ...  # TODO
    tokenizer = ...  # TODO

    do_run_trainer = False
    if do_run_trainer:
        run_trainer(config, model, tokenizer)

    do_run_custom = False
    if do_run_custom:
        run_custom(config, model, tokenizer)
