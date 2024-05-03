from datasets import load_dataset, DatasetDict


def load_mnli_dataset() -> DatasetDict:
    return load_dataset(
        "glue",
        "mnli",
        split={
            "train": "train[:10%]",
            "validation": "validation_matched",
            "test": "test_matched",
        },
    )


def prepare_dataset(
    dataset: DatasetDict,
) -> DatasetDict: ...  # TODO: Implement this function
