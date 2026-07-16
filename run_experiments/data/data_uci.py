from data_hf_text_classification import prepare_hf_text_classification_data


UCI_LABELS = ["1", "5", "10"]


def prepare_uci_data(config):
    return prepare_hf_text_classification_data(
        config=config,
        dataset_name="Duyacquy/UCI_drug",
        text_col="review",
        label_col="rating",
        label_names=UCI_LABELS,
        cache_name="uci",
        max_examples_per_class=getattr(config, "max_examples_per_class", 1000),
    )
