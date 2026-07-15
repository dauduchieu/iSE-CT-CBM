from data_hf_text_classification import prepare_hf_text_classification_data


STACKOVERFLOW_LABELS = ["HQ", "LQ_EDIT", "LQ_CLOSE"]


def prepare_stackoverflow_data(config):
    return prepare_hf_text_classification_data(
        config=config,
        dataset_name="Duyacquy/Stack_overflow_question",
        text_col="Text",
        label_col="Y",
        label_names=STACKOVERFLOW_LABELS,
        cache_name="stackoverflow",
        max_examples_per_class=getattr(config, "max_examples_per_class", 1000),
    )
