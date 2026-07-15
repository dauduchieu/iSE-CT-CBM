from data_hf_text_classification import prepare_hf_text_classification_data


PUBMED_LABELS = ["METHODS", "RESULTS", "CONCLUSIONS", "BACKGROUND", "OBJECTIVE"]


def prepare_pubmed_data(config):
    return prepare_hf_text_classification_data(
        config=config,
        dataset_name="Duyacquy/Pubmed_20k",
        text_col="abstract_text",
        label_col="target",
        label_names=PUBMED_LABELS,
        cache_name="pubmed",
        max_examples_per_class=getattr(config, "max_examples_per_class", 1000),
    )
