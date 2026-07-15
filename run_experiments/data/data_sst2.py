from data_hf_text_classification import prepare_hf_text_classification_data


SST2_LABELS = ["0", "1"]


def prepare_sst2_data(config):
    return prepare_hf_text_classification_data(
        config=config,
        dataset_name="SetFit/sst2",
        text_col="text",
        label_col="label",
        label_names=SST2_LABELS,
        cache_name="sst2",
        max_examples_per_class=getattr(config, "max_examples_per_class", 1000),
    )
