from data_hf_text_classification import prepare_hf_text_classification_data


MEDICAL_2_LABELS = ["1", "2", "3", "4", "5"]


def prepare_medical_2_data(config):
    return prepare_hf_text_classification_data(
        config=config,
        dataset_name="Duyacquy/Single-label-medical-abstract",
        text_col="medical_abstract",
        label_col="condition_label",
        label_names=MEDICAL_2_LABELS,
        cache_name="medical_2",
        max_examples_per_class=getattr(config, "max_examples_per_class", 1000),
    )
