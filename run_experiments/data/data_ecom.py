from data_hf_text_classification import prepare_hf_text_classification_data


ECOM_LABELS = ["Books", "Clothing & Accessories", "Household", "Electronics"]


def prepare_ecom_data(config):
    return prepare_hf_text_classification_data(
        config=config,
        dataset_name="Duyacquy/Ecommerce-text",
        text_col="text",
        label_col="label",
        label_names=ECOM_LABELS,
        cache_name="ecom",
        max_examples_per_class=getattr(config, "max_examples_per_class", 1000),
    )
