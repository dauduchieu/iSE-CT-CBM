from config_hf_text_classification_base import HFTextClassificationConfig


class Config(HFTextClassificationConfig):
    model_name = "bert-base-uncased"
    DATASET = "ecom"
    dataset_name = "ecom"
    num_labels = 4
    max_len = 256
    batch_size = 16
    dim = 1024 if model_name == "deberta-large" else 768
    DATASET_PATH = f"/home/bhan/Yann_CBM/Launch/dbfs/dataset/{DATASET}/"
    SAVE_PATH = f"/home/bhan/Yann_CBM/Launch/dbfs/results_{DATASET}/"
    SAVE_PATH_CONCEPTS = f"{SAVE_PATH}concepts_discovery"
