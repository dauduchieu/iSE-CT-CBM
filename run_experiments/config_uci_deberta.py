from config_hf_text_classification_base import HFTextClassificationConfig


class Config(HFTextClassificationConfig):
    model_name = "deberta-large"
    DATASET = "uci"
    dataset_name = "uci"
    num_labels = 3
    max_len = 256
    batch_size = 8
    dim = 1024 if model_name == "deberta-large" else 768
    DATASET_PATH = f"/home/bhan/Yann_CBM/Launch/dbfs/dataset/{DATASET}/"
    SAVE_PATH = f"/home/bhan/Yann_CBM/Launch/dbfs/results_{DATASET}/"
    SAVE_PATH_CONCEPTS = f"{SAVE_PATH}concepts_discovery"
