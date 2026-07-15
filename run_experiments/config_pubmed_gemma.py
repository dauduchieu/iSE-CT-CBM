from config_hf_text_classification_base import HFTextClassificationConfig


class Config(HFTextClassificationConfig):
    model_name = "gemma"
    DATASET = "pubmed"
    dataset_name = "pubmed"
    num_labels = 5
    max_len = 256
    batch_size = 4
    dim = 1024 if model_name == "deberta-large" else 768
    DATASET_PATH = f"/home/bhan/Yann_CBM/Launch/dbfs/dataset/{DATASET}/"
    SAVE_PATH = f"/home/bhan/Yann_CBM/Launch/dbfs/results_{DATASET}/"
    SAVE_PATH_CONCEPTS = f"{SAVE_PATH}concepts_discovery"
