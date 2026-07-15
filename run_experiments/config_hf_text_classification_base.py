import re

import torch
import torch.nn as nn


class HFTextClassificationConfig:
    annotation = "our_annotation"
    infra = "A100"
    mode = "joint"

    model_name = "bert-base-uncased"
    dataset_name = "hf_text_classification"
    DATASET = "hf_text_classification"
    num_labels = 2

    dim = 1024 if re.search(r"large", model_name, re.IGNORECASE) else 768
    max_len = 256
    batch_size = 16
    lambda_XtoC = 0.5
    is_aux_logits = False
    num_epochs = 10
    num_each_concept_classes = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expand_dim = 0
    num_concept_labels = 4
    seed = 42
    n_concept_initial = 1
    max_examples_per_class = 1000

    DATASET_PATH = f"/home/bhan/Yann_CBM/Launch/dbfs/dataset/{DATASET}/"
    SAVE_PATH = f"/home/bhan/Yann_CBM/Launch/dbfs/results_{DATASET}/"
    SAVE_PATH_CONCEPTS = f"{SAVE_PATH}concepts_discovery"

    criterion = nn.CrossEntropyLoss()
    use_cls_token = True
    lr = 0.001

    alpha = 0.01
    l1_ratio = 0.5
    l2_lambda = 0.01

    use_sigmoid = False
    use_relu = False
    eval_concepts = True
    top_n = 1

    if use_sigmoid:
        sigmoid_or_relu_state = "sigmoid"
    elif use_relu:
        sigmoid_or_relu_state = "relu"
    else:
        sigmoid_or_relu_state = "linearity"

    dropout = 0.1
    projection = 256

    cavs_type_arg = "mean"
    agg_mode = "abs"
    agg_scope = "all"
    cavs_type = "mean"
