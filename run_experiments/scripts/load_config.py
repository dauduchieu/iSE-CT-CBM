def load_config(model_name,dataset):
    # --- A) Sélection et instanciation de la config ---
    if model_name == 'bert-base-uncased':
        if dataset == 'movies':
            from config_movies import Config as Config_movies
            config = Config_movies()
        elif dataset == 'agnews':
            from config_agnews import Config as Config_agnews
            config = Config_agnews()
        elif dataset == 'dbpedia':
            from config_dbpedia import Config as Config_dbpedia
            config = Config_dbpedia()
        elif dataset == 'medical':
            from config_medical import Config as Config_medical
            config = Config_medical()
        elif dataset == 'pubmed':
            from config_pubmed import Config as Config_pubmed
            config = Config_pubmed()
        elif dataset == 'stackoverflow':
            from config_stackoverflow import Config as Config_stackoverflow
            config = Config_stackoverflow()
        elif dataset == 'medical_2':
            from config_medical_2 import Config as Config_medical_2
            config = Config_medical_2()
        elif dataset == 'ecom':
            from config_ecom import Config as Config_ecom
            config = Config_ecom()
        elif dataset == 'sst2':
            from config_sst2 import Config as Config_sst2
            config = Config_sst2()
        elif dataset == 'uci':
            from config_uci import Config as Config_uci
            config = Config_uci()
        else:
            raise ValueError("Entrez un nom de dataset valide parmi ['movies','agnews','dbpedia','medical','pubmed','stackoverflow','medical_2','ecom','sst2','uci']")
    elif model_name == 'deberta-large':
        if dataset == 'movies':
            from config_movies_deberta import Config as Config_movies
            config = Config_movies()
        elif dataset == 'agnews':
            from config_agnews_deberta import Config as Config_agnews
            config = Config_agnews()
        elif dataset == 'dbpedia':
            from config_dbpedia_deberta import Config as Config_dbpedia
            config = Config_dbpedia()
        elif dataset == 'medical':
            from config_medical_deberta import Config as Config_medical
            config = Config_medical()
        elif dataset == 'pubmed':
            from config_pubmed_deberta import Config as Config_pubmed
            config = Config_pubmed()
        elif dataset == 'stackoverflow':
            from config_stackoverflow_deberta import Config as Config_stackoverflow
            config = Config_stackoverflow()
        elif dataset == 'medical_2':
            from config_medical_2_deberta import Config as Config_medical_2
            config = Config_medical_2()
        elif dataset == 'ecom':
            from config_ecom_deberta import Config as Config_ecom
            config = Config_ecom()
        elif dataset == 'sst2':
            from config_sst2_deberta import Config as Config_sst2
            config = Config_sst2()
        elif dataset == 'uci':
            from config_uci_deberta import Config as Config_uci
            config = Config_uci()
        else:
            raise ValueError("Entrez un nom de dataset valide parmi ['movies','agnews','dbpedia','medical','pubmed','stackoverflow','medical_2','ecom','sst2','uci']")
    elif model_name == 'gemma':
        if dataset == 'movies':
            from config_movies_gemma import Config as Config_movies
            config = Config_movies()
        elif dataset == 'agnews':
            from config_agnews_gemma import Config as Config_agnews
            config = Config_agnews()
        elif dataset == 'dbpedia':
            from config_dbpedia_gemma import Config as Config_dbpedia
            config = Config_dbpedia()
        elif dataset == 'medical':
            from config_medical_gemma import Config as Config_medical
            config = Config_medical()
        elif dataset == 'pubmed':
            from config_pubmed_gemma import Config as Config_pubmed
            config = Config_pubmed()
        elif dataset == 'stackoverflow':
            from config_stackoverflow_gemma import Config as Config_stackoverflow
            config = Config_stackoverflow()
        elif dataset == 'medical_2':
            from config_medical_2_gemma import Config as Config_medical_2
            config = Config_medical_2()
        elif dataset == 'ecom':
            from config_ecom_gemma import Config as Config_ecom
            config = Config_ecom()
        elif dataset == 'sst2':
            from config_sst2_gemma import Config as Config_sst2
            config = Config_sst2()
        elif dataset == 'uci':
            from config_uci_gemma import Config as Config_uci
            config = Config_uci()
        else:
            raise ValueError("Entrez un nom de dataset valide parmi ['movies','agnews','dbpedia','medical','pubmed','stackoverflow','medical_2','ecom','sst2','uci']")
    else:
        raise ValueError("Entrez un nom de modèle valide parmi ['bert-base-uncased','deberta-large', 'gemma']")

    return config
