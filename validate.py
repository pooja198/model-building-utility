from configobj import ConfigObj
import os
import pandas as pd
import sys
import yaml


ALLOWED_EXTENSIONS = set(['csv'] )
# folder to load config file
CONFIG_PATH = "./config/"

class MyException( Exception ):

    pass

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_config(config_name):

    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


class MyConfig(ConfigObj):

    def __init__(self, config_file,model_name):
        super(MyConfig, self).__init__()
        self.model_name = model_name
        self.config = load_config(config_file)
        self.check_file_exists()
        self.validate_config()

    def check_file_exists(self):
        print(self.config)
        dataset_path=os.path.join(self.config["data_directory"],self.config["data_name"] )

        if os.path.exists( dataset_path ) and allowed_file( dataset_path ):
            print("DataSet present")

        else:
            print( "Dataset File {} does not exist or not Valid".format( dataset_name ) )
            exit( 0 )

    def validate_config(self):
        required_values = {
        'data_directory':None,
        'data_name' : None,
        'drop_columns': None,
        'target_name': None,
        'test_size' : None,
        'model_directory' :None,
        'encode_data':None,
        'model_name': None,
        'model_file_name': None,
        'metric_file_name': None,
        'KNN': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 15,
            'p': 2,
            'metric': 'minkowski',
            'n_jobs': 1
        },
        'LogisticRegression': {
            'penalty': 'l2',
            'dual': False,
            'tol': 0.0001,
            'C': 1,
            'multi_class': 'auto'
        },
        'SVM': {
            'C': 1,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale'
        },
        'DecisionTree': {
            'criterion': 'gini',
            'splitter': 'best',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0,
            'ccp_alpha': 0
        }
        }

        for key,val in required_values.items():
            if key not in self.config:
                sys.exit( 'Missing parameter {} in the config file'.format(key))
                for param, values in keys.items():
                    if param not in self.config[key]:
                        sys.exit( 'Missing value for {} under section {} in the config file'.format(param,key))
