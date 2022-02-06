# Import important packages
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn.metrics import  precision_score#for checking the model accuracy
from sklearn.metrics import recall_score,accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from utils.validate import MyConfig,MyException
from configobj import ConfigObj
import joblib
import argparse
import os,csv

import datetime,time
import yaml,random
from jinja2 import Environment, FileSystemLoader


# folder to load config file
CONFIG_PATH = "./config/"

config = ConfigObj("config.yaml")
number = random.randint(1000,9999)
ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%M%d%H%M%S')

# Function to load yaml configuration file

def load_config(config_name):

    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


def build_knn_classifier(X_train,y_train):

    # call our classifer and fit to our data
    model = KNeighborsClassifier(
        n_neighbors=config[model_name]["n_neighbors"],
        weights=config[model_name]["weights"],
        algorithm=config[model_name]["algorithm"],
        leaf_size=config[model_name]["leaf_size"],
        p=config[model_name]["p"],
        metric=config[model_name]["metric"],
        n_jobs=config[model_name]["n_jobs"],
    )
    # training the classifier
    model.fit( X_train, y_train )
    return model

def build_logistic_regression_model(X_train,y_train):

    model = LogisticRegression(penalty=config[model_name]["penalty"],
                               dual=config[model_name]["dual"],
                               tol=config[model_name]["tol"],
                               C=config[model_name]["C"],
                               multi_class=config[model_name]["multi_class"])
    model.fit( X_train, y_train)
    return model


def build_decision_tree_classifier(X_train,y_train):

    model = DecisionTreeClassifier( criterion= config[model_name]["criterion"],
                                    splitter=config[model_name]["splitter"],
                                    min_samples_split=config[model_name]["min_samples_split"],
                                    min_samples_leaf=config[model_name]["min_samples_leaf"],
                                    min_weight_fraction_leaf=config[model_name]["min_weight_fraction_leaf"],
                                    ccp_alpha=config[model_name]["ccp_alpha"])
    model.fit(  X_train, y_train)
    return model

def build_svm_classifier( X_train, y_train):
    print("IN SVM {}",config[model_name]["C"])
    model = svm.SVC(C = config[model_name]["C"],
                    kernel = config[model_name]["kernel"],
                    degree = config[model_name]["degree"],
                    gamma = config[model_name]["gamma"])
    model.fit( X_train, y_train )
    return model


def save_model_metrics(file_name, result):

    if os.path.exists(os.path.join("./results/",file_name)):
        path=os.path.join("./results/",file_name)
        df=pd.read_csv(path)
        print(df)
        df2=pd.DataFrame(np.array(result).reshape(-1,len(result)),columns=["accuracy","precision","recall","f1","model_name"])
        df=df.append(df2,ignore_index=True)
        print(df)
        df.to_csv( f"./results/{file_name}", index=False )
        print( " Metrics appended to ", file_name )
    else:
        df=pd.DataFrame(np.array(result).reshape(-1,len(result)),columns=["accuracy","precision","recall","f1","model_name"])
        df.to_csv(f"./results/{file_name}",index=False)
        print(" Metrics saved to new file ",file_name)

def generate_results(predictions, name):

    cl_rep = classification_report(y_test, predictions)
    print("\nThe classification report for " + name + " is:", cl_rep, sep = "\n")
    cm_model = confusion_matrix(y_test, predictions)
    plt.figure(figsize = (8, 6))
    sns.heatmap(cm_model, annot = True, cmap = 'Blues', annot_kws = {'size': 15}, square = True)
    plt.title('Confusion Matrix for ' + name, size = 15)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    # plt.show()
    plt.savefig(f'./results/visualizations/{name}-{timestamp}.png')

def test_classifier(model_name,model,X_test,y_test):

    y_pred = model.predict( X_test)
    accuracy=accuracy_score(y_pred,y_test)
    precision=precision_score(y_test, y_pred,pos_label='positive',average='micro')
    recall=recall_score(y_test, y_pred,pos_label='positive',
                                           average='micro')
    f1=f1_score(y_test, y_pred,pos_label='positive',average='micro')
    result=[accuracy,precision,recall,f1,model_name+f"-{timestamp}"]
    generate_results(y_pred,model_name)
    save_model_metrics(config["metric_file_name"],result)


    print( "Accuracy score is. {:.1f}".format( accuracy ) )

def compare_all_models():
    data=pd.read_csv("./results/Metrics.csv")
    fig = plt.figure( figsize=(15, 10) )
    sns.barplot( y=data['model_name'], x=data['accuracy'] )
    plt.xlabel( "Score", size=20 )
    plt.xticks( size=12 )
    plt.ylabel( "Model Used", size=20 )
    plt.yticks( size=10 )
    plt.title( "Score for Different Models", size=25 )
    # plt.show()
    plt.savefig( f'./results/Score_for_different_Models.png' )

def  validate_config():
    cfg = {}
    try:
        cfg = MyConfig('config.yaml',model_name=model_name )
    except MyException as e:
        print(e)
    print( "Config File Validated" )

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--modelName",
       help="Provide the ML Algorithm Name you want to use for Classification\n "
            "Valid arguments are ['KNN', 'SVM', 'LogisticRegression', 'DecisionTree']")
    ap.add_argument("-c","--compareAll",default=False)
    args = vars(ap.parse_args())

    model_name=args['modelName']

    config = load_config("config.yaml")
    config['model_name'] = model_name

    validate_config()
    print(args['compareAll'],args)

    if  (args['modelName']==None) and (args['compareAll']=='True'):
        print("enter")
        compare_all_models()
        exit(0)


    else:
        # load data
        data = pd.read_csv(os.path.join('./'+config["data_directory"], config["data_name"]))

        # replace "?" with -99999
        data = data.replace("?", -99999)

         #  drop id column
        data = data.drop(config["drop_columns"], axis=1)

        # Define X (independent variables) and y (target variable)
        X = np.array(data.drop(config["target_name"], 1))
        y = np.array(data[config["target_name"]])

        # split data into train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["test_size"], random_state=42
        )

        sc = StandardScaler()

        sc.fit( X_train )

        X_train_std = sc.transform( X_train )
        X_test_std = sc.transform( X_test )

        classfier_model=""
        if args['modelName']=='KNN':
            classfier_model=build_knn_classifier(X_train,y_train)
        elif args['modelName']=='LogisticRegression':
            classfier_model=build_logistic_regression_model(X_train,y_train)
        elif args['modelName']=='DecisionTree':
            classfier_model=build_decision_tree_classifier(X_train,y_train)
        elif args['modelName']=='SVM':
            classfier_model=build_svm_classifier(X_train,y_train)

        #extract metrics for classification problems and save it to csv
        test_classifier(model_name,classfier_model,X_test,y_test)
        # save our classifier in the model directory
        joblib.dump(classfier_model, os.path.join(config["model_directory"], model_name+"-"f"{timestamp}"+".pkl"))

