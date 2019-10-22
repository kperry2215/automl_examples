import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

def run_h2o_automl(dataframe, variable_to_predict,
                   max_number_models):
    """
    This function initiates an h2o cluster, converts
    the dataframe to an h2o dataframe, and then runs
    the autoML function to generate a list of optimal 
    predictor models. The best models are displayed via a 
    scoreboard.
    Arguments:
        dataframe: Pandas dataframe. 
        variable_to_predict: String. Name of the dataframe that we're predicting.
        max_number_models: Int. Total number of models to run.
    Outputs:
        Leader board of best performing models in the console, plus performance of
        best fit model on the test data, including confusion matrix
    """
    h2o.init()
    #Convert the dataframe to an h2o dataframe
    dataframe = h2o.H2OFrame(dataframe)
    #Convert the variable we're predicting to a factor; otherwise this
    #will run as a regression problem
    dataframe[variable_to_predict] = dataframe[variable_to_predict].asfactor()
    #Declare the x- and y- variables for the database. 
    #x-variables are predictor variables, and y-variable is what
    #we wish to predict
    x = dataframe.columns
    y = variable_to_predict
    x.remove(y)
    #Pull the training/validation/test data out at a 75/12.5/12.5 split.
    train, test, validate = dataframe.split_frame(ratios=[.75, .125])
    # Run AutoML (limited to 1 hour max runtime by default)
    aml = H2OAutoML(max_models=max_number_models, seed=1)
    aml.train(x=x, y=y, training_frame = train, validation_frame = validate)
    # View the AutoML Leaderboard
    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))
    #Get performance on test data
    performance = aml.leader.model_performance(test)
    print(performance)
    
def run_tpot_automl(dataframe, 
                    variable_to_predict, 
                    number_generations,
                    file_to_export_pipeline_to = 'tpot_classifier_pipeline.py'):
    """
    This function runs a TPOT classifier on the dataset, after splitting into
    a training and test set, and then oversampling the training set.
    Args:
        dataframe: pandas dataframe. Master dataframe containing the feature and target
        data
        variable_to_predict: String. Name of the target variable that we want to predict.
        number_of_generations: Int. Number of generations to iterate through.
    Outputs:
        File containing the machine learning pipeline for the best performing model.
    """
    #Remove the target column to get the features dataframe
    features_dataframe = dataframe.loc[:, dataframe.columns != variable_to_predict]
    X_train, X_test, y_train, y_test = train_test_split(features_dataframe, dataframe[variable_to_predict],
                                                    train_size=0.75, test_size=0.25)
    #Run the TPOT pipeline
    tpot = TPOTClassifier(generations= number_generations, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export(file_to_export_pipeline_to)

if __name__ == "__main__" :
    #Read in the cancer data set
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data', 
                 header=None)
    #Declare the column names of the cancer data set
    df.columns=["Class", "Age", "Menopause",
            "Tumor_Size", "Inv_Nodes", 
            "Node_Caps", "Deg_Malig",
            "Breast", "Breast_quad",
            "Irradiat"]
    #Convert all of the categorical features variables to numeric (use LabelEncoder)
    d = defaultdict(LabelEncoder)    
    df_label_encoded = df.apply(lambda x: d[x.name].fit_transform(x))
    #Run the model through h2o's autoML function and 
    #generate a list of the best performing models
    run_h2o_automl(dataframe=df, 
                   variable_to_predict='Class',
                   max_number_models=10)
    #Run TPOT autoML
    run_tpot_automl(dataframe =  df_label_encoded, 
                    variable_to_predict = 'Class', 
                    number_generations =10)


