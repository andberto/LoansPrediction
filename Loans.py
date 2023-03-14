import pandas as pd
import numpy as np
import Constants
import Plot_utils as p_utils
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import Recall, Precision
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pymongo
import os

def load_data():
    path = 'Generated_files'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    srv_available = True
    client, dblist = None, None
    try: 
        client = pymongo.MongoClient(Constants.MONGODB_HOST, serverSelectionTimeoutMS = Constants.TIMEOUT)
        dblist = client.list_database_names()
    except pymongo.errors.ConnectionFailure as err:
        print("MongoDB server not found at " + Constants.MONGODB_HOST)
        srv_available = False

    if srv_available and "Loans_default" in dblist:
        print("MongoDB DB found...\nloading the data...")
        Loans_db = client["Loans_default"]
        Loans_collection = Loans_db["Loans_collection"]
        df = pd.DataFrame(list(Loans_collection.find()))
        df = df.drop("_id", axis = 1)
        print("Done!")
        return df
    else:
        print("No MongoDB DB found...\nGetting data from " + Constants.RAW_DATASET_PATH)
        return pd.read_csv(Constants.RAW_DATASET_PATH)

def view_dataset_info(df):
    #some infos
    print("Dataset shape -> Columns: " + str(len(df.columns) - 1) + " feat. + 1 targ.\nRows (samples): " + str(len(df)))
    target_percentage = df["Status"].value_counts(normalize=True) * 100
    print("Target variable: \"status\"\nNumber of unreliable borrowers: " + str(target_percentage[1]) + "%"
        + "\nNumber of reliable borrowers: " + str(target_percentage[0]) + "%")
    print("% of missing values -> " + str((df.isnull().any(axis=1).sum() * 100) / len(df.index)) + "%")

def get_dataset_mesures(df):
    missing_count = df.isnull().sum()  #missing values per column
    percent_missing = df.isnull().sum() * 100 / len(df) #percentage of missing values per column
    std = df.std(numeric_only = True, skipna= True)
    var = df.var(numeric_only = True, skipna= True)
    avg = df.mean(numeric_only = True, skipna= True)
    mode = (df.mode(dropna=True)).transpose()[0]
    med = df.median(numeric_only = True, skipna= True)
    dfi =  pd.DataFrame({'D_type': df.dtypes, 'Missing_count': missing_count, 'Percent_missing': percent_missing,
                         'Standard_deviation': std, 'Variance': var, 'Average': avg,'Median': med,'Mode': mode})
    return dfi

def map_categoricals(df):
    for col in df:
        if(not is_numeric_dtype(df[col])):
            uniques = df[col].unique()
            for i in range(len(uniques)):
                df[col].replace(uniques[i], i, inplace = True)
            print("Feature " + col + " mapping: " + str(uniques.tolist()) + " -> " + str([*range(len(uniques))]))

def fix_missing_values(df):
    #loan_limit, submission_of_application, approv_in_adv, Neg_ammortization
    #filled with the mode
    print("[loan_limit, submission_of_application, approv_in_adv, Neg_ammortization] -> missing values filled with mode")
    df["loan_limit"].fillna(df["loan_limit"].mode(dropna=True)[0], inplace=True)
    df["submission_of_application"].fillna(df["submission_of_application"].mode(dropna=True)[0], inplace=True)
    df["approv_in_adv"].fillna(df["approv_in_adv"].mode(dropna=True)[0], inplace=True)
    df["Neg_ammortization"].fillna(df["Neg_ammortization"].mode(dropna=True)[0], inplace=True)

    #age, loan_purpose remove the row
    print("[age, loan_purpose] -> Missing values rows dropped")
    df = df[df['age'].notna()]
    df = df[df['loan_purpose'].notna()]

    #numeric features with %of missing < 10% dropped, others filled with mean
    
    print("numericals -> missing < 10% dropped , misssing > 10% filled with mean")
    for col in df:
        if((df[col].isnull().sum() * 100 / len(df)) > 10 and is_numeric_dtype(df[col])):
            df[col].fillna(df[col].mean(skipna = True), inplace=True)
        elif((df[col].isnull().sum() * 100 / len(df)) < 10 and is_numeric_dtype(df[col])):
            df = df[df[col].notna()]
    print("% of missing values -> " + str((df.isnull().any(axis=1).sum() * 100) / len(df.index)) + "%")
    return df

def tsne_dimensionality_reduction(X, n_dim):
    return TSNE(n_components = n_dim, learning_rate='auto', init="random").fit_transform(X)

def fit_scaler(X_train_numeric):
    scaler = StandardScaler().fit(X_train_numeric)
    return scaler

def scale_features(X_numeric, fitted_scaler):
    return fitted_scaler.transform(X_numeric)

def get_MI_scores(X,y):
    kbest = SelectKBest(score_func=mutual_info_classif, k = Constants.K_BEST_FEATURES)
    fit = kbest.fit(X, y)
    scores = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(X.columns)
    df_best = X.iloc[:,kbest.get_support(indices = True)]
    df_scores = pd.concat([columns,scores],axis=1)
    df_scores.columns = ['Feature','MI_Score']
    df_scores.sort_values(by=['MI_Score'], inplace = True)
    return (df_best, df_scores)

def get_ml_models_scores(X_train, y_train, X_validation, y_validation):
    models_map, predictions_map = {}, {}
    scores_data = []
    models_map["logistic_regression"] = LogisticRegression()
    models_map["random_forest"] = RandomForestClassifier()
    models_map["ada_boost"] = AdaBoostClassifier()
    models_map["gradient_boosting"] = GradientBoostingClassifier()
    models_map["extreme_gradient_boosting"] = XGBClassifier(use_label_encoder=False, verbosity = 0)
    models_map["decision_tree"] = DecisionTreeClassifier()

    for key in models_map:
        print("Evaluating " + key + "...")
        models_map[key].fit(X_train.values, y_train)
        predictions_map[key] = models_map[key].predict(X_validation.values)
        scores_data.append([key, precision_score(y_validation, predictions_map[key]),
                            recall_score(y_validation, predictions_map[key]), 
                            f1_score(y_validation, predictions_map[key])])

    return pd.DataFrame(scores_data, columns=['Model','Precision', 'Recall', 'F1-Score'])

def get_ann_scores(X_train, y_train, X_validation, y_validation):
    print("Evaluating neural_network...")
    ann = models.Sequential()
    ann.add(layers.Dense(64, activation = 'relu', input_shape=(X_train.shape[1],)))
    ann.add(layers.Dense(48, activation = 'relu'))
    ann.add(layers.Dense(1, activation = 'sigmoid')) #classification => SIGMOID
    ann.compile(loss = 'binary_crossentropy', metrics=[Precision(), Recall()])
    history = ann.fit(X_train, y_train, validation_split = 0.2, verbose = 0)
    _, pr, rc = ann.evaluate(X_validation, y_validation)
    return ["neural_network", pr, rc,  (2 * pr * rc) / (pr + rc)]

def gradient_boosting_rnd_search(X_train, y_train, X_validation, y_validation):
    print("Performig a random search on n_estimators...")
    train_scores, validation_scores = {}, {}
    n_estimators_set = [5, 20, 50, 100, 250, 500, 750, 1000, 1500]
    
    for i in range(len(n_estimators_set)):
        GBC = GradientBoostingClassifier(n_estimators = n_estimators_set[i])
        print("trying n_estimators = " + str(n_estimators_set[i]))
        GBC.fit(X_train, y_train)
        ypv = GBC.predict(X_validation)
        ypt = GBC.predict(X_train)
        validation_scores[n_estimators_set[i]] = f1_score(y_validation, ypv)
        train_scores[n_estimators_set[i]] = f1_score(y_train, ypt)
    
    best_score = max(validation_scores.values())
    best_n_estimators = 0

    for key in validation_scores:
        if(validation_scores[key] == best_score):
            best_n_estimators = key
            break

    return best_n_estimators, train_scores, validation_scores

def gradient_boosting_grid_search(n_estimators, X_train, y_train, X_validation, y_validation):
    print("Performig a grid search on n_estimators...")
    train_scores, validation_scores = {}, {}
    for i in range(n_estimators - 50, n_estimators + 50, 5):
        GBC = GradientBoostingClassifier(n_estimators = i)
        print("trying n_estimators = " + str(i))
        GBC.fit(X_train, y_train)
        ypv = GBC.predict(X_validation)
        ypt = GBC.predict(X_train)
        validation_scores[i] = f1_score(y_validation, ypv)
        train_scores[i] = f1_score(y_train, ypt)

    best_score = max(validation_scores.values())
    final_estimators = 0
    
    for key in validation_scores:
        if(validation_scores[key] == best_score):
            final_estimators = key
            break

    return final_estimators, train_scores, validation_scores

def gradient_boosting_evaluation(X_train, y_train, X_test, y_test, estimators):
    final_model = GradientBoostingClassifier(n_estimators = estimators)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    scores_data = []
    scores_data.append(["Gradient boosting fine tuned", precision_score(y_test, y_pred),
                            recall_score(y_test, y_pred), 
                            f1_score(y_test, y_pred)])
    c_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)
    scores = pd.DataFrame(scores_data, columns=['Model','Precision', 'Recall', 'F1-Score'])

    return scores, c_matrix

def main():
    print("\n\nBig data & Business intelligence project")
    print("Andrea Bertogalli - 307673 - UNIPR - 2021/2022\n")
    df = load_data()

    # the ID and Year column are not usefull (explained in the document)
    df = df.drop("ID", axis = 1)
    df = df.drop("year", axis = 1)

    # print some infos about the dataset
    print("\nDATASET INFO:")
    view_dataset_info(df)

    # print on a csv all the important info
    mesures_df = get_dataset_mesures(df)
    mesures_df.to_csv('Generated_files/Loan_Default_Info.csv')

    # plot the charts related with each features
    p_utils.plot_mesures_table(mesures_df)
    p_utils.plot_features_charts(df.iloc[:,:16],4,4)
    p_utils.plot_features_charts(df.iloc[:,16:],4,4)

    # missing values fixing
    print("\nMISSING VALUES FIX:")
    df = fix_missing_values(df)
    mesures_df = get_dataset_mesures(df)
    mesures_df.to_csv('Generated_files/Loan_Default_Info_Cleaned.csv')
    p_utils.plot_mesures_table(mesures_df)

    # map categoricas string values into numeric
    print("\nCATEGORICAL FEATURES ENCODING:")
    map_categoricals(df) #inplace
    df.to_csv('Generated_files/Loan_Default_Cleaned.csv')

    # retrieve the feature matrix and the target vector
    unsplitted_df = df.copy()
    y, X = df.pop(Constants.TARGET), df

    # features selection
    print("\nFEATURES SELECTION:")
    X, MI_scores = get_MI_scores(X, y)
    MI_scores.to_csv('Generated_files/Loan_Default_MI_Scores.csv')
    print("MI Scores:")
    print(MI_scores)
    print("The best " + str(Constants.K_BEST_FEATURES) + " features are:")
    print(X.head())
    p_utils.plot_MI_scores(MI_scores)
    
    # train test_split
    print("\nSPLITING INT TRAIN TEST VALIDATION (30% test, 70% train, (10% validation from training))")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = Constants.TEST_PERC)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = Constants.VALIDATION_PERC)
    
    # features scaling (Scaler fitted only on test set numeric colums, 
    # then scale test and train numeric cols, the y doesn't need scaling)
    # the variances and the means are about 1 and 0 but not perfectly 1 and 0
    # but if we round we obtain 0 and 1 so is an approximation problem
    print("\nFEATURES SCALING (Z-SCORE):")
    scaler = fit_scaler(X_train[Constants.SCALABLE])
    X_train[Constants.SCALABLE] = scale_features(X_train[Constants.SCALABLE], scaler)
    X_test[Constants.SCALABLE] = scale_features(X_test[Constants.SCALABLE], scaler)
    X_validation[Constants.SCALABLE] = scale_features(X_validation[Constants.SCALABLE], scaler)
    p_utils.plot_scaled_plot(X_train[Constants.SCALABLE])
    print("Now std of numeric cols is: ")
    print(abs(round(X_train[Constants.SCALABLE].var())))
    print("Now mean of numeric cols is: ")
    print(abs(round(X_train[Constants.SCALABLE].mean())))

    # data visualization with TSNE on the entire dataset
    # the y is used only for the visualization, tsne is unsupervised...
    print("\nDATA VISUALIZATION WITH T-SNE:")
    tsne_embedded_X = tsne_dimensionality_reduction(X_train, Constants.TSNE_2D)
    p_utils.plot_tsne_scatter(tsne_embedded_X, y_train, Constants.TSNE_2D)
    print("2d virtual space plotted!")
    #uncomment to have 3d tsne plot 
    '''
    tsne_embedded_X = tsne_dimensionality_reduction(X_train, Constants.TSNE_3D)
    p_utils.plot_tsne_scatter(tsne_embedded_X, y_train, Constants.TSNE_3D)
    print("3d virtual space plotted!")
    '''

    # machine learning model selection between logistic_regression, random_forest, ada_boost
    # gradient_boosting, extreme_gradient_boosting, decision_tree and neural network
    print("\nBEST MODEL SELECTION:")
    print("getting models score...")
    models_scores = get_ml_models_scores(X_train, y_train, X_validation, y_validation)
    models_scores.loc[len(models_scores.index)] = get_ann_scores(X_train, y_train, X_validation, y_validation)
    p_utils.plot_models_scores(models_scores)
    models_scores.to_csv('Generated_files/Loan_Default_ML_Models_Scores.csv')

    # best model fine tuning without high correlation features
    print("\nBEST MODEL FINE TUNING (Gradient boosting):")
    X_tuning_train = X_train.drop(Constants.HIGH_CORRELATION, axis = 1)
    X_tuning_vaidation = X_validation.drop(Constants.HIGH_CORRELATION, axis = 1)
    rnd_est, ts, vs = gradient_boosting_rnd_search(X_tuning_train, y_train, X_tuning_vaidation, y_validation)
    rnd_est = 500 #best on random search (comment to re-random search)
    p_utils.plot_fitting_chart(ts, vs, "GBC n_estimators", "F1-Score", "n_estimators random search")
    n_estimators, ts, vs = gradient_boosting_grid_search(rnd_est, X_tuning_train, y_train, X_tuning_vaidation, y_validation)
    n_estimators = 520 #best on grid search (comment to re-gridsearch)
    p_utils.plot_fitting_chart(ts, vs, "GBC n_estimators", "F1-Score", "n_estimators grid search")
    print("The best n_estimators is: " + str(n_estimators))

    # final evaluation
    print("\nFINAL EVALUATION (Gradient boosting):")
    GBC_scores, confusion_matrix = gradient_boosting_evaluation(X_train, y_train, X_test, y_test, n_estimators)
    print("confusion matrix:")
    print(confusion_matrix)
    print("final metrics:")
    print(GBC_scores)
    p_utils.plot_models_scores(GBC_scores)
    p_utils.plot_confusion_matrix(confusion_matrix)

main()