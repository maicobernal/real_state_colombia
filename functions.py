def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)



def normalize_column(df, column_name):
    df[column_name] = df[column_name].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    return df[column_name]



#https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
def clean_values(series, to_replace, value = '', regex = True):
    for i in to_replace:
        series = series.str.replace(i, value, regex=regex)
    return series


import requests

def get_lat_lon(address, access_key = '2e843c7ee44a8f52742a8168d0121a0a', URL = "http://api.positionstack.com/v1/forward"):
    PARAMS = {'access_key': access_key, 'query': address}
    r = requests.get(url = URL, params = PARAMS)
    data = r.json()
    return data['data'][0]['latitude'], data['data'][0]['longitude']


import pandas as pd

def run_exps(X_train: pd.DataFrame , y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
        '''
        Lightweight script to test many models and find winners
        :param X_train: training split
        :param y_train: training target vector
        :param X_test: test split
        :param y_test: test target vector
        :return: DataFrame of predictions
        '''
        
        dfs = []

        dt = DecisionTreeClassifier(max_depth=1)

        models = [
            ('LogReg', LogisticRegression()), 
            ('RF', RandomForestClassifier()),
            ('KNN', KNeighborsClassifier()),
            ('GNB', GaussianNB()),
            ('XGB', XGBClassifier()),
            ('ADA', AdaBoostClassifier(base_estimator=dt))
            ]
        results = []
        names = []

        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

        target_names = ['malignant', 'benign']

        for name, model in models:
                kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
                cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
                clf = model.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                print(name)
                print(classification_report(y_test, y_pred, target_names=target_names))
                
        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        final = pd.concat(dfs, ignore_index=True)
        return final