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