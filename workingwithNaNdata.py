#dataset from: https://www.youtube.com/redirect?v=8p6XaQSIFpY&redir_token=8UG7srjfyHVY7HHXVvV1cbB80cx8MTUxODkzMDIyMkAxNTE4ODQzODIy&event=video_description&q=https%3A%2F%2Fpythonprogramming.net%2Fstatic%2Fdownloads%2Fmachine-learning-data%2Ftitanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
style.use('ggplot')


df= pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)   #drop body numbers as only dead people have body numbers
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)    #add 0 in place of NaNs
    
def handle_NaN(df):
    columns= df.columns.values

    for column in columns:
        text_digit_vals= {}
        def convert_to_int(val):
            return text_digit_vals[val]    #return the value

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:    #if data is non numeric:
            column_contents= df[column].values.tolist()   #convert to list
            unique_elements= set(column_contents)    #convert to set
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x    #fill in the empty dictionary
                    x+=1
            df[column]= list(map(convert_to_int, df[column]))    #returns the numeric value
            s
    return df

print(df.head())    #print here to see before the conversion
df= handle_NaN(df)
print("\nConverting...\n")    #lame
print(df.head())


