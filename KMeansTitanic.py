#dataset from: https://www.youtube.com/redirect?v=8p6XaQSIFpY&redir_token=8UG7srjfyHVY7HHXVvV1cbB80cx8MTUxODkzMDIyMkAxNTE4ODQzODIy&event=video_description&q=https%3A%2F%2Fpythonprogramming.net%2Fstatic%2Fdownloads%2Fmachine-learning-data%2Ftitanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
style.use('ggplot')


df= pd.read_excel('titanic.xls', skipinitialspace= True)
#'skipinitialspace'as found on stackoverflow to delete the initial space to solve the ValueError. Does not seem to work
#---UPDATE--- ValueError solved, I was doing df.drop(['label', 1]) whereas it is df.drop(['label'], 1) duh me.

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
            
    return df

df= handle_NaN(df)  
#print(df.head())
#This was just the handling non numeric data part. Now comes KMeans



df.drop(['ticket', 'boat', 'sex'], 1)    #try dropping different stuff to see how it affects accuracy

X= np.array(df.drop(['survived'], 1).astype(float))    #'1' indicates drop the column
X= preprocessing.scale(X)    #feature scaling the data
#preprocessing improved accuracy from 0.5x to 0.7x. WOAH!
y= np.array(df['survived'])

clf= KMeans(n_clusters=2)
clf.fit(X)

correct=0
for i in range(len(X)):
    predict_me=np.array(X[i].astype(float))
    predict_me= predict_me.reshape(-1, len(predict_me))
    prediction= clf.predict(predict_me)
    if prediction[0] == y[i]:    #if predicted is same as survived
        correct +=1
        
print(correct/len(X))























