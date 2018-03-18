#dataset from: https://www.youtube.com/redirect?v=8p6XaQSIFpY&redir_token=8UG7srjfyHVY7HHXVvV1cbB80cx8MTUxODkzMDIyMkAxNTE4ODQzODIy&event=video_description&q=https%3A%2F%2Fpythonprogramming.net%2Fstatic%2Fdownloads%2Fmachine-learning-data%2Ftitanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd
style.use('ggplot')


df= pd.read_excel('titanic.xls')
original_df= pd.DataFrame.copy(df)

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

X= np.array(df.drop(['survived'], 1).astype(float))    #'1' indicates drop the column
X= preprocessing.scale(X)    #feature scaling the data
#preprocessing improved accuracy from 0.5x to 0.7x. WOAH!
y= np.array(df['survived'])

clf= MeanShift()
clf.fit(X)

labels= clf.labels_
cluster_centers= clf.cluster_centers_

original_df['cluster_group']= np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i]= labels[i]    #iloc[i] indicates row number i

n_clusters_= len(np.unique(labels))    #number of cluster is equal to the number of unique labels

survival_rates={}    #key- cluster_group and value- survival rate
for i in range(n_clusters_):
    temp_df= original_df[(original_df['cluster_group'] == float(i))]    #this is kinda like an if statement
    survival_cluster= temp_df[(temp_df['survived'] == 1)]    #has the elements where people survived
    survival_rate= len(survival_cluster)/len(temp_df)
    survival_rates[i]= survival_rate

#the answer is not gonna be the same every time
print(survival_rates)
    





























