import pandas as pd
import quandl, math, datetime
import matplotlib.pyplot as plt
from  matplotlib import style
import numpy as np
from sklearn import preprocessing as pp
from sklearn import model_selection as cv   #from sklearn import cross_validation has been replaced in  sklearn 0.20
from sklearn import svm
from sklearn.linear_model import LinearRegression
import pickle

style.use('ggplot') #Gives a stylish plot 
df= quandl.get('WIKI/GOOGL')    #df is dataframe
#DataFrame is a 2-dimensional labeled data structure
#with columns of potentially different types. You can
#think of it like a spreadsheet or SQL table, or a dict of Series objects.

#Adj= Adjusted after stock splits and stuff
#df is a list
df= df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT']= (df['Adj. High']- df['Adj. Open'])/ df['Adj. Open']*100
df['PCT_change']= (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100
df= df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]   #get rid of NaNs by df.fillna/dropna


forecast_col= 'Adj. Close'
#This is not NaN data, so you replace all NaNs by -99999; you can also get rid of it
df.fillna(-99999, inplace=True)

forecast_out= int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)  #Go forecast_out amount of days back so that the ylabel is the Adj. Close the same days into the future. 


X= np.array(df.drop(['label'],1))
X= pp.scale(X)  #feature scaling
X_lately= X[-forecast_out:]  #X_lately has all the values without labels
X= X[:-forecast_out]    #go the the point -forecast_out i.e the point till which  we have labels 
#df.dropna(inplace=True)
#y= np.array(df['label'])
df.dropna(inplace=True)     #drop all the nans
y= np.array(df['label'])


#np.isnan(X)- returns a boolean mask back with True for positions containing NaNs
np.where(np.isnan(X)) #get a tuple with i,j coordinates of NaNs
X= np.nan_to_num(X)
#Shuffle and split, take 20% of the data as training data, rest is test data
X_train, X_test, y_train, y_test= cv.train_test_split(X, y, test_size=0.2)

clf= LinearRegression()    #n_jobs= -1 as parameter for max CPU util
#fit is analogous to train:
clf.fit(X_train, y_train)   #THis training set takes the most time
##with open('linearreg.pickle', 'wb') as f:
##    pickle.dump(clf, f)

#score is analogous to test
accuracy= clf.score(X_test, y_test)
print("Accuracy is: {}".format(accuracy))
print("Prices for the next {} days\n".format(forecast_out))
forecast_set= clf.predict(X_lately) #predict the stuff that does not have the label
print(forecast_set, accuracy, forecast_out)    #print the price forecast, accuracy of LR and no. of days(forecast_out)

#Doing this to list the data out in proper datewise format
#Populating the dataframe with the dates and forecast values
df['Forecast']=np.nan

last_date= df.iloc[-1].name
last_unix= last_date.timestamp()
one_day= 86400
next_unix= last_unix+one_day

for i in forecast_set:
    next_date= datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]= [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)   #loc=4 is bottom right
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()










