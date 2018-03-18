#SVMs are classifiers used to classify 2 or more types/segments of data.
#XW+b=1 foro positive class and XW+b=-1 for a negative class; XW+b=0 on the decision boundary
#Support vectors are the points lying on the hyperplanes
#Separating hyperplane can be calculated by width/2 or margin/2 i.e distance
#between the 2 hyperplanes by 2

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class SVM:
    def __init__(self, visualization=True):
        self.visualization= visualization
        self.colors= {1:'r', -1:'b'}
        if self.visualization:
            self.fig= plt.figure()
            self.ax= self.fig.add_subplot(1,1,1)
            
    #training
    def fit(self, data):
        self.data=data
        #opt_dict will store the minimized norm of vector w as the key and
        #the corresponding w,b values as the values
        opt_dict= {}

        transforms= [
                     [1,1], [1,-1],[-1,1],[-1,-1]
                    ]

        all_data=[]
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value= max(all_data)
        self.min_feature_value= min(all_data)
        all_data= None

        #varying step sizes to be smaller and smaller as you get close to the global minimum
        step_sizes= [self.max_feature_value*0.1,
                     self.max_feature_value*0.01,
                     self.max_feature_value*0.001]

        #extremely expensive:
        b_range_multiple= 2    #5 initially

        b_multiple=5

        latest_optimum= self.max_feature_value*10
        
        for step in step_sizes:
            w= np.array([latest_optimum, latest_optimum])
            #we can do  this because the function is convex
            #we know that it can be optimized as there exists a global mimimum
            optimized= False

            while not optimized:    #np.arange is like range
                #for loop goes from -x to +x with steps of step*b_multiple, each step is costly
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), self.max_feature_value*b_range_multiple, step*b_multiple):
                    for transformation in transforms:
                        w_t=w*transformation
                        found_option=True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option= False
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)]= [w_t, b]
                if w[0] <0:        #with our step, we are more than likely to skip w=0 i.e absolute minimum, so we test for w<0 rather than w == 0
                    optimized=True
                    print("Optimized a step")           
                else:
                    w=w-step
            norms=sorted([n for n in opt_dict])
            opt_choice= opt_dict[norms[0]]

            self.w= opt_choice[0]
            self.b= opt_choice[1]
            latest_optimum= opt_choice[0][0]+step*2
            

    def predict(self, features):
        #sign of this->(x.w+b)
        classification= np.sign(np.dot(np.array(features), self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker= '*', c=self.colors[classification])
        
        return classification
    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color= self.colors[i]) for x in data_dict[i]] for i in data_dict]
        
        #hyperplane= x.w + b
        #v=x.w + b
        #psv=1
        #nsv=-1
        #decision bdry= 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v)/w[1]

        datarange= (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min= datarange[0]
        hyp_x_max= datarange[1]

        #positive support vector; w.x + b =1
        psv1= hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2= hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')    #k is black

        #negative support vector: w.x + b = -1
        nsv1= hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2= hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')
        db1= hyperplane(hyp_x_min, self.w, self.b, 0)
        db2= hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')    #y-- is yellow dashed linem, resembling a road

        plt.show()
        
#----SVM class ends----#


data_dict= {-1:np.array([[1,7], [2,8], [3,8]]),
            1:np.array([[5,1], [6,-1], [7,3]])
           }

svm= SVM()    #initiatize object svm to SVM class
svm.fit(data=data_dict)

predict_us= [[0,10], [1,2], [4,5], [-5,3], [3,5], [6,-7], [3,-8]]
for p in predict_us:
    svm.predict(p)


svm.visualize()














