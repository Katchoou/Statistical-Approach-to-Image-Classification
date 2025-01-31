import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
import pathlib
from glob import glob
from scipy import stats
import math
from skimage.feature import graycomatrix, graycoprops
import cv2
import seaborn as sns
%matplotlib notebook
%matplotlib inline 

# image + filtre de bande differente, 
# regarder en moyenne 
#1
nbit=8
L=2**nbit
def probabilities(hist, nbre_pixel_image):
    prob=[card_B/nbre_pixel_image for card_B in hist]
    return prob
    
#2
def expectation(prob, levels):
    e=0
    for i in range(L-1):
        e+=prob[i]*levels[i]
    return e
    
#3
def variance(e, prob):
    var=0
    for i in range(L-1):
        var+=prob[i]*(i-e)**2
    return var

#4
def asymetry(e, prob):
    asy=0
    for i in range(L-1):
        asy+=prob[i]*(i-e)**3
    return asy

#5
def entropy(prob):
    entropy=0
    for i in range(L-1):
        if prob[i]>0:
            entropy+=prob[i]*math.log2(prob[i])
    return -1*entropy

#6
def excess(e, prob):
    excess=0
    for i in range(L-1):
        excess+=prob[i]*(i-e)**4
    return excess

#7
def homogeneity(prob):
    homo=0
    for i in range(L-1):
        homo+=prob[i]**2
    return homo

#8
def skewness(variance, asymetry):
    sigma=np.sqrt(variance)
    skew=asymetry/sigma**3
    return skew

#9
def kurtosis(var, excess):
    return excess/(var**2)  

#                                     Second-order statistics

def GLCM(image):
    glcm=graycomatrix(image, distances=[1], 
                     angles=[0, np.pi/4], levels=256,
                    symmetric=True, normed=True)
    glcm_mean=glcm.mean(axis=3)
    GLCM=glcm_mean.reshape(256, 256, 1,1)
    return GLCM

# definition of the repository 
direct_plane='E:\\AI\\Machine_Learning\\Image-classification\\Emotion_recognition\\dataset\\planes'
direct_car='E:\\AI\\Machine_Learning\\Image-classification\\Emotion_recognition\\dataset\\cars'
direct_train='E:\\AI\\Machine_Learning\\Image-classification\\Emotion_recognition\\dataset\\trains'

data1={'expectations':[], 'variance':[], 'asymetry':[], 'entropy':[], 
      'excess':[], 'homogeneity1':[], 'skewness':[], 'kurtosis':[], 'contrast':[],
     'correlation':[], 'homogeneity2':[], 'energy':[], 'dissimilarity':[], 'ASM':[]}
data2={'expectations':[], 'variance':[], 'asymetry':[], 'entropy':[], 
      'excess':[], 'homogeneity1':[], 'skewness':[], 'kurtosis':[], 'contrast':[],
     'correlation':[], 'homogeneity2':[], 'energy':[], 'dissimilarity':[], 'ASM':[]}
data3={'expectations':[], 'variance':[], 'asymetry':[], 'entropy':[], 
      'excess':[], 'homogeneity1':[], 'skewness':[], 'kurtosis':[], 'contrast':[],
     'correlation':[], 'homogeneity2':[], 'energy':[], 'dissimilarity':[], 'ASM':[]}

# Extraction function of the statistical properties 

def lonz_nematode(directory, data):
    for file in glob(os.path.join(directory, '*.jpg')):
        image=cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image=cv2.resize(image, (300, 300))
        x_len, y_len=image.shape
        hist, levels=np.histogram(image.ravel(), bins=255, range=[0, 255])
        nbre_pixel_image=x_len*y_len
        probability=probabilities(hist, nbre_pixel_image)
        expect=expectation(probability, levels)
        data["expectations"].append(expect)
        var=variance(expect, probability)
        data["variance"].append(var)
        asym=asymetry(expect, probability)
        data["asymetry"].append(asym)
        entrpy=entropy(probability)
        data["entropy"].append(entrpy)
        excs=excess(expect, probability)
        data["excess"].append(excs)
        homo1=homogeneity(probability)
        data["homogeneity1"].append(homo1)
        skew=skewness(var, asym)
        data["skewness"].append(skew)
        kurt=kurtosis(var, excs)
        data["kurtosis"].append(kurt)
        glcm=GLCM(image)
        contrast=graycoprops(glcm, 'contrast')
        data["contrast"].append(contrast[0][0])
        correlation=graycoprops(glcm, 'correlation')
        data["correlation"].append(correlation[0][0])
        homogeneity2=graycoprops(glcm, 'homogeneity')
        data["homogeneity2"].append(homogeneity2[0][0])
        energy=graycoprops(glcm, 'energy')
        data["energy"].append(energy[0][0])
        dissimilarity=graycoprops(glcm, 'dissimilarity')
        data["dissimilarity"].append(dissimilarity[0][0])
        ASM=graycoprops(glcm, 'ASM')
        data['ASM'].append(ASM[0][0])
    return data

data_plane=lonz_nematode(direct_plane, data1)
plane=pd.DataFrame(data_plane)

data_car=lonz_nematode(direct_car, data2)
car=pd.DataFrame(data_car)
data_train=lonz_nematode(direct_train, data3)
train=pd.DataFrame(data_train)
variables=list(plane.columns)

# Variable normalization  
def normalize_function(data):
    data_normalized=data.copy()
    for var in list(data_normalized.columns):
        var_min=data_normalized[var].min()
        var_max=data_normalized[var].max()
        data_normalized[var]=(data_normalized[var]-var_min)/var_max
    return data_normalized

plane_plane=normalize_function(plane)
car_car=normalize_function(car)
train_train=normalize_function(train)

plane_zeros=[]
for i in range(plane_plane.shape[0]):
    plane_zeros.append(0)
plane_plane["car_plane_train"]=plane_zeros

cars_ones=[]
for i in range(car_car.shape[0]):
    cars_ones.append(1)
car_car["car_plane_train"]=cars_ones

train_threes=[]
for i in range(train_train.shape[0]):
    train_threes.append(2)
train_train["car_plane_train"]=train_threes

car_plane_train_data=pd.concat([plane_plane, car_car, train_train], axis=0).sample(frac=1).reset_index(drop=True)

#####
sns.scatterplot(x=car_plane_train_data["correlation"], y= car_plane_train_data["entropy"], 
                hue=car_plane_train_data["car_plane_train"])
plt.show()
#
car_plane_train_data.columns

sns.set_style("darkgrid")
plt.figure(figsize=(10,6))
sns.kdeplot(data=car_plane_train_data, x="entropy", 
             hue="car_plane_train", fill=True)
plt.show()
#
from mpl_toolkits.mplot3d import Axes3D
variables=car_plane_train_data.columns
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
#1------symetry+entropy+correlation
#2------entropy + excess+ correlation
%matplotlib widget 
fig = plt.figure()
ax=Axes3D(fig)
my_plot = ax.scatter(car_plane_train_data["correlation"], car_plane_train_data["entropy"], 
                   car_plane_train_data["kurtosis"], c=car_plane_train_data["car_plane_train"])
plt.show()

# Matrice de correlation entre les differentes variables
numeric_col=car_plane_train_data.select_dtypes(include=['number'])
cov_matric=numeric_col.corr()
print(cov_matric)


#                                  import necessary packages and modules for the model developments
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold, learning_curve, cross_val_score, validation_curve
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
#
y=car_plane_train_data["car_plane_train"]
X=car_plane_train_data.drop("car_plane_train", axis=1)
x_train, x_test, y_train, y_test=train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)
#
 #                                       Model 1: KNeighborsClassifier
#
# choice of best parameters 
model1=KNeighborsClassifier()

model1_params={"n_neighbors":np.arange(1, 20),
        "weights":["uniform", "distance"], 
        "algorithm":["ball_tree", "kd_tree"], 
        "leaf_size": np.arange(10, 20), 
        "metric":["euclidean", "manhattan", "minkowski"]}

cv=StratifiedKFold(5)

model1_grid=GridSearchCV(model1, param_grid=model1_params, cv=cv, scoring='accuracy')
model1_grid.fit(x_train, y_train)
model1_grid.score
model1_grid.best_score_
best_model1=model1_grid.best_estimator_
n, train_score, val_score= learning_curve(best_model1, x_train, y_train, cv=cv, train_sizes=np.linspace(0.8, 1.0, 20))
predictions=best_model1.predict(x_train)
best_model1.score(x_test, y_test)

##

print(n)
plt.plot(n, train_score.mean(axis=1), label="train")
plt.plot(n, val_score.mean(axis=1), label="validation", c='red')
plt.legend()
plt.show()

# testing 
predictions1=best_model1.predict(x_train)
confusion_matrix(y_train, predictions1)

predictions11=best_model1.predict(x_test)
confusion_matrix(y_test, predictions11)


#
 #                                       Model 2: RandomForestClassifier
#
# 
model2=RandomForestClassifier()

model2_params={"n_estimators":np.arange(3, 50),
        "criterion":["gini", "entropy", "log_loss"], 
        "min_samples_split":np.arange(2, 50), 
        "max_features":["sqrt", "log2"],
        "class_weight":["balanced", "balanced_subsample"]        
       }

cv2=StratifiedKFold(5)

#
model2_grid=GridSearchCV(model2, param_grid=model2_params, cv=cv2, scoring='accuracy')
model2_grid.fit(x_train, y_train)
model2_grid.score

#
best_model2=model2_grid.best_estimator_
best_model2.score(x_train, y_train)

n2, train_score2, val_score2= learning_curve(best_model2, x_train, y_train, cv=cv, train_sizes=np.linspace(0.8, 1.0, 20))

# Predictions on the training and testing datasets

predictions2=best_model2.predict(x_train)
best_model2.score(x_test, y_test)

predictions_mode2=best_model2.predict(x_test)
confusion_matrix(y_test, predictions_mode2)

print(n2)
plt.plot(n2, train_score2.mean(axis=1), label="train")
plt.plot(n2, val_score2.mean(axis=1), label="validation", c='red')
plt.legend()
plt.show()

#
 #                                       Model 3: svm
#
model3=svm.SVC()

model3_params={"kernel": ["poly", "rbf", "sigmoid"], 
              "degree": np.arange(2, 5), 
              "gamma":["scale"], 
              "cache_size":np.linspace(100, 400, 10)
              }
cv3=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

model3_grid = GridSearchCV(model3, param_grid=model3_params, cv=cv3, scoring='accuracy')
model3_grid.fit(x_train, y_train)

model3_grid.score

best_model3= model3_grid.best_estimator_

best_model3.score(x_train, y_train)

n3, train_score3, val_score3= learning_curve(best_model3, x_train, y_train, cv=cv3, train_sizes=np.linspace(0.2, 1.0, 50))

predictions3=best_model3.predict(x_test)

print(n3)
plt.plot(n3, train_score3.mean(axis=1), label="train")
plt.plot(n3, val_score3.mean(axis=1), label="validation", c='red')
plt.legend()
plt.show()

# confusion matrix
confusion_matrix(y_train, best_model3.predict(x_train))
confusion_matrix(y_test, predictions3)

#
 # model4: xgboost
#

model4_params = {
    'n_estimators':np.arange(5, 15),
    'tree_method': ['approx'],
    'max_depth':[2, 3, 4, 5, 6], 
    'learning_rate': [0.1, 0.2, 0.05, 0.01]
}
num_boost_round = 10
cv=StratifiedKFold(5)
model4 = xgb.XGBClassifier(objective= 'multi:softprob')
model4_grid=GridSearchCV(model4, param_grid=model4_params, cv=cv, scoring='accuracy')

model4_grid.fit(x_train, y_train)

model4_grid.best_params_

best_model4=model4_grid.best_estimator_

n4, train_score4, val_score4= learning_curve(best_model4, x_train, y_train, cv=cv, train_sizes=np.linspace(0.2, 1.0, 50))

print(n4)
plt.plot(n4, train_score4.mean(axis=1), label="train")
plt.plot(n4, val_score4.mean(axis=1), label="validation", c='red')
plt.legend()
plt.show()

predictions4=best_model4.predict(x_test)
confusion_matrix(y_test, predictions4)

