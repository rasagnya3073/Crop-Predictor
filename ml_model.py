import numpy as np
import pandas as pd

crop=pd.read_csv("Crop_recommendation.csv")

crop_dict={
    'rice'       :    1,
    'maize'      :    2,
    'jute'       :    3,
    'cotton'     :    4,
    'coconut'    :    5,
    'papaya'     :    6,
    'orange'     :    7,
    'apple'      :    8,
    'muskmelon'  :    9,
    'watermelon' :    10,
    'grapes'     :    11,
    'mango'      :    12,
    'banana'     :    13,
    'pomegranate':    14,
    'lentil'     :    15,
    'blackgram'  :    16,
    'mungbean'   :    17,
    'mothbeans'  :    18,
    'pigeonpeas' :    19,
    'kidneybeans':    20,
    'chickpea'   :    21,
    'coffee'     :    22
}
crop['crop_num']=crop['label'].map(crop_dict)

crop.drop('label',axis=1,inplace=True)

x=crop.drop('crop_num',axis=1)
y=crop['crop_num']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import MinMaxScaler
ms=MinMaxScaler()
ms.fit_transform(x_train)
ms.fit_transform(x_test)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit_transform(x_train)
ss.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
accuracy_score(y_test,y_pred)

crop_dict2={
    1:'rice',
    2:'maize',
    3:'jute',
    4:'cotton',
    5:'coconut',
    6:'papaya',
    7:'orange',
    8:'apple',
    9:'muskmelon',
    10:'watermelon',
    11:'grapes',
    12:'mango',
    13:'banana',
    14:'pomegranate',
    15:'lentil',
    16:'blackgram',
    17:'mungbean',
    18:'mothbeans',
    19:'pigeonpeas',
    20:'kidneybeans',
    21:'chickpea',
    22:'coffee'
}



def predict(N,P,K,temperature,humidity,ph,rainfall):
    features=np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    prediction=rfc.predict(features).reshape(1,-1)
    predict = prediction[0]
    if predict[0] in crop_dict2:
        crop=crop_dict2[predict[0]]
    return crop

N=40
P=50
K=50
temperature=40.0
humidity=20
ph=100
rainfall=100
res=predict(N,P,K,temperature,humidity,ph,rainfall)

