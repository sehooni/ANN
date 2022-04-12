# ----------------------- Artificial Neural Network for classification ----------------------- #
# Workplace environment
# - tensorflow-gpu == 2.4.0
# - cuda 11.1
# - cudnn 11.2

# importing required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# ----------------------- Data Pre-processing ----------------------- #
# Checking the tensorflow version
print(tf.__version__)

# Loading the data
bank_data = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")

# Taking all rows and all columns in the data except the last column as X(feature matrix)
# the raw numbers and customer id's are not necessary for the modeling so we get rid of and start with credit score
X = bank_data.iloc[:,3:-1].values
    ## .iloc: one of slicing method, that means all the rows and the columns from 4th to end.
    ## .values: 데이터 프레임의 셀에서 값을 가져온다.
print("Independent variables are: ", X)
# taking all rows but only the last column as y(dependent variable)
y = bank_data.iloc[:, -1].values
print("Dependent variable is:", y)


# Transforming the gender variable, labels are chosen randomly
le = LabelEncoder()
    ## LabelEncoder
    ## : 카테고리형 데이터(Categorical Data)를 수치형 데이터(Numerical Data)로 변환해주는 작업 (전처리 작업)
X[:,2] = le.fit_transform(X[:,2])
    ## fit_transform
    ## : train dataset에서만 사용
    ##   우리가 만든 모델은 train data에 있는 mean과 variance를 학습
    ##   이렇게 학습된 Scaler()의 parameter는 test data를 scale하는데 사용
    ##   다시 말해 train data로 학습된 Scalar()의 parameter를 통해 test data의 feature 값들이 스케일 되는 것
print(X)

# Transforming the geography column variable, labels are chosen randomly, the ct asks for argument[1] the index of the target vb
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])], remainder = 'passthrough')
    ## ColumnTransformer: array로 작동하며, 인스턴스에 변수들을 저장하기 때문에 새로운 데이터셋을 마주해도 똑같은 차원으로 가공을 할 수 있다.
    ##                    transformer라는 변수 안에, 이름/ 작업할 함수/ 선택할 컬럼 이렇게 3개를 연달아 집어넣는다.
    ##                    transformer 변수 자체가 list of tuples의 형태로 입력받는다는 것이다.
    ##     remainer     : 작업할 때 걸리는 시간을 알려준다.
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    ## train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify):
    ##  - arrays : 분할시킬 데이터를 입력 (Python list, Numpy array, Pandas dataframe 등..)
    ##  - test_size: test set을 20%의 비율로 추출하여 분할 (default = 0.25)
    ##  - train_size: 학습 데이터셋의 비율(float)이나 갯수(int) (default = test_size의 나머지)
    ##  - random_state: 데이터 분할시 셔플이 이루어지는데 이를 위한 seed 값 지정 (int나 RandomState로 입력)
    ##  - shuffle: 셔플여부설정 (default = True)
    ##  - stratify : 지정한 Data의 비율을 유지
# printing the dimensions of each of those snapshots to see amount of rows and columns i each of them
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Data Scaling/normalization of the features that will fo to the NN
sc = StandardScaler()
    ## StandardScaler:
    ## - 평균 = 0 / 표준편차(분산) = 1
    ## - 표준화 standardization
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(len(X_train))
print(len(X_test))


# ----------------------- Building the model ----------------------- #

# Initializing the ANN by calling the Sequential class from keras of Tensorflow
ann = tf.keras.models.Sequential()

# ---------------------------------------------------------------------------------
# Adding "fully connected" INPUT layer to the Sequential ANN by calling Dense class
# ---------------------------------------------------------------------------------
# Number of Units = 6 and Activation Function = Rectifier
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# ---------------------------------------------------------------------------------
# Adding "fully connected" SECOND layer to the Sequential ANN by calling Dense class
# ---------------------------------------------------------------------------------
# Number of Units = 6 and Activation Function = Rectifier
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# ---------------------------------------------------------------------------------
# Adding "fully connected" OUTPUT layer to the Sequential ANN by calling Dense class
# ---------------------------------------------------------------------------------
# Number of Units = 1 and Activation Function = Sigmoid
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# ----------------------- Training Model ----------------------- #
# Compiling the ANN
# Type of Optimizer = Adam Optimizer, Loss Function = crossentropy for binary variable, and Optimization is done w.r.t. accuracy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN model on training set (fit method always the same)
# batch_size = 32, the default value, number fos epochs = 100
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


# ----------------------- Evaluating the Model ----------------------- #
# the goal is to use this ANN model to predict the probability of the customer leaving the bank
# Predicting the churn probability(가입 해지율) for single observation

# Geography: French 1
# Credit Score: 600
# Gender: Male 1
# Age: 40 years old![](C:/Users/Sehooni/AppData/Local/Temp/googlelogo_color_92x30dp.png)
# Tenure: 3 years
# Balance: $60,000
# Number of Products: 2
# with Credit Card
# Active member
# Estimated Salary: $50,000

# print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
# print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
# this customer has 2% chance to leave the bank


# show the vector of predictions and real values
# probabilities
y_pred_prob = ann.predict(X_test)

# probabilities to binary
y_pred = (y_pred_prob > 0.5)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
    ## concatenate: 선택한 축(axis)의 방향으로 배열을 연결해주는 메소드
    ## concatenate((배열, 배열), 축)
    ## 이때 axis = 1이면, 열방향(좌 → 우)을 의미
# print('y_test:', y_test)
print('y_pred:', y_pred)
# # Confusion Matrix
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix", confusion_matrix)
# print("Accuracy Score", accuracy_score(y_test, y_pred))