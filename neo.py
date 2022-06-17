# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:27:55 2022

@author: doguilmak

Context

There is an infinite number of objects in the outer space. Some of them are 
closer than we think. Even though we might think that a distance of 70,000 Km 
can not potentially harm us, but at an astronomical scale, this is a very small 
distance and can disrupt many natural phenomena. These objects/asteroids can 
thus prove to be harmful. Hence, it is wise to know what is surrounding us and 
what can harm us amongst those. Thus, this dataset compiles the list of NASA 
certified asteroids that are classified as the nearest earth object.

Sources

NASA Open API - https://api.nasa.gov/
NEO Earth Close Approaches - https://cneos.jpl.nasa.gov/ca/
Kaggle - https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects?select=neo.csv

"""
#%%
# 1. Importing Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import time
import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Data Preprocessing

# 2.1. Uploading data
start = time.time()
df = pd.read_csv('neo.csv')
HAZARDOUS=df['hazardous']
print("\n", df.head(10))

# 2.2. Removing unnecessary columns
print('\n', df['orbiting_body'].value_counts(), '\n')
print('\n', df['sentry_object'].value_counts(), '\n')

# As we can see, there is only 1 unique value in orbiting_body and sentry_object.
# Because of that, we are dropping there columns. In addition, we don't need to 
# use id and name parameters.
 
df.drop(['orbiting_body', 'sentry_object', 'id', 'name'], axis = 1, inplace = True)

# 2.3. Looking for anomalies and duplicated datas
print(df.isnull().sum())
print("\n", df.head(10))
print(df.tail(10))
print("\n", df.describe().T)
print("\n{} duplicated.".format(df.duplicated().sum()))
print('\n', df.info())
print('\n', HAZARDOUS.value_counts(), '\n')
print('\n', HAZARDOUS.unique(), '\n')

# 2.4. Plotting  
explode = (0, 0.1)
fig = plt.figure(figsize = (10, 10), facecolor='w')
out_df=pd.DataFrame(df.groupby('hazardous')['hazardous'].count())

patches, texts, autotexts = plt.pie(out_df['hazardous'], autopct='%1.1f%%',
                                    textprops={'color': "w"},
                                    explode=explode,
                                    startangle=90, shadow=True)

for patch in patches:
    patch.set_path_effects({path_effects.Stroke(linewidth=2.5,
                                                foreground='w')})

plt.legend(labels=['False','True'], bbox_to_anchor=(1., .95))
# plt.savefig('gender_pie')
plt.show()

for i in range(3):
  label = df.columns[i+2]
  plt.hist(df[df['hazardous']==1][label], color='blue', label="True", alpha=0.7, density=True, bins=15)
  plt.hist(df[df['hazardous']==0][label], color='red', label="False", alpha=0.7, density=True, bins=15)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()


plt.figure(figsize = (12,6))
sns.heatmap(df.corr(),annot = True)

# 2.5. Label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(df['hazardous'])
df['label'] = label_encoder.transform(df['hazardous'])
df['label'].unique()
print(df.head())

# 2.6. Determination of dependent and independent variables
X = df.drop(["label", "hazardous"], axis = 1)
y = df["label"]

# 2.7. Splitting test and train 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 2.8. Scaling datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) # Apply the trained

#%%
# 3 Artificial Neural Network
"""
# 3.1 Loading Created Model
from keras.models import load_model
model = load_model('model.h5')

# 3.2 Checking the Architecture of the Model
model.summary()
"""

# 3.3. Define neural network parameters

INPUT = X.shape[1]
EPOCHS = 12

# 3.4. Building neural networks with 2 hidden layers (1st(32 neurons) and 2nd(16 neurons)).

model = Sequential()
# Creating first hidden layer:
model.add(Dense(32, activation="relu", input_dim=INPUT))  
# Creating second hidden layer:
model.add(Dense(16, activation="relu"))  
# Creating output layer:
model.add(Dense(1, activation="sigmoid"))

# 3.5. Training
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  
model_history = model.fit(X_train, y_train, epochs=EPOCHS)

# 3.6. Show summary and save model
print(model_history.history.keys())
model.summary()
model.save('model.h5')

# 3.7. Plot loss and accuracy
def plot_accuracy_loss(training_results):
    plt.subplot(2, 1, 1)
    plt.plot(training_results.history['loss'], 'r')
    plt.ylabel('Loss')
    plt.title('Training loss iterations')
    plt.subplot(2, 1, 2)
    plt.plot(training_results.history['accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.show()
    
plot_accuracy_loss(model_history)

# 3.8. Evaluate model
model.evaluate(X_test, y_test)

# 3.8. System success
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  #  Comparing results
print("\nConfusion Matrix:\n", cm)

from sklearn.metrics import accuracy_score
print(f"\nAccuracy score: {accuracy_score(y_test, y_pred)}")

end = time.time()
cal_time = end - start
print("\nBuilding model for NASA - Nearest Earth Objects data set took {} seconds.".format(cal_time))
