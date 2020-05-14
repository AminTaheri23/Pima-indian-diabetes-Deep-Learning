import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf 
import matplotlib.pyplot as plt
import sklearn as sk
from tensorflow import keras
import tensorflow.keras.backend as K
import seaborn as sns
import warnings
from sklearn.dummy import DummyClassifier

# TODO
# - ⬜ add unblacend class
# - ⬜ batch norm
# - ⬜ regularizer
# - ⬜ train/dev/test
# - ⬜ lr tuner
# - ✅ add normalizing 

# Button adding 
# button = st.button("Add a and b")
#     if button:
#         st.write(f"a+b=")

warnings.simplefilter('ignore')
sns.set(rc={'figure.figsize' : (10, 5)})
sns.set_style("darkgrid", {'axes.grid' : True})

st.title("Diabetes Prediction with Deep Learning")
st.write("""
<p dir="rtl">Write this text right-to-left!</p>

## تست برای فارسی
""")

diabetes = pd.read_csv('diabetes.csv')

##################Checkbox for peeking data 
option=0
if st.checkbox('peek a particular data'): 
    option = st.text_input(
    'Which number do you like best?', 1)
    st.write(
    '*below is 5 examples of dataframe*',
    diabetes.iloc[[int(option)]])

############### HEAD 
num_of_data_to_show = st.slider('Number of data point to display', 0, 20, 5)
st.write(f"Showing {num_of_data_to_show} data points")
st.write(diabetes.head(int(num_of_data_to_show)))

st.write("Now lets describe the dataframe", diabetes.describe())

########## correleation
st.subheader("correleation matirx")
corrMatrix = diabetes.corr()
sns.heatmap(corrMatrix, annot=True)
st.write("This is a Correlation Matrix")
st.pyplot()

######### data
x = diabetes.drop(columns = 'Outcome')
y = diabetes['Outcome']

################ dummy classifier
dummy = DummyClassifier('most_frequent') #returining most frequent class in this case 1/
results = dummy.fit(x,y)
st.write("A dummy classifier score", results.score(x,y))

number = st.number_input('Insert a number')
st.write('The current number is ', number)

# import time
# my_bar = st.progress(0)
# for percent_complete in range(10):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete*10 + 1)


st.error("test")

st.warning('This is a warning')
st.info('This is a purely informational message')
st.success('This is a success message!')

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def max_metric (history):
    max_acc = max(history.history['accuracy'])
    max_f1 = max(history.history['get_f1'])
    min_loss = min(history.history['loss'])
    max_val_acc = max(history.history['val_accuracy'])
    max_val_f1 = max(history.history['val_get_f1'])
    min_val_loss = min(history.history['val_loss'])
    st.write(
        f"""Maximum Accuracy: {max_acc} \n
        Maximum F1 Score: {max_f1} \n
        Minimum Binary CrossEntropy Loss: {min_loss} \n
        Maximum Validation Accuracy: {max_val_acc} \n
        Maximum Validation F1 Score: {max_val_f1} \n
        Maximum Validation Binary CrossEntropy Loss: {min_val_loss} \n"""
        )


def plot_this(history):
    # summarize history for accuracy
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()
    
    # summarize history for f1
    plt.plot(history.history['val_get_f1'])
    plt.plot(history.history['get_f1'])
    plt.title('model f1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()
    
    # summarize history for loss
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, input_dim=x.shape[1], activation = 'relu' ))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))




model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy', get_f1]
    )

# st.write(model)

model.summary(print_fn = st.write)
#history = model.fit(x,y, validation_split=0.33, batch_size=128, epochs=200, workers=4, verbose=3)

# max_metric(history)
# plot_this(history)


# ## with Normalization



diabetes.columns




# normalize the data
# we do not want to modify our label column Exited
cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
       'DiabetesPedigreeFunction', 'Age']

# copy churn dataframe to churn_norm to do not affect the original data
dia_norm = diabetes.copy()

# normalize churn_norm dataframe 
dia_norm[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/ (x.max() - x.min()) )



dia_norm



dia_norm.describe()



model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(16, input_dim=x.shape[1], activation = 'relu' ))
model2.add(tf.keras.layers.Dense(16, activation='relu'))
model2.add(tf.keras.layers.Dense(16, activation='relu'))
model2.add(tf.keras.layers.Dense(16, activation='relu'))
model2.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model2.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy', get_f1])

x = dia_norm.drop(columns = 'Outcome')
y = dia_norm['Outcome']



# history2 = model.fit(x,y, validation_split=0.33, batch_size=128, workers=4, epochs=200, verbose=3)

# max_metric(history2)
# plot_this(history2)



# print("without normalization")
# max_metric(history)

# print("#################################################")

# print("\nwith normalization")
# max_metric(history2)


