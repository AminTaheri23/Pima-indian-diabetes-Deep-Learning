import warnings

import joblib
# from tensorflow import keras
import keras
# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sklearn as sk
import streamlit as st
from PIL import Image
from sklearn.dummy import DummyClassifier


def norm_a_data(data):
    data[0] = (data[0] - 0) / (17 - 0)
    data[1] = (data[1] - 0) / (199 - 0)
    data[2] = (data[2] - 0) / (122 - 0)
    data[3] = (data[3] - 0) / (99 - 0)
    data[4] = (data[4] - 0) / (846 - 0)
    data[5] = (data[5] - 0) / (67 - 0)
    data[6] = (data[6] - 0.078) / (2 - 0.078)
    data[7] = (data[7] - 21) / (81 - 21)
    return data[:]


warnings.simplefilter('ignore')
sns.set(rc={'figure.figsize': (10, 5)})
sns.set_style("darkgrid", {'axes.grid': True})

st.title("Diabetes Prediction with Deep Learning")

image = Image.open('img/pima.jpg')

st.image(image, width=500, use_column_width=True, )

st.sidebar.title("Diabetes Detection")
st.sidebar.success(
    "Dibetes Prediction with The power of **Artificial Intelligence**!")

st.sidebar.title("Navigation")
radio = st.sidebar.radio(label="Pages", options=[
                         "Home", "Technical Report", "About"])
if radio == "Home":
    st.write("""
    
    # HOME
    
    ## What is diabetes

    According to the NIH, "Diabetes is a disease that occurs when your **blood glucose**,
     also called blood sugar, is **too high**. Blood **glucose** is your main source of
      energy and **comes from the food you eat**. **Insulin**, a hormone made from the pancreas,
       **helps glucose** from food get into your cells to be used for energy. Sometimes your 
       body doesn’t make enough or any insulin or doesn’t use insulin well. Glucose then stays 
       in your blood and doesn’t reach your cells.
    Over time, **having too much glucose in your blood** can cause health problems. """)

    st.sidebar.title("Write your Data here")
    patient = []
    patient.append(
        st.sidebar.number_input(
            label="Pregnancies",
            min_value=0,
            max_value=40,
            value=0,
            format="%i"
        ))
    patient.append(
        st.sidebar.number_input(
            label="Glucose",
            min_value=0,
            max_value=400,
            value=0,
            format="%i"
        ))
    patient.append(
        st.sidebar.number_input(
            label="BloodPressure",
            min_value=0,
            max_value=400,
            value=0,
            format="%i"))
    patient.append(
        st.sidebar.number_input(
            label="SkinThickness",
            min_value=0,
            max_value=400,
            value=0,
            format="%i"))
    patient.append(
        st.sidebar.number_input(
            label="Insulin",
            min_value=0,
            max_value=1600,
            value=0,
            format="%i"))
    patient.append(
        st.sidebar.number_input(
            label="BMI",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            format="%f",
            step=1.0))
    patient.append(
        st.sidebar.number_input(
            label="DiabetesPedigreeFunction",
            min_value=0.0,
            max_value=400.0,
            value=1.0,
            format="%f",
            step=1.0))
    patient.append(
        st.sidebar.number_input(
            label="Age",
            min_value=0,
            max_value=150,
            value=0,
            format="%i"))

    st.write(f"""
    ## Your data is 

    **Please enter your data in the side bar and double check your data.**

    |Pregnancies| Glucose |  BloodPressure | SkinThickness | Insulin | BMI | Diabetes Pedigree Function | Age|
    |-----------|---------|----------------|---------------|---------|-----|----------------------------|----|
    |{patient[0]}| {patient[1]}| {patient[2]}| {patient[3]}| {patient[4]}|  {patient[5]} | {patient[6]}|{patient[7]}|
     \n """)
    st.write("\n")
    button = st.button("Predict")

    # @st.cache(allow_output_mutation=True)
    def load_it():
        """load saved models from disk

        Returns:
            a tuple of (model, lr_clf, svm_clf, gnb_clf) -- where model is a keras model, 
            lr_clf is a logistic regression classification scikit learn model
            svm_clf is a support vector machine classification scikit learn model
            gnb_clf is a Guassian naive bayse classification model
        """        
        model = keras.models.load_model("model2")
        model._make_predict_function()
        # session = model.get_session()

        lr_clf_file = 'saved_model/lr_clf.sav'
        lr_clf = joblib.load(lr_clf_file)

        svm_clf_file = 'saved_model/svm_clf.sav'
        svm_clf = joblib.load(svm_clf_file)

        gnb_clf_file = 'saved_model/gnb_clf.sav'
        gnb_clf = joblib.load(gnb_clf_file)

        return model, lr_clf, svm_clf, gnb_clf

    def pred_human_readable(pred, balloons):
        """human readable prediction for with streamlit st.write

        Arguments:
            pred {float} -- prediction probablity of model
            balloons {bool} -- whether to show balloons or not

        Returns:
            pred -- float for voting system
        """        
        if pred >= 0.5:
            st.error("Unfortunately, we are **" +
                     str("{:.2f}".format(pred*100)) + "%** sure that you have diabetes\n")
            return pred
        else:
            st.success(f"Hooray! we are **" + str("{:.2f}".format(
                (1-pred)*100)) + "%** sure that you don't have diabetes")
            if balloons:
                st.balloons()
            return pred

    if button:
        vote = 0
        model, lr_clf, svm_clf, gnb_clf = load_it()
        # model.set_session(session)
        my_data = norm_a_data(patient)
        my_data = np.array(my_data)
        my_data = my_data.reshape(8, 1)

        st.write("**Model 1 Results (Deep Learning):** \n")
        pred = model.predict(my_data.transpose())
        pred = float(pred)
        vote += pred_human_readable(pred, 0)

        st.write("**Model 2 Results (Logistic Regression):** \n")
        pred = lr_clf.predict_proba(my_data.transpose())
        pred = float(pred[0][1])
        vote += pred_human_readable(pred, 0)

        st.write("**Model 3 Results (Support Vector Machine):** \n")
        pred = svm_clf.predict_proba(my_data.transpose())
        pred = float(pred[0][1])
        vote += pred_human_readable(pred, 0)

        st.write("**Model 4 Results (Naive Bayes):** \n")
        pred = gnb_clf.predict_proba(my_data.transpose())
        pred = float(pred[0][1])
        vote += pred_human_readable(pred, 0)

        st.write("""
        ## The final result is: 
        **This result is based on Weighted Averages of 4 above models. If the probability is near 50 or 60 percent,
         this means that the model is unsure of its decision.
         Keep in mind that this result cannot replace a professional doctor's diagnosis. **
         """)
        pred_human_readable(vote/4, 1)

    st.write("""
    ### Data Description

    |Columns|Description|
    |-------|------------|
    |**Pregnancies**|Number of times pregnant|
    |**Glucose**|Plasma glucose concentration for 2 hours in an oral glucose tolerance test|
    |**BloodPressure**|Diastolic blood pressure (mm Hg)|
    |**SkinThickness**|Triceps skin fold thickness (mm)|
    |**Insulin**|2-Hour serum insulin (mu U/ml)|
    |**BMI**|Body mass index (weight in kg/(height in m)^2)|
    |**DiabetesPedigreeFunction**|Diabetes pedigree function|
    |**Age**|Age (years)|
    """)

    st.write("""

    ## More on Diabetes

    Although diabetes has no cure, you can take steps to manage your diabetes and stay healthy.
    Sometimes people call diabetes “a touch of sugar” or “borderline diabetes.” These terms suggest that someone doesn’t really have diabetes or has a less serious case, but every case of diabetes is serious.
    What are the different types of diabetes? The most common types of diabetes are type 1, type 2, and gestational diabetes.

    - Type 1 diabetes: If you have type 1 diabetes, your body does not make insulin. Your immune system attacks and destroys the cells in your pancreas that make insulin. Type 1 diabetes is usually diagnosed in children and young adults, although it can appear at any age. People with type 1 diabetes need to take insulin every day to stay alive.

    - Type 2 diabetes: If you have type 2 diabetes, your body does not make or use insulin well. You can develop type 2 diabetes at any age, even during childhood. However, this type of diabetes occurs most often in middle-aged and older people. Type 2 is the most common type of diabetes.

    ### Gestational diabetes 
    Gestational diabetes develops in some women when they are pregnant. Most of the time, this type of diabetes goes away after the baby is born. However, if you’ve had gestational diabetes, you have a greater chance of developing type 2 diabetes later in life. Sometimes diabetes diagnosed during pregnancy is type 2 diabetes.
    Other types of diabetes Less common types include monogenic diabetes, which is an inherited form of diabetes, and cystic fibrosis-related diabetes.
    """)

elif radio == "Technical Report":
    st.write("""
    
    # Technical Report

    ## Data

    the datasets consist of several medical predictor (independent) variables and one target (dependent)
     variable, Outcome. Independent variables include the number of pregnancies the patient has had,
      their BMI, insulin level, age, and so on. 
      [link of data in kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)""")

    diabetes = pd.read_csv('diabetes.csv')

    # Checkbox for peeking data
    option = 0
    if st.sidebar.checkbox('Peek a data record'):
        option = st.sidebar.number_input(
            label='Which date record do you like to see?', min_value=0, max_value=767, value=0)
        st.write(
            f'*Data record Number: {option}*',
            diabetes.iloc[[int(option)]])
    st.write("""

    ### Columns

    |Columns|Description|
    |-------|------------|
    |Pregnancies|Number of times pregnant|
    |Glucose|Plasma glucose concentration for 2 hours in an oral glucose tolerance test|
    |BloodPressure|Diastolic blood pressure (mm Hg)|
    |SkinThickness|Triceps skin fold thickness (mm)|
    |Insulin|2-Hour serum insulin (mu U/ml)|
    |BMI|Body mass index (weight in kg/(height in m)^2)|
    |DiabetesPedigreeFunction|Diabetes pedigree function|
    |Age|Age (years)|
    |Outcome|Class variable (0 or 1) 268 of 768 are 1, the others are 0|

    ### Context

    This dataset is originally from the National Institute of Diabetes, Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

    ### Inspiration

    Can we build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes?

    ## Who is Pima Indians

    "The Pima (or Akimel O'odham, also spelled Akimel O'Otham, or "River People," formerly known as Pima) is a group of Native Americans living in an area consisting of what is now central and southern Arizona. The majority of the surviving two bands of Akimel O'odham are based on two reservations: the Keli Akimel O'Otham of the Gila River Indian Community (GRIC) and the On'k Akimel O'odham of the Salt River Pima-Maricopa Indian Community (SRPMIC). Wikipedia
    """)

    st.write(""" ## Correlation Matrix """)
    # correleation
    corrMatrix = diabetes.corr()
    sns.heatmap(corrMatrix, annot=True)
    st.write("This is the Correlation Matrix of our data. as we can see there not much correlation between features")
    st.pyplot()

    st.write(f"""
    ## Data Informations:
    """)
    st.dataframe(diabetes.describe().T)

    st.write("""
    ## Model Performance

    To measure the performance of a model, we need several elements :

    **Confusion matrix** : also known as the error matrix, allows visualization of the performance of an algorithm :

    - True Positive (TP): Diabetic, correctly identified as diabetic
    - True Negative (TN): Healthy, correctly identified as healthy
    - False Positive (FP): Healthy, incorrectly identified as diabetic
    - False Negative (FN): Diabetic, incorrectly identified as healthy
    ### Metrics

    - Accuracy : (TP +TN) / (TP + TN + FP +FN)
    - Precision : TP / (TP + FP)
    - Recall : TP / (TP + FN)
    - F1 score : 2 x ((Precision x Recall) / (Precision + Recall))
    - Roc Curve : The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) in various threshold settings.

    **Precision Recall Curve** : shows the tradeoff between precision and recall for different thresholds
    To train and test our algorithm we'll use cross validation K-Fold

    In **K-fold cross-validation**, the original sample is randomly partitioned into K equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining K − 1 subsamples are used as training data. The cross-validation process is then repeated K times, with each of the k subsamples used exactly once as validation data. The k results can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is that all observations are used for both training and validation, and each observation is used for validation exactly once.
        
    """)
    st.write("""### Deep Learning Model Architecture""")
    model_1 = Image.open("img/model1.png")
    model_2 = Image.open("img/model2.png")
    model_3 = Image.open("img/model3.png")
    st.image(model_1, use_column_width=True)
    st.image(model_2, use_column_width=True)
    st.image(model_3, use_column_width=True)

    st.write("""
    ## Deep Learning Metrics 
    **Maximum Accuracy**: 0.7858880758285522 \n
    **Maximum F1 Score**: 0.709932267665863 \n
    **Minimum Binary CrossEntropy Loss**: 0.08910478377791797 \n
    **Maximum Validation Accuracy**: 0.7961165308952332 \n
    **Maximum Validation F1 Score**: 0.7352941036224365 \n
    **Maximum Validation Binary CrossEntropy Loss**: 0.09106832598019572 \n
    **Maximum Test Accuracy**: 0.6969 \n
    **Maximum F1 Score**: 0.6435643564356436 \n
    **Maximum Test Binary CrossEntropy loss**: 0.2129 \n
    """)
    accuracy_epoch = Image.open("img/accuracy-epoch.png")
    f1_epoch = Image.open("img/f1-epoch.png")
    loss_epoch = Image.open("img/loss-epoch.png")
    st.image(accuracy_epoch, caption="accuracy / epoch", use_column_width=True)
    st.image(f1_epoch, caption="f1 / epoch", use_column_width=True)
    st.image(loss_epoch, caption="loss / epoch", use_column_width=True)

    #TODO I should write hyperparamters of ml algorithms. 
    st.write("""
    ## Logistic Regression Metrics
    **Maximum Test Accuracy**: 0.7086614173228346 \n
    **Maximum F1 Score**: 0.6145833333333333  \n
    """)

    st.write("""
    ## Support Vector Machine Metrics
    **Maximum Test Accuracy**: 0.7125984251968503 \n
    **Maximum F1 Score**: 0.6256410256410256  \n
    """)

    st.write("""
    ## Naive Bayes Metrics
    **Maximum Test Accuracy**: 0.7362204724409449 \n
    **Maximum F1 Score**: 0.6171428571428572 \n
    """)


elif radio == "About":
    st.write("""

# About
Diabetes prediction with deep MLP models webapp (StreamLit.io) + 
[ipynb](https://github.com/AminTaheri23/Pima-indian-diabetes-Deep-Learning/blob/master/pima-diabetes-classification-deep-learning.ipynb)

This app is maintained by **Amin Taheri**. You can learn more about me at
[My personal Website](https://amintaheri23.github.io).

## Acknowledgments
This project was developed under the supervision of Dr. Attarzadeh. 

        """)

st.sidebar.title("Contribute")
st.sidebar.info(
    "This an open source project and you are very welcome to **contribute** "
    "to the [source code](https://github.com/AminTaheri23/Pima-indian-diabetes-Deep-Learning)."
)
