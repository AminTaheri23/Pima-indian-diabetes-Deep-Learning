# Pima indian diabetes Deep Learning
<p align="center">
   <img  width="460" height="300" src="https://www.ampersandhealth.co.uk/wp-content/uploads/2018/11/Digital-Health-KD-01_WEB-674x450-2.jpg">
</p>

Diabetes prediction with deep MLP models webapp (StreamLit.io) + [ipynb](https://github.com/AminTaheri23/Pima-indian-diabetes-Deep-Learning/blob/master/pima-diabetes-classification-deep-learning.ipynb)

## Data
the datasets consist of several medical predictor (independent) variables and one target (dependent) variable, Outcome. Independent variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on. [link of data in kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

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

### Acknowledgements
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

### Inspiration
Can you build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes?

## Who is Pima Indians ?

"The Pima (or Akimel O'odham, also spelled Akimel O'Otham, or "River People," formerly known as Pima) is a group of Native Americans living in an area consisting of what is now central and southern Arizona. The majority of the surviving two bands of Akimel O'odham are based on two reservations: the Keli Akimel O'Otham of the Gila River Indian Community (GRIC) and the On'k Akimel O'odham of the Salt River Pima-Maricopa Indian Community (SRPMIC). Wikipedia

## What is diabetes ?
According to the NIH, "Diabetes is a disease that occurs when your **blood glucose**, also called blood sugar, is **too high**. Blood **glucose** is your main source of energy and **comes from the food you eat**. **Insulin**, a hormone made from the pancreas, **helps glucose** from food get into your cells to be used for energy. Sometimes your body doesn’t make enough or any insulin or doesn’t use insulin well. Glucose then stays in your blood and doesn’t reach your cells.
Over time, **having too much glucose in your blood** can cause health problems. Although diabetes has no cure, you can take steps to manage your diabetes and stay healthy.
Sometimes people call diabetes “a touch of sugar” or “borderline diabetes.” These terms suggest that someone doesn’t really have diabetes or has a less serious case, but every case of diabetes is serious.
What are the different types of diabetes? The most common types of diabetes are type 1, type 2, and gestational diabetes.

 - Type 1 diabetes: If you have type 1 diabetes, your body does not make insulin. Your immune system attacks and destroys the cells in your pancreas that make insulin. Type 1 diabetes is usually diagnosed in children and young adults, although it can appear at any age. People with type 1 diabetes need to take insulin every day to stay alive.

 - Type 2 diabetes: If you have type 2 diabetes, your body does not make or use insulin well. You can develop type 2 diabetes at any age, even during childhood. However, this type of diabetes occurs most often in middle-aged and older people. Type 2 is the most common type of diabetes.

Gestational diabetes Gestational diabetes develops in some women when they are pregnant. Most of the time, this type of diabetes goes away after the baby is born. However, if you’ve had gestational diabetes, you have a greater chance of developing type 2 diabetes later in life. Sometimes diabetes diagnosed during pregnancy is type 2 diabetes.
Other types of diabetes Less common types include monogenic diabetes, which is an inherited form of diabetes, and cystic fibrosis-related diabetes ."

## Model Performance
To measure the performance of a model, we need several elements :


**Confusion matrix** : also known as the error matrix, allows visualization of the performance of an algorithm :

- True Positive (TP): Diabetic, correctly identified as diabetic
- True Negative (TN): Healthy, correctly identified as healthy
- False Positive (FP): Healthy, incorrectly identified as diabetic
- False Negative (FN): Diabetic, incorrectly identified as healthy


### Metrics :

- Accuracy : (TP +TN) / (TP + TN + FP +FN)
- Precision : TP / (TP + FP)
- Recall : TP / (TP + FN)
- F1 score : 2 x ((Precision x Recall) / (Precision + Recall))
- Roc Curve : The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) in various threshold settings.


Precision Recall Curve : shows the tradeoff between precision and recall for different thresholds
To train and test our algorithm we'll use cross validation K-Fold


In K-fold cross-validation, the original sample is randomly partitioned into K equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining K − 1 subsamples are used as training data. The cross-validation process is then repeated K times, with each of the k subsamples used exactly once as validation data. The k results can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is that all observations are used for both training and validation, and each observation is used for validation exactly once.
