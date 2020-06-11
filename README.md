# Pima indian diabetes Deep Learning

<p align="center">
   <img  src="https://www.ampersandhealth.co.uk/wp-content/uploads/2018/11/Digital-Health-KD-01_WEB-674x450-2.jpg">
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

## TODO

- [x] add unblacend class
- [x]  train/dev/test
- [x] add normalizing
- [x] regularizer
- [x] Linear Regression
- [x] Support vector machine
- [x] Naive bayes
- [x] Streamlit
- [x] Deep Learning
- [x] Graduation Project Report
- [ ] batch norm
- [ ] lr tuner
- [ ] lime
- [ ] production tips for speed ( tf.server, purning, qunatization)
- [ ] deploy diabetes_large data
- [ ] Persian Blog
- [ ] English Blog
- [ ] upload dataset to kaggle
- [ ] transfer learning
- [ ] online learning
- [ ] adding more languages for UI
- [ ] refactor paging capability
- [ ] dockerize it (fandoughe for eg or my vps)
- [ ] flask api
- [ ] flutter app
- [ ] error analysis
- [ ] feature enginearing
- [ ] database for patients
- [ ] test learning curve
- [ ] fill missing data (with a model or other stuff)