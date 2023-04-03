# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
– Person developed model: Adam Elshimi
– Model date: 31st of March 2023
– Model version: version 1
– Model type: Logistic regression trained on default hyperparameters with max iteration of a 1000.  Using latest scikit-learn 1.2.2
– Data slicing: I also performed data slices on different unique elements of the feature 'workclass' to further investigate biases and overall performance of the model

– License: MIT
– Contact: If you have any concerns or questions please email me on adamelshimi1234@hotmail.co.uk
## Intended Use
This model is intended for anyuone wanting to make prediction on income according to a set of features like age, working class etc
The model focuses on a binary case for income, 50k or below and above 50k.

## Training Data
The data was obtained from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/census+income

**Classes are:**
    Class 1: larger than 50K
    Class 2: 50K or less.

**The following are the features and attribute characteristics**
    age: continuous.
    workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    capital-gain: continuous.
    capital-loss: continuous.
    hours-per-week: continuous.
    native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

Number of instances in the data is 48,842 this was split into training data and test data by using a ration split of 75% training and 25% test data.  To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data
Datasets: Used the same data from UCI: https://archive.ics.uci.edu/ml/datasets/census+income we splitted data into a 25% test data to evaluate our model.

Motivation: The data was extracted by Barry Becker from the 1994 Census database and had a comprehensive features to predict income as well as the number of data.

Preprocessing: To clean the data we had to remove spaces from both rows and columns.  Also, we splitted data into training 75% and testing data 25%.  For the both X in the traing and test data we used a OneHotEncoder and for the labels y we used a LabelBinarizer.

## Metrics
Evaulation metric used were precision, recall and fbeta score (harmonic mean of precision and recall).

    **Prediction on all the data**
    ('precision: 0.7082152974504249', 'Recall: 0.25342118601115055', 'fbeta: 0.37327360955580446')

    **Data slicing on the workclass attribute**
    model_State-gov : ('precision: 0.9230769230769231', 'Recall: 0.3037974683544304', 'fbeta: 0.45714285714285713')
    model_Self-emp-not-inc : ('precision: 0.7532467532467533', 'Recall: 0.3314285714285714', 'fbeta: 0.4603174603174603')
    model_Private : ('precision: 0.6951501154734411', 'Recall: 0.2408', 'fbeta: 0.35769459298871065')
    model_Federal-gov : ('precision: 0.56', 'Recall: 0.14432989690721648', 'fbeta: 0.22950819672131148')
    model_Local-gov : ('precision: 0.7358490566037735', 'Recall: 0.23353293413173654', 'fbeta: 0.35454545454545455')
    model_? : ('precision: 1.0', 'Recall: 0.10416666666666667', 'fbeta: 0.18867924528301885')
    model_Self-emp-inc : ('precision: 0.8142857142857143', 'Recall: 0.3904109589041096', 'fbeta: 0.5277777777777778')
    model_Without-pay Not valid data, target values contains only one class: 0
    model_Never-worked Not valid data, target values contains only one class: 0

## Ethical Considerations

Mitigations: To make sure class was accounted for in a fair way we done data slicing on all the attribut charateristic of 'workclass' which highlighted that both 'Without-pay' and 'Never-worked' only had 0 as a class label with corresponds to the below 50K salary.  Therefore, these should have smaller weight than the rest of the attributes charateristics since this may skew the data leading us to wrong evaluation and conclusions.

Data: All data are real people but anonamised to safeguard individual identity and provacy.

Risks and harms: Has been considered but yet to find any cases where this might apply.

## Caveats and Recommendations
One recommendation would be to add industry category to be able to further assess implications to economic factors and education, and whether things like education, ethinicity or class has any relation to what industry you work.  This could bring light to hidden disparities amongst a population, for example whether a one group has more access to education, or more availability in opportunity in a given sector etc.