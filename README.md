# Income Prediction using Census Data

## Model Details
Predict whether income exceeds $50K/yr based on census data, which is broken down into 14 attributes:


### Selected Model
I selected a Logistic Regression classify trained with:
* max_iter = 1000
* random_state = 0
* Hyperparameters set to defualts

I trained the model with a test_size=0.25 and  a random_state=99, the details of which can be found in the Training and evaluation section.

## Data
Data was sourced from the following [link][UCI Machine Learning].  Data comprised of 15 columns, the first 14 columns are the attributes used to predict the 15th column - salary.  This is a binary classification problem where the salary can only be two values less than or greater than 50K.  

Motivation: The data was extracted by Barry Becker from the 1994 Census database and had a comprehensive features to predict income as well as the number of data.

Preprocessing: To clean the data we had to remove spaces from both rows and columns.  Also, we splitted data into training 75% and testing data 25%.  For the both X in the traing and test data we used a OneHotEncoder and for the labels y we used a LabelBinarizer.

*Attributes and classes*

1. age: continuous.
2. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
3. fnlwgt: continuous.
4. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
5. education-num: continuous.
6. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
7. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
8. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
9. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
10. sex: Female, Male.
11. capital-gain: continuous.
12. capital-loss: continuous.
13. hours-per-week: continuous.
14. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

*Training Data*

The training data is from the U.S. Census data, which can be fspecifically the census.csv file. It contains demographic and job-related features such as workclass, education, marital status, occupation, relationship, race, sex, and native country.
15. salary: >50K, <=50K.

*Evaluation Data*

The evaluation data is also sourced from the U.S. Census data, which can be found above by following the link to the repo.  The evaluation was split into 25% of the original dataset using train_test_split from the sklearn model_selection module.   The evaluation data is unseen data to the model and it is used to evaluated our model performance by finding out the how well it can do on new unseen data.

## Metrics
The metrics used to see the models performance is the following 
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

# Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.


[UCI Machine Learning]: https://archive.ics.uci.edu/ml/datasets/census+income
