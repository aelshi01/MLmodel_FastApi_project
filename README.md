# Income Prediction using Census Data

## Model Details
Intended Use
This model is intended for the prediction of salary classes (above or below a certain threshold, e.g., 50K) based on various demographic and job-related features collected from the U.S. Census data.

## Selected Model
*Model:* RandomForestClassifier
*Training Library:* scikit-learn
Random State: 0

*Training Data*
The training data is sourced from the U.S. Census data, specifically the census.csv file. It contains demographic and job-related features such as workclass, education, marital status, occupation, relationship, race, sex, and native country.

## Evaluation Data
The evaluation data is also sourced from the U.S. Census data, with an 80-20 train-test split performed on the dataset. The test set is used to evaluate the model's performance on unseen data.

## Metrics
The primary metrics used to evaluate the model's performance are weighted precision, recall, and F1 score. The model's performance on these metrics after training and evaluation is as follows:

Precision: 0.7655
Recall: 0.6378
F1 Score: 0.6958


# Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.
