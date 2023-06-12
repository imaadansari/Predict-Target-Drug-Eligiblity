# Predict-Target-Drug-Eligiblity
Analyzed over 3,200,000 records and over 27,000 unique Patient-Uids to develop a model that predicts that if a patient is eligible for the "TARGET DRUG" 1 month prior.

This artifical neural network (ANN) is made using PyTorch and trained for 50 epochs.

Files data_preparation.ipynb and data_preparation2.ipynb contain the code for preparing the data.

drug_eligiblity.py contains the code for the ANN, training the ANN and saving the model.

Performance of the model:
- Accuracy: 84.5%
- Precision: 74.0%
- Recall: 81.8%
- F1 Score: 77.7%
