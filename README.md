# EHR_LLM_BERTAEF

## Description:
This project leverages the power of BERT for classifying patient data, predicting whether a patient is "in care" or "out of care" based on key hematological and demographic features. The model is trained on medical data, offering an efficient solution for healthcare predictive analytics.

## Requirements:
Make sure to install the required libraries by running the following command:
```
pip install -r requirements.txt
```

## How to Use:
1. **Install the required libraries**:
   To install all the dependencies, run:
   ```
   pip install -r requirements.txt
   ```

2. **Prepare your data**:
   Ensure you have the `data-ori.csv` file in the same directory as the script. This file should contain the necessary patient data with columns such as `HAEMATOCRIT`, `HAEMOGLOBINS`, `AGE`, `SEX`, etc.

3. **Running the code**:
   After setting up the environment and preparing the data, you can run the Python script to start training and testing the model:
   ```
   python ehr_llm_bertaef.py
   ```

   The script will:
   - Load the data and preprocess it.
   - Split the data into training and test sets.
   - Fine-tune the BERT model.
   - Save the trained model in the `./results` directory.

4. **Model Predictions**:
   After training, you can use the saved model to make predictions for new patient data. The model will classify whether the patient is "in care" or "out of care."


## Model Details:
The model is based on the `bert-base-uncased` pre-trained BERT model. It is fine-tuned on the given medical data to predict the patient's status (in or out of care). The model is saved in the `./results` directory after training.

## Model Download

The trained model (`EHR-BERT-MODEL.zip`) can be downloaded from the following link:

- [Download EHR-BERT-MODEL](https://drive.google.com/file/d/1X_81e5E0LtAU-s5ct3tucNFzVvYuv5Tq/view?usp=sharing)

Alternatively, you can train the model yourself by running the provided code.


## Contributing:
If you'd like to contribute to this project, feel free to fork it and create a Pull Request with your suggestions, improvements, or bug fixes.
