# file backend/server/apps/ml/classifier/xgboost.py
import joblib
import pandas as pd

from os import path

class XGBoostClassifier:
    def __init__(self):
        path_to_artifacts = "../../research/models/"
        self.values_fill_missing = 0. # joblib.load(path_to_artifacts + "train_mode.joblib")
        # self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")

        assert path.exists(path.join(path_to_artifacts, "xgboost.joblib"))
        self.model = joblib.load(path.join(path_to_artifacts, "xgboost.joblib"))

    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        # fill missing values
        input_data.fillna(self.values_fill_missing)
        # # convert categoricals
        # for column in [
        #     'FG%', 
        #     '3P%', 
        #     'FT%', 
        #     'REB', 
        #     'AST', 
        #     'STL', 
        #     'BLK', 
        #     'TOV'                        
        # ]:
        #     categorical_convert = self.encoders[column]
        #     input_data[column] = categorical_convert.transform(input_data[column])

        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def postprocessing(self, input_data):
        label = "Wait"
        if input_data[1] > 0.5:
            label = "Buy"

        return {
            "probability": input_data[1], 
            "label": label,
              "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction