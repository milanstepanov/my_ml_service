from django.test import TestCase

from apps.ml.classifier.xgboost import XGBoostClassifier

class MLTests(TestCase):

    def test_xgboost_algorithm(self):
        input_data = {
            "FG%": -1.589187263346149,
            "3P%": 0.5524842632526439,
            "FT%": -0.13678449545518512,
            "REB": 0.791423725286965,
            "AST": 0.7003852206441243,
            "STL": -0.4198744868571634,
            "BLK": 0.5675070422621474,
            "TOV": 0.44054625427094024
        }
        my_alg = XGBoostClassifier()

        response = my_alg.compute_prediction(input_data)

        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('Wait', response['label'])
