from django.test import TestCase

import inspect
from apps.ml.registry import MLRegistry

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

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "classifier"
        algorithm_object = XGBoostClassifier()
        algorithm_name = "xgboost"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Milan"
        algorithm_description = "XGBoostClassifier with simple pre- and post-processing"
        algorithm_code = inspect.getsource(XGBoostClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)