from django.test import TestCase
from rest_framework.test import APIClient

class EndpointTests(TestCase):

    def test_predict_view(self):
        client = APIClient()
        input_data =  {
            "FG%": -1.589187263346149,
            "3P%": 0.5524842632526439,
            "FT%": -0.13678449545518512,
            "REB": 0.791423725286965,
            "AST": 0.7003852206441243,
            "STL": -0.4198744868571634,
            "BLK": 0.5675070422621474,
            "TOV": 0.44054625427094024
        }
        classifier_url = "/api/v1/classifier/predict"
        response = client.post(classifier_url, input_data, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["label"], "Wait")
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status" in response.data)