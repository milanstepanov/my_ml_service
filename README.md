# my_ml_service

# Run server
python manage.py runserver

# Go to 
http://127.0.0.1:8000/api/v1

# To check prediciton
http://127.0.0.1:8000/api/v1/classifier/predict

Add a dictinary to content field and post.

E.g.

{
    "FG%": -1.589187263346149,
    "3P%": 0.5524842632526439,
    "FT%": -0.13678449545518512,
    "REB": 0.791423725286965,
    "AST": 0.7003852206441243,
    "STL": -0.4198744868571634,
    "BLK": 0.5675070422621474,
    "TOV": 0.44054625427094024
}

Should return:

{
    "probability": 0.13001027703285217,
    "label": "Wait",
    "status": "OK",
    "request_id": 2
}
