# Mining Engineering
Machine Learning using data from mining industry

# Dockerized

1. Build Docker Image:
```
$ docker build -t mining-engineering .
```
2. Run Docker Container:
```
$ docker run -e PORT=8501 mining-engineering
```

# Python

1. Create Virtual Environment (Windows):
```
$ virtualenv venv
$ .\venv\Scripts\activate
$ pip install -r requirements.txt
```
1. Create Virtual Environment (Linux):
```
$ virtualenv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
```
2. Run App (Windows/Liux):
```
$ streamlit run app.py
```
