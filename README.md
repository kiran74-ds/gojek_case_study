# gojek_case_study

## Project Instructions:

### This project can be run locally by following the steps:

+ Clone the repository 

```
git clone https://github.com/kiran74-ds/gojek_case_study.git
```

+ Create Virtual Environmnet
```
python3 -m venv venv
```
+ Activate Virtual Environment

```
source venv/bin/activate
```
+ Install necessary libraries using pip

```
pip install -r requirements.txt 
```

+ Finally train the model and get predictions
```
python3  train.py
```

### Running the project using docker  

+ Clone the repository 

+ Build Docker Image
```
docker build . -t ml_project
```
+ Run Docker Image
```
docker run  ml_project
```
```
