# Disaster-Response-Pipelines

# Motivation
This project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages.
The data set contains real messages that were sent during disaster events that wil be used to create a machine learning pipeline 
to categorize these events.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app also displays visualizations of the data.
 
 # Required libraries
 
 - pandas 
 - scikit-learn
 - plotly
 - nltk
 - flask
 - sqlalchemy
 
 # Python Scripts 
- `process_data.py `
- `train_classifier.py`

These Python scripts should be able to run with additional arguments specifying the files used for the data and model.

      python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
      python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Running the Web App

      python run.py

 # Flask App
  The file structure of the project
 
         - app
         | - template
         | |- master.html  # main page of web app
         | |- go.html  # classification result page of web app
         |- run.py  # Flask file that runs app

         - data
         |- disaster_categories.csv  # data to process 
         |- disaster_messages.csv  # data to process
         |- process_data.py
         |- InsertDatabaseName.db   # database to save clean data to

         - models
         |- train_classifier.py
         |- classifier.pkl  # saved model 

         - README.md
# Screenshots 

![Visulas](https://raw.githubusercontent.com/mahajye90/Disaster-Response-Pipelines/main/FireShot%20Capture%20003%20-%20Disasters%20-%20view6914b2f4-3001.udacity-student-workspaces.com.png)

![e.g](https://raw.githubusercontent.com/mahajye90/Disaster-Response-Pipelines/main/FireShot%20Capture%20009%20-%20Disasters%20-%20view6914b2f4-3001.udacity-student-workspaces.com.png)

# Acknowledgements
This project was prepared as part of the Udacity Data Scientist nanodegree programme. The data was provided by Figure Eight
