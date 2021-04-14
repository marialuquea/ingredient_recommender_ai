# Data Mining and Exploration

## Features

- Make predictions of the type of a cuisine given a recipe
- Make ingredients recommendation given a partial recipe

###  Mini Project by 
- s2122286 - [Tiago Lé](mailto:s2122286@sms.ed.ac.uk)
- s1649104 - [Maria Luque Anguita](mailto:s1649104@ed.ac.uk)
- s2091900 - [Rodrigo Morales Flores](mailto:s2091900@ed.ac.uk)
- s2115429 - [Madison Van Horn](mailto:s2115429@ed.ac.uk)

## Project structure
```
│   LICENSE
│   main.py
│   README.md
│   requirements.txt
│   
├───algorithms
│       naive_bayes.py
│       RandomForest.py
│       SVM.py
│      
├───data
│       cuisine-descriptions.txt
│       Cuisines.csv
│       recipes-mallet.txt
│       recipes.arff
│       recipes.csv
│       split_train_test.py
│       suggestion_testbed.mat
│      
├───models
│       naive_bayes.sav
│       randomForest.sav
│       svm.sav
│       
├───notebooks
│       dme data cleaning.ipynb
│       Exploratory_analysis.ipynb
│       PCA.ipynb
│      
└───results
        memory_based_models.csv
```

## Tech

This project uses a number of open source projects to work properly:

- [Surprise](https://surprise.readthedocs.io/en/stable/) - For the recommendation task
- [Scikit Learn](https://scikit-learn.org/stable/) - For the model generation and evaluation

## Installation

1. Download the project locally `$ git clone https://github.com/marialuquea/dme_miniProject.git`
1. Open a terminal and change directory to the project folder `$ cd dme_miniProject`
1. Create a new environment with `$ conda create -n dmeMiniProject python=3.8`. If you don't have conda installed, follow the insructions [here](https://github.com/uoe-iaml/iaml-labs/blob/master/README.md) on how to set up an environment.
1. Activate environment `$ conda activate dmeMiniProject`
1. Install all required modules `$ pip install -r requirements.txt`
   
### To run the notebooks (Exploratory Data Analysis)
1. `$ pip install jupyter` and then `$ jupyter notebook` and navigate through the folder until you open one of the notebooks in this directory

### To run the program (Cuisine predictor and ingredient recommendation system)
1. Run the main file and pass the required arguments _(they are all optional)_
``` 
$ python main.py [--render RENDER] [--verbose VERBOSE] [--task {cuisine,recommendation}] 
                 [--cuisine_model {random_forest,naive_bayes,svm}] [--load_or_train {load,train}] 
                 [--recommendation_model {baseline,knn_basic,knn_baseline,knn_with_means,knn_with_z_score,svd}] 
                 [--baseline_method {als,sgd}] [--similarity {cosine,msd,pearson}] 
```

Argument options
```
optional arguments:
  -h, --help                      show this help message and exit
  --render RENDER                 text to render at the start of the project
  --verbose VERBOSE               verbosity of the process
  --task {cuisine,recommendation} choose between predicting cuisine of recipe or using the recommendation system
  --cuisine_model {random_forest,naive_bayes,svm} 
                                  model to use on the cuisine prediction
  --load_or_train {load,train}    load a pre-trained model or train one
  --recommendation_model {baseline,knn_basic,knn_baseline,knn_with_means,knn_with_z_score,svd}                    
                                  memory model to use on the recommendation task
  --baseline_method {als,sgd}     method used by the baseline model (recommendation task)
  --similarity {cosine,msd,pearson} similarity metric used by the KNN models (recommendation task)
```

## License

MIT

**Free Software, Hell Yeah!**
