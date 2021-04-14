# Data Mining and Exploration

Mini Project by 
- s2122286 - Tiago Lé
- s1649104 - Maria Luque Anguita
- s2091900 - Rodrigo Morales Flores
- s2115429 - Madison Van Horn

## Features

- Make predictions of the type of a cuisine given a recipe
- Make ingredients recommendation given a partial recipe

## Project structure
```
│   KNN_basic.py
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
│   │   dme data cleaning.ipynb
│   │   Exploratory_analysis.ipynb
│   │   PCA_KMeans.ipynb
│   │   SVM.ipynb
│   │   
│   └───.ipynb_checkpoints
│           Exploratory_analysis-checkpoint.ipynb
│           
├───project_management
│       dme_gantt_chart.xlsx
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
1. Install all required modules `$ pip install -r requirements.txt`
1. Run the main file and pass the required arguments 
``` 
$ python main.py [--render RENDER] [--verbose VERBOSE] [--task {cuisine,recommendation}] [--cuisine_model {random_forest,naive_bayes,svm}] [--load_or_train {load,train}] [--recommendation_model {baseline,knn_basic,knn_baseline,knn_with_means,knn_with_z_score,svd,svdpp,nmf}] [--baseline_method {als,sgd}] [--similarity {cosine,msd,pearson}] 
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
  --recommendation_model {baseline,knn_basic,knn_baseline,knn_with_means,knn_with_z_score,svd,svdpp,nmf}                    memory model to use on the recommendation task
  --baseline_method {als,sgd}     method used by the baseline model (recommendation task)
  --similarity {cosine,msd,pearson} similarity metric used by the KNN models (recommendation task)
```

## License

MIT

**Free Software, Hell Yeah!**
