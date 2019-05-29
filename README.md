>      
>                                          __  __                                   
>                                   ___ ___\ \/ /__ _                               
>                                  / _ \_  /\  // _` |                              
>                                 |  __// / /  \ (_| |                              
>                                 \___/___/_/\_\__, |                               
>                                              |___/                                
>                              Easy Xgboost Implementation                          


# Simple xgboost wrapper

**ezXg** is a simple R utility, designed as a package, whose goal is to simplify the calibration with the [xgboost](https://xgboost.readthedocs.io/en/latest/) *R* library and make it directly usable with new data.

It is inspired from Über's [ludwig](https://github.com/uber/ludwig) project, which allows to easily train model with Tensorflow.

It uses 5 functions in order to prepare data, train a model and make predictions.
+ `xg_load_data`: load and clean the data.
+ `xg_train`: train the model.
+ `xg_gs`: grid search for hyperparameter selection.
+ `xg_predict`: prediction using the model.
+ `xg_auto_ml`: auto ML feature for running a model in 1 line.

## Install package

Tha package can be installed from its *Github* repository with the `install_github` function from the `devtools` library.

```splus
devtools::install_github("arnaudbu/ezXg")
```

It relies on the `xgboost`, `data.table` and `yaml` libraries.

## Load data

For example purpose, we will use the famous *Kaggle* titanic dataset, that can be loaded from the library.

```splus
titanic <- system.file("extdata", "titanic.csv", package = "ezXg")
```

The data can easily be loaded with the `xg_load_data` function:

```splus
d <- xg_load_data(titanic,
                  inputs = c("Pclass",
                             "Sex",
                             "Age",
                             "SibSp",
                             "Parch",
                             "Fare",
                             "Embarked"),
                  output = "Survived",
                  train.size = 0.8)
```

## Train model

The model can be trained with the `xg_train` function, that is just a wrapper of the `xgb.train` function.

```splus
md <- xg_train(d)
```

## Grid search

The `xg_gs` function implements a two-step calibration process in order to find the optimal set of hyperparameters for model:

+ a coordinate descent, which just consist of moving, one coordinate at a time and one step at a time, along a grid and in the direction given by the best improvement until the optimization function cannot be improved anymore;
+ a complete search on each parameter for all the possible values specified for this hyperparameter.

```splus
gs <- xg_gs(d)
```

## Make predictions

The prediction function takes the data (as a *data.frame* or *data.table*) as an input in order to make predictions on these new values.

```splus
new_data <- fread(titanic)
p <- xg_predict(md, new_data)
```

## Auto ML feature

The auto ML feature is simply a wrapper that load the data, implements a grid search and train the model.

```splus
conf <- system.file("extdata", "ex_param.yml", package = "ezXg")

md <- xg_auto_ml(titanic, conf)
```

It is fed by the path to the dataset and a yaml configuration file, whose structure is the one represented below.

```yaml
global:
  seed: 1 # Seed for the model
  train.size: 0.8 # Training size for validation
  max.levels: 50 # maximum number of levels for factors
  nthread: 2 # Number of thread for the training
  verbose: true # Should information be printed ?
  retrain.full: true # If set to true, the model is trained with full dataset at the end
model:
  inputs: # Input columns (either list or set to "auto")
    - "Pclass"
    - "Sex"
    - "Age"
    - "SibSp"
    - "Parch"
    - "Fare"
    - "Embarked"
  output: "Survived" # Output column
  inputs.class: "auto" # Input class (either list or set to "auto")
  output.class: "auto" # Output class (either "cat", "num" or "auto")
  na.handle: "mean" # Way to handle NA for numeric values
param:
  eta: # Eta parameter, either one value or list if cv > 1
    - 0.05
    - 0.1
    - 0.15
  gamma: # Gamma parameter, either one value or list if cv > 1
    - 0.0
    - 0.1
    - 0.2
    - 0.3
  max_depth: # Max_depth parameter, either one value or list if cv > 1
    - 5
    - 6
    - 8
  colsample_bytree: # Colsample_bytree parameter, either one value or list if cv > 1
    - 0.8
    - 0.9
    - 1.0
  min_child_weight: # Min_child_weight parameter, either one value or list if cv > 1
    - 1
    - 3
  nrounds: 100 # Number of rounds for the training
  objective: "auto" # Objective function
  cv: 5 # Number of folds for cross validation. If set to 1, there will not be any.
```

The fields are simply the different variables of the functions contained in the library, grouped under three categories:
+ **global** for global parameters;
+ **model** for model related parameters, such as the input and output values;
+ **param** the parameters and hyperparameters for model calibration.

```splus
md <- xg_auto_ml(system.file("extdata", "titanic.csv", package = "ezXg"),
                 system.file("extdata", "ex_param.yml", package = "ezXg"))
```

Most of the fields are optional and have default values.

The only required field is the name of the output value.

The following file structure is thuss correct:

```yaml
model:
  output: "Survived"
```
