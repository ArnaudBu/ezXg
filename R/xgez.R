###############################################################################
#                                    __  __                                   #
#                             ___ ___\ \/ /__ _                               #
#                            / _ \_  /\  // _` |                              #
#                           |  __// / /  \ (_| |                              #
#                           \___/___/_/\_\__, |                               #
#                                        |___/                                #
#                        Easy Xgboost Implementation                          #
###############################################################################

# Load Data -----

#' Load Data
#'
#' \code{load_data} returns a list with all the prepared element for loading
#' and preparing the data for xgboost modelling. The model principally relies
#' on two categories of input: numeric (\emph{num}) and category (\emph{cat}).
#'
#' @param file \strong{Character}. The link to the file containing the data.
#'  The data are imported with the \code{fread} function from
#'  the \code{data.table} package (\link[data.table]{fread}), so the format
#'  must be consistent with a csv file.
#' @param inputs \strong{Character vector}. Vector of the column names for
#' the inputs of the model. Only those columns will be used for the model.
#' Using the "auto" value will use as inputs all the columns from the
#' table except the one labelled as output.
#' @param output \strong{Character}. A single string specifying the name
#' of the output column for the model training.
#' @param inputs.class \strong{Character vector}. A vector specifying the
#' classes for the input column. If set to "auto", the classes will be
#' determined from the output of the \link[data.table]{fread} function.
#' Else, it must me a vector whose size is exactly the number of input and
#' whose values can only be \emph{num} (for numerical inputs) and \emph{cat}
#'  (for categorical inputs).
#' @param output.class \strong{Character}. Class for output. If set to "auto",
#' the class will be determide from the output of the \link[data.table]{fread}
#' function. Else, it must be equal to \emph{num} or \emph{cat} for numerical
#' or categorical inputs.
#' @param train.size \strong{Numeric}. Size for training set for the future
#' model. Can go from 0 (no training set: will produce an error) to 1 (no test
#' set).
#' @param seed \strong{Numeric}. Seed for reproducibility of the results.
#' @param na.handle \strong{Character}. Way to handle na value in numeric
#' inputs. Five possibilities have been implemented:
#' \itemize{
#'  \item{\emph{inf}}: replace missing values with \code{Inf}.
#'  \item{\emph{mean}}: replace missing values with the mean of the column.
#'  \item{\emph{median}}: replace missing values with the median of the column.
#'  \item{\emph{max}}: replace missing values with the max of the column.
#'  \item{\emph{min}}: replace missing values with the min of the column.
#' }
#' @param max.levels \strong{Numeric}. Maximum number of levels admitted for
#' a category. This parameters is here to make sure that the model does not
#' have to many input data when transformed into a one-hot encoded matrix.
#'
#' @return A list with following values:
#' \itemize{
#'  \item{\strong{train}}: training set for the model, with a matrix for the
#'  input values and a vector for the target variables.
#'  \item{\strong{test}}: test set for the model, on the same format that the
#'  training set
#'  \item{\strong{formula}}: the formula used for constructing the model matrix
#'  and that is applied when running the model.
#'  \item{\strong{template}}: an empty \code{data.table} that has saved all the
#'  input values and that is used to appropriately format data when using
#'  the prediction function.
#'  \item{\strong{data}}: A data.table with the cleaned data and an additional
#'  logical column, \emph{train}, that indicates which data are used in the
#'  training data set.
#' }
#'
#' @import data.table
#'
#' @examples
#' temp.file <- tempfile()
#' data(iris)
#' fwrite(iris, temp.file, row.names = FALSE)
#' d <- load_data(temp.file, output = "Species")
#'
#' @export
load_data <- function(file,
                      inputs = "auto",
                      output,
                      inputs.class = "auto",
                      output.class = "auto",
                      train.size = 1,
                      seed = 1,
                      na.handle = "inf",
                      max.levels = 50){
  # Load the data
  d <- data.table::fread( file, stringsAsFactors = T)

  # Compute the input names in case their value is "auto"
  if (inputs[1] == "auto"){
    inputs <- colnames(d)[colnames(d) != output]
  }

  # Validation for inputs
  # Check that na.handle is in the accepted values
  if (!(na.handle %in% c("inf", "min", "max", "mean", "median"))){
    stop(na.handle,
         " is an invalid value for na.handle. Please check documentation
         for authorized values.")
  }
  # Check that the inputs classes are consistent
  if (inputs.class[1] != "auto"){
    if (length(inputs.class) != length(inputs)){
      stop("Unconsitent number of inputs and classes for inputs. The values
           should be the same")
    }
    if (!all(inputs.class %in% c("num", "cat"))){
      stop("Invalid input classes. Authorized values are 'num' and 'cat'.")
    }
  }
  # Check consistency for output
  if (length(output.class) != 1 | length(output) != 1){
    stop("Only one output may be trained at a time.")
    if (!all(output.class %in% c("num", "cat"))){
      stop("Invalid output class. Authorized values are 'num' and 'cat'.")
    }
  }
  # Check train size consistency
  if (train.size > 1 | train.size <= 0){
    stop("Unvalid train size. The value should be between 0 and 1.")
  }

  # Select relevant columns
  d <- d[, .SD, .SDcols = c(inputs, output)]

  # Input data type management
  if (inputs.class[1] != "auto"){
    # On numerical values
    sc <- inputs[inputs.class == "num"]
    if (length(sc) > 0){
      d[, (sc) := lapply(.SD, function(x) as.numeric(gsub(",",
                                                          ".",
                                                          x,
                                                          fixed = T))),
        .SDcols = sc]
    }
    # On categorical values
    sc <- inputs[inputs.class == "cat"]
    if (length(sc) > 0){
      d[, (sc) := lapply(.SD, function(x) factor(x, exclude = NULL)),
        .SDcols = sc]
    }
  }

  # Output data management
  # Apply numerical transformation on output
  if (output.class == "num"){
    d[, (output) := lapply(.SD,
                           function(x) as.numeric(gsub(",",
                                                       ".",
                                                       x,
                                                       fixed = T))),
      .SDcols = output]
  }
  # Apply categorical transformation on output
  if (output.class == "cat"){
    d[, (output) := lapply(.SD, function(x) factor(x, exclude = NULL)),
      .SDcols = output]
  }

  # Test if the number of levels is consistent
  n.lev <- unlist(d[, lapply(.SD, function(x) length(levels(x)))])
  if (!all(n.lev <= max.levels)){
    stop(cat("Too many levels on category: ",
             names(n.lev)[n.lev > max.levels][1],
             ". Max authorized value is ",
             max.levels, "."))
  }

  # NA management
  # selection of the numeric column
  sc <- sapply(d, class)
  sc <- names(sc[sc %in% c("integer", "numeric")])
  if (length(sc) > 0){
    # case infinity
    if (na.handle == "inf"){
      d[, (sc) := lapply(.SD, function(x) ifelse(is.na(x), Inf, x)),
        .SDcols = sc]
    } else {
      # case function
      fn <- function(x) {
        x[is.na(x)] <- eval(parse(text = paste0(na.handle,
                                                "(x, na.rm = T)")))
        return(x)
      }
      d[, (sc) := lapply(.SD, fn), .SDcols = sc]
    }
  }
  # Remove remaining NA
  d <- na.omit(d)

  # Create data partition
  # Set seed (for reproducibility)
  set.seed(seed)
  if (class(d[, get(output)]) == "factor"){
    d[, train := sample(.N) / .N <= train.size, by = output]
    t <- which(d$train)
  } else{
    d[order(get(output)), quantile :=  cut(1:.N,
                                           quantile(1:.N,
                                                    seq(0, 1, 0.05)))]
    d[, train := sample(.N) / .N <= train.size, by = quantile]
    t <- which(d$train)
    d[, quantile := NULL]
  }

  # Create model matrix
  formula <- paste0("~-1+", paste(inputs, collapse = "+"))
  m.d <- model.matrix(as.formula(formula), d)
  # Creation of the xgb Matrix
  dtrain <- list(data = m.d[t, ],
                 label = d[t, get(output)])
  # Creation of the test matrix
  dtest <- list(data = m.d[-t, ],
                label = d[-t, get(output)])

  # Return values
  return(list(train = dtrain,
              test = dtest,
              formula = formula,
              template = d[0, inputs, with = F],
              data = d))
}

# Calibrate Model -----

#' Calibrate Model
#'
#' \code{xg_train} trains an xgboost model in order on the data structure
#' generated by the \link[ezXg]{load_data} function.
#'
#' @param data \strong{Object}. A data structure created by the call of the
#' \link[ezXg]{load_data} function.
#' @param eta \strong{Numeric}. Eta parameter for xgboost calibration.
#' See \link[xgboost]{xgb.train} for more details.
#' @param gamma \strong{Numeric}. Gamma parameter for xgboost calibration.
#' See \link[xgboost]{xgb.train} for more details.
#' @param max_depth \strong{Numeric}. Max_depth parameter for xgboost
#' calibration. See \link[xgboost]{xgb.train} for more details.
#' @param colsample_bytree \strong{Numeric}. Colsample_bytree parameter
#' for xgboost calibration. See \link[xgboost]{xgb.train} for more details.
#' @param min_child_weight \strong{Numeric}. Min_child_weight parameter for
#' xgboost calibration. See \link[xgboost]{xgb.train} for more details.
#' @param nrounds \strong{Numeric}. Nrounds parameter for xgboost calibration.
#' See \link[xgboost]{xgb.train} for more details.
#' @param nthread \strong{Numeric}. Nthread parameter for xgboost calibration.
#' See \link[xgboost]{xgb.train} for more details.
#' @param verbose \strong{Numeric}. Verbose parameter for xgboost calibration.
#' See \link[xgboost]{xgb.train} for more details.
#' @param cv \strong{Numeric}. Number of folds in cross validation. If this
#' parameter is set to 1, this means that cross-validation will not be
#' performed.
#' @param seed \strong{Numeric}. Seed for computation reproducability.
#' @param objective \strong{Character}. Objective function for the
#' optimization. . Eta parameter for xgboost calibration. See
#'  \link[xgboost]{xgb.train} for more details. Can be set to \emph{auto}
#'  in order to let the function choose the better model regarding the
#'  output variable.
#'
#' @return A trained model with following values:
#' \itemize{
#'  \item{\strong{model}}: calibrated model as returned by the
#'  \link[xgboost]{xgb.train} function.
#'  \item{\strong{ntree}}: optimal number of tree according to the test set.
#'  \item{\strong{formula}}: the formula used for constructing the model matrix
#'  and that is applied when running the model.
#'  \item{\strong{template}}: an empty \code{data.table} that has saved all the
#'  input values and that is used to appropriately format data when using
#'  the prediction function.
#'  \item{\strong{labels}}: The possible labels for prediction when performing
#'  a classification task with xgboost.
#' }
#' In case the parameter \emph{cv} is set to anithing but 1, the function
#' returns a 1 line data.table with the average error on the
#' cross-validation.
#'
#' @import data.table
#' @import xgboost
#'
#' @examples
#' temp.file <- tempfile()
#' data(iris)
#' fwrite(iris, temp.file, row.names = FALSE)
#' d <- load_data(temp.file, output = "Species", train.size = 0.8)
#' md <- xg_train(d)
#'
#' @export
xg_train <- function(data,
                     eta = 0.3,
                     gamma = 0,
                     max_depth = 6,
                     colsample_bytree = 1,
                     min_child_weight = 1,
                     nrounds = 100,
                     nthread = 2,
                     verbose = 1,
                     cv = 1,
                     seed = 1,
                     objective = "auto"){
  # Copy data
  d <- data
  # Seed value
  set.seed(seed)

  # definition of the objective function
  if (objective == "auto"){
    if (is.numeric(d$train$label)){
      # Regression
      if (all(d$train$label %in% c(0, 1))){
        objective <- "reg:logistic"
      } else {
        objective <- "reg:linear"
      }
    } else {
      # Classification
      if ( length(levels(d$train$label)) == 2){
        objective <- "binary:logistic"
      } else {
        objective <- "multi:softprob"
      }
    }
  }

  # Labels defintion in case of classification
  if (objective %in% c("reg:linear", "reg:logistic")){
    lab <- NULL
  } else if (objective %in% c("binary:logistic", "multi:softprob")){
    lab <- levels(d$train$label)
    num_class <- length(lab)
    d$train$label <- as.numeric(d$train$label) - 1
    d$test$label <- as.numeric(d$test$label) - 1
  }

  # Definition of the parameters for running the xgboost model
  params <- list(eta = eta,
                 gamma = gamma,
                 max_depth = max_depth,
                 colsample_bytree = colsample_bytree,
                 min_child_weight = min_child_weight,
                 nrounds = nrounds,
                 nthread = nthread,
                 objective = objective)
  if (objective == "multi:softprob"){
    params$num_class <- num_class
  }

  # Creation of the xgb matrices
  dtrain <- xgb.DMatrix(data = d$train$data, label = d$train$label)
  dtest <- xgb.DMatrix(data = d$test$data, label = d$test$label)

  # Cross validation or not
  if (cv < 2){
    # No cross validation
    if (length(d$test$label) > 0){
      # Case of a test dataset
      # Creation of the watchlist
      watchlist <- list(train = dtrain, test = dtest)
      # Run the model
      bst <- xgb.train(data = dtrain,
                       params = params,
                       nrounds = nrounds,
                       watchlist = watchlist,
                       verbose = verbose)
      # Return the output
      return(
        list(model = bst,
             ntree = as.numeric(which.min(unlist(bst$evaluation_log[, 3]))),
             template = d$template,
             formula = d$formula,
             labels = lab)
      )
    } else {
      # Case of no test dataset
      # Creation of the watchlist
      watchlist <- list(train = dtrain)
      # Run the model
      bst <- xgb.train(data = dtrain,
                       params = params,
                       nrounds = nrounds,
                       watchlist = watchlist,
                       verbose = verbose)
      # Return the output
      return(
        list(model = bst,
             ntree = nrounds,
             template = d$template,
             formula = d$formula,
             labels = lab)
      )
    }
  } else{
    # case of a cross validation
    # Create partition
    Xtrain <- d$train$data
    ytrain <- data.table(lab = d$train$label)
    if (class(ytrain$lab) == "factor"){
      ytrain[, class := sample(1:cv, .N, replace = T), by = lab]
    } else{
      ytrain[order(lab),
             quantile :=  cut(1:.N, quantile(1:.N, seq(0, 1, 0.05)))]
      ytrain[, class := sample(1:cv, .N, replace = T), by = quantile]
    }
    # Loop on model
    rez <- data.table()
    for (i in 1:cv){
      if (verbose > 0){
        cat("           Fold", i, "/", cv, "\n")
        Sys.sleep(1)
      }
      # Training and test matrices
      dtrain <- xgb.DMatrix(data = Xtrain[ytrain$class != i, ],
                            label = ytrain[class != i, (lab)])
      dtest <- xgb.DMatrix(data = Xtrain[ytrain$class == i, ],
                           label = ytrain[class == i, (lab)])
      watchlist <- list(train = dtrain, test = dtest)
      # Model training
      bst <- xgb.train(data = dtrain,
                       params = params,
                       nrounds = nrounds,
                       watchlist = watchlist,
                       verbose = verbose)
      # Model validation
      rez <- rbindlist(list(rez,
                            data.table(cv = i,
                                       tree = as.numeric(which.min(unlist(bst$evaluation_log[, 3]))),
                                       train = as.numeric(bst$evaluation_log[which.min(unlist(bst$evaluation_log[, 3])), 2]),
                                       test = min(unlist(bst$evaluation_log[, 3])))))
    }
    # Return value
    return(cbind(as.data.table(params), rez))
  }
}

# Grid Search -----

#' Grid search
#'
#' \code{xg_gs} use coordinate descent
#'  (\url{https://en.wikipedia.org/wiki/Coordinate_descent}) in order
#'  to select the best set of parameters for an xgboost model. At the end
#'  of the coordinate descent algorithm, a full search on each of the
#'  individual parameter vectors is made in order to potentially improve
#'  the selection.
#'
#' @param data \strong{Object}. A data structure created by the call of the
#' \link[ezXg]{load_data} function.
#' @param eta \strong{Numeric Vectors}. Eta parameter list for grid search.
#' See \link[xgboost]{xgb.train} for more details.
#' @param gamma \strong{Numeric Vector}. Gamma parameter list for grid search.
#' See \link[xgboost]{xgb.train} for more details.
#' @param max_depth \strong{Numeric Vector}. Max_depth parameter list for grid
#'  search. See \link[xgboost]{xgb.train} for more details.
#' @param colsample_bytree \strong{Numeric Vector}. Colsample_bytree parameter
#' list for grid search. See \link[xgboost]{xgb.train} for more details.
#' @param min_child_weight \strong{Numeric Vector}. Min_child_weight parameter
#' list for grid search. See \link[xgboost]{xgb.train} for more details.
#' @param nrounds \strong{Numeric}. Nrounds parameter for xgboost calibration.
#' See \link[xgboost]{xgb.train} for more details.
#' @param nthread \strong{Numeric}. Nthread parameter for xgboost calibration.
#' See \link[xgboost]{xgb.train} for more details.
#' @param verbose \strong{Logical}. Verbose parameter for grid search.
#' @param cv \strong{Numeric}. Number of folds in cross validation. Needs
#' to be more than 2.
#' @param seed \strong{Numeric}. Seed for computation reproducability.
#' @param objective \strong{Character}. Objective function for the
#' optimization. . Eta parameter for xgboost calibration. See
#'  \link[xgboost]{xgb.train} for more details. Can be set to \emph{auto}
#'  in order to let the function choose the better model regarding the
#'  output variable.
#'
#' @return The optimization results with the following fields:
#' \itemize{
#'  \item{\strong{param}}: the optimal set of parameters.
#'  \item{\strong{err}}: the error associated to the optimal parameter
#'  set.
#'  \item{\strong{results}}: the history of the results for the
#'  cross-validation with all the tested sets of parameters.
#' }
#'
#' @import data.table
#' @import xgboost
#' @importFrom stats as.formula median model.matrix na.omit predict quantile
#'
#' @examples
#' temp.file <- tempfile()
#' data(iris)
#' fwrite(iris, temp.file, row.names = FALSE)
#' d <- load_data(temp.file, output = "Species", train.size = 0.8)
#' t <- xg_gs(d)
#'
#' @export
xg_gs <- function(data,
                  eta = c(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
                  gamma = c(0, 0.1, 0.2, 0.3, 0.4, 0.5),
                  max_depth = c(1, 3, 4, 5, 6, 8, 10, 12, 15),
                  colsample_bytree = c(0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1),
                  min_child_weight = c(1, 3, 5, 7),
                  nrounds = 100,
                  nthread = 2,
                  cv = 5,
                  seed = 1,
                  verbose = TRUE,
                  objective = "auto"){

  # Check the parameters
  if (cv < 2) stop("Invalid number of cross validation")

  # Optimization function
  fn <- function(x){
    md <- xg_train(data,
                   cv = 5,
                   verbose = 0,
                   eta = x[1],
                   gamma = x[2],
                   max_depth = x[3],
                   colsample_bytree = x[4],
                   min_child_weight = x[5],
                   nthread = nthread,
                   nrounds = nrounds,
                   seed = seed,
                   objective = objective
    )
    return(mean(md$test))
  }

  # Definition of the grid
  grid <- list(eta = eta,
               gamma = gamma,
               max_depth = max_depth,
               colsample_bytree = colsample_bytree,
               min_child_weight = min_child_weight)

  # Initial point
  ini.pos <- list( eta = round(median(1:length(eta))),
                   gamma = round(median(1:length(gamma))),
                   max_depth = round(median(1:length(max_depth))),
                   colsample_bytree = round(median(1:length(colsample_bytree))),
                   min_child_weight = round(median(1:length(min_child_weight))))

  # Initialization of the error
  ini.err <- fn(sapply(1:5, function(x) grid[[x]][ini.pos[[x]]]))
  best.pos <- ini.pos
  best.err <- ini.err
  rez <- cbind(as.data.table(ini.pos), data.table(err = ini.err))

  # Print the initial state
  if (verbose){
    cat("Initializing with value:\n")
    cat("             eta:", grid$eta[ini.pos$eta], "\n")
    cat("             gamma:", grid$gamma[ini.pos$gamma], "\n")
    cat("             max_depth:", grid$max_depth[ini.pos$max_depth], "\n")
    cat("             colsample_bytree:",
        grid$colsample_bytree[ini.pos$colsample_bytree], "\n")
    cat("             min_child_weight:",
        grid$min_child_weight[ini.pos$min_child_weight], "\n")
    cat("with error :", best.err, "\n")
  }

  # While loop as long as there is improvement
  imp <- T
  while (imp){
    temp.rez <- data.table()
    for (i in 1:5){
      if (ini.pos[[i]] > 1){
        pos <- ini.pos
        pos[[i]] <- pos[[i]] - 1
        err <- fn(sapply(1:5, function(x) grid[[x]][pos[[x]]]))
        temp.rez <- rbind(temp.rez,
                          cbind(as.data.table(pos), data.table(err = err)))
        if (err < best.err){
          best.pos <- pos
          best.err <- err
        }
      }
      if (ini.pos[[i]] < length(grid[[i]])){
        pos <- ini.pos
        pos[[i]] <- pos[[i]] + 1
        err <- fn(sapply(1:5, function(x) grid[[x]][pos[[x]]]))
        temp.rez <- rbind(temp.rez,
                          cbind(as.data.table(pos), data.table(err = err)))
        if (err < best.err){
          best.pos <- pos
          best.err <- err
        }
      }
    }
    rez <- rbind(rez, temp.rez)
    if (best.err < ini.err){
      val.modif <- names(which(unlist(ini.pos) - unlist(best.pos) != 0))
      if (verbose){
        cat(val.modif,
            "switched from",
            grid[[val.modif]][ini.pos[[val.modif]]],
            "to",
            grid[[val.modif]][best.pos[[val.modif]]],
            "for new error",
            best.err,
            "\n"
        )
      }
      ini.pos <- best.pos
      ini.err <- best.err
    } else{
      imp <- F
    }
  }
  # From the best position, check all values on univariates.
  for (i in 1:5){
    topass <- best.pos[[i]]
    for (j in 1:length(grid[[i]])){
      if (j != topass){
        pos <- best.pos
        pos[[i]] <- j
        err <- fn(sapply(1:5, function(x) grid[[x]][pos[[x]]]))
        rez <- rbind(rez,
                     cbind(as.data.table(pos), data.table(err = err)))
        if (err < best.err){
          val.modif <- names(which(unlist(pos) - unlist(best.pos) != 0))
          if (verbose){
            cat(val.modif,
                "switched from",
                grid[[val.modif]][best.pos[[val.modif]]],
                "to",
                grid[[val.modif]][pos[[val.modif]]],
                "for new error",
                err,
                "\n"
            )
          }
          best.pos <- pos
          best.err <- err
        }
      }
    }
  }

  # Select the best parameters
  best.param <- sapply(1:5, function(x) grid[[x]][best.pos[[x]]])
  names(best.param) <- names(best.pos)
  for (i in 1:5){
    rez[, c(names(rez)[i]) := grid[[i]][unlist(rez[, i, with = F])]]
  }

  # Return the results
  return(list(param = best.param,
              err = best.err,
              results = rez))
}

# Predict Function -----

#' Predict on new values
#'
#' \code{xg_predict} use the previously trained xgboost model to perform
#' prediction on new data.
#'
#' @param model \strong{Object} A model object created by the function
#' \link[ezXg]{xg_train}.
#' @param data \strong{data.frame}. A data.frame or data.table structure with
#' column names equal to the input names.
#'
#' @return The prediction with the following fields:
#' \itemize{
#'  \item{\strong{pred}}: the prediction for the model.
#'  \item[\strong{proba}}: (optionnal) the associated probabilities for the
#'  prediction in case of a classification
#' }
#'
#' @import data.table
#' @import xgboost
#'
#' @examples
#' temp.file <- tempfile()
#' data(iris)
#' fwrite(iris, temp.file, row.names = FALSE)
#' d <- load_data(temp.file, output = "Species", train.size = 0.8)
#' md <- xg_train(d)
#' p <- xg_predict(md, iris)
#'
#' @export
xg_predict <- function(model,
                       data){
  # formating data
  col <- sapply(model$template, class)
  data <- data.table(data)
  col2cat <- names(which(col == "factor"))
  for (c in col2cat){
    data[, (c) := factor(as.character(get(c)),
                         levels = levels(model$template[, get(c)]))]
  }
  col2num <- names(which(col == "numeric"))
  if (length(col2num) > 0){
    data[, (col2num) := lapply(.SD,
                               function(x) as.numeric(gsub(",",
                                                           ".",
                                                           x,
                                                           fixed = T))),
         .SDcols = col2num]
  }

  # Model matrix
  m.d <- model.matrix(as.formula(model$formula), data)

  # Prediction
  if (model$model$params$objective %in% c("reg:linear", "reg:logistic")){
    # Case of regression
    return(list(pred = predict(model$model, m.d, ntreelimit = model$ntree)))
  } else if (model$model$params$objective == "binary:logistic"){
    # Case of binary classification
    p <- predict(model$model, m.d, ntreelimit = model$ntree)
    proba <- p
    pred <- ifelse(p < 0.5, model$labels[1], model$labels[2])

    # Return
    return(list(pred = pred,
                proba = proba))
  } else if (model$model$params$objective == "multi:softprob"){
    # Case of multi-class classification
    p <- predict(model$model, m.d, ntreelimit = model$ntree)
    proba <- matrix(p, ncol = model$model$params$num_class, byrow = TRUE)
    pred <- model$labels[apply(proba, 1, which.max)]

    # Return
    return(list(pred = pred,
                proba = proba))
  }
}

# Auto Machine Learning Function -----

xg_auto_ml <- function(data,
                       parameters){

}
