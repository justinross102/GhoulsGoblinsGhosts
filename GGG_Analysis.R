
# load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom) # reading and writing file
library(embed) # target encoding
library(discrim) # naive bayes
library(keras)
library(tensorflow)



# load in the data --------------------------------------------------------

setwd("~/Documents/BYU/stat348/GhoulsGoblinsGhosts")
train <- vroom("train.csv") %>% 
  mutate(type = as.factor(type))

test <- vroom("test.csv") # already has 'type' removed

# check NAs
na_count_train <- colSums(is.na(missSet)) 
na_count_test <- colSums(is.na(test))
print(na_count_train)
print(na_count_test)

# check correlation
cor_matrix <- cor(select(train, where(is.numeric)))
print(cor_matrix)

# imputation practice -----------------------------------------------------


missSet <- vroom("trainWithMissingValues.csv")
columns_with_missing_values <- colnames(missSet)[apply(is.na(missSet), 2, any)]
print(columns_with_missing_values)

missing_recipe <- recipe(type ~ ., missSet) %>% 
  step_impute_mean(bone_length, rotting_flesh, hair_length)

knn_impute <- recipe(type ~ ., missSet) %>% 
  step_impute_knn(bone_length, impute_with = imp_vars(has_soul), neighbors=10) %>% 
  step_impute_knn(rotting_flesh, impute_with = imp_vars(has_soul), neighbors=10) %>% 
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul), neighbors=10)

prepped_recipe <- prep(second_try)
baked <- bake(prepped_recipe, new_data = missSet)

rmse_vec(train[is.na(missSet)], baked[is.na(missSet)])

# format function ---------------------------------------------------------

predict_and_format <- function(model, newdata, filename){
  predictions <- predict(model, new_data = newdata)
  
  submission <- predictions %>% 
    mutate(id = test$id) %>% 
    rename("type" = ".pred_class") %>% 
    select(2,1)
  
  vroom_write(submission, filename, delim = ',')
}

# recipes -----------------------------------------------------------------

basic_recipe <- recipe(type ~ ., train) %>% 
  step_rm(id) %>% 
  step_dummy(color) %>% 
  step_normalize(all_numeric_predictors())

nn_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role="id") %>%
  step_mutate(color = as.factor(color)) %>%  # Turn 'color' into a factor
  step_dummy(color, one_hot = TRUE) %>% # dummy encode 'color'
  step_range(all_numeric_predictors(), min=0, max=1) # scale to [0,1]

# apply the recipe to the data
prepped_recipe <- prep(second_recipe)
baked <- bake(prepped_recipe, new_data = train)
baked


# random forest -----------------------------------------------------------

rand_forest_mod <- rand_forest(mtry = tune(),
                               min_n=tune(),
                               trees = 3000) %>% 
  set_engine("ranger") %>%
  set_mode("classification")

rand_forest_wf <- workflow() %>%
  add_recipe(basic_recipe) %>%
  add_model(rand_forest_mod)

rand_forest_tuning_grid <- grid_regular(mtry(range = c(1, (ncol(train)-1))),
                                        min_n(),
                                        levels = 5) ## L^2 total tuning possibilities

forest_folds <- vfold_cv(train, v = 10, repeats = 1)

CV_results <- rand_forest_wf %>%
  tune_grid(resamples = forest_folds,
            grid = rand_forest_tuning_grid,
            metrics = metric_set(accuracy)) # f_meas, sens, recall, spec, precision, accuracy

forest_bestTune <- CV_results %>%
  select_best("accuracy")

final_forest_wf <- rand_forest_wf %>%
  finalize_workflow(forest_bestTune) %>%
  fit(data = train)

predict_and_format(final_forest_wf, test, "random_forest_predictions.csv")
# 0.72022

# naive bayes -------------------------------------------------------------

nb_mod <- naive_Bayes(Laplace = tune(),
                      smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(basic_recipe) %>%
  add_model(nb_mod)

# cross validation
nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)

nb_folds <- vfold_cv(train, v = 5, repeats = 2)

CV_results <- nb_wf %>%
  tune_grid(resamples = nb_folds,
            grid = nb_tuning_grid,
            metrics = metric_set(accuracy)) # f_meas, sens, recall, spec, precision, accuracy, roc_auc

nb_bestTune <- CV_results %>%
  select_best("accuracy")

final_nb_wf <- nb_wf %>%
  finalize_workflow(nb_bestTune) %>%
  fit(data = train)

predict_and_format(final_nb_wf, test, "naive_bayes_preds.csv")
# 0.72778


# knn ---------------------------------------------------------------------

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_workflow <- workflow() %>%
  add_recipe(basic_recipe) %>%
  add_model(knn_model)

# cross validation
knn_tuning_grid <- grid_regular(neighbors(),
                                levels = 5)

knn_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- knn_workflow %>%
  tune_grid(resamples = knn_folds,
            grid = knn_tuning_grid,
            metrics = metric_set(accuracy))

knn_bestTune <- CV_results %>%
  select_best("accuracy")

# finalize workflow
final_knn_wf <- knn_workflow %>%
  finalize_workflow(knn_bestTune) %>%
  fit(data = train)

predict_and_format(final_knn_wf, test, "./knn_predictions.csv")
# 0.70132


# svm ---------------------------------------------------------------------

# SVM models

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_model(svmRadial) %>%
  add_recipe(basic_recipe)

# cross validation
svm_tuning_grid <- grid_regular(cost(),
                                rbf_sigma(),
                                levels = 5)

svm_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- svm_wf %>%
  tune_grid(resamples = svm_folds,
            grid = svm_tuning_grid,
            metrics = metric_set(accuracy))

svm_bestTune <- CV_results %>%
  select_best("accuracy")

# finalize workflow
final_svm_wf <- svm_wf %>%
  finalize_workflow(svm_bestTune) %>%
  fit(data = train)

predict_and_format(final_svm_wf, test, "./svmRadial_predictions.csv")
# 0.72778

# neural networks ---------------------------------------------------------

nn_model <- mlp(hidden_units = tune(),
                epochs = 50, #or 100 or 250
                activation="relu") %>%
  set_engine("keras", verbose=0) %>% #verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 5)),
                            levels = 5)

nn_folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- nn_wf %>%
  tune_grid(resamples = nn_folds,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy))

nn_bestTune <- CV_results %>%
  select_best("accuracy")

tuned_nn <- nn_wf %>%
  tune_grid(...)

tuned_nn %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results





