require(reticulate)
require(randomForestSRC)
require(Metrics)
require(ggplot2)
require(gridExtra)
require(dplyr)
require(akima)
require(tictoc)
memory.limit(100000)
options(warn=-1)
#options(warn=0)
## set the seed to make your partition reproducible
set.seed(123)
# Set number of most important variables to use
topimpv = 30
ntree = 200
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))

# 'ST4000DM000' failed! RAM
# 'TOSHIBA MQ01ABF050' failed, characters
### TODO!!!!
# models = c('WDC WD800AAJS', 'HGST HMS5C4040ALE640', 'HGST HMS5C4040BLE640', 'Hitachi HDS5C3030ALA630',
#            'Hitachi HDS5C4040ALE630', 'Hitachi HDS722020ALA330', 'Hitachi HDS723030ALA640',
#            'ST320LT007', 'ST500LM012 HN', 'ST500LM030', 'ST4000DM001', 'ST4000DX000',
#            'ST6000DX000', 'ST8000DM002', 'ST8000NM0055', 'ST10000NM0086', 'ST12000NM0007',
#            'ST3160318AS', 'WDC WD30EFRX', 'WDC WD60EFRX')
models = c('WDC WD800AAJS')
print(paste("Program started at ", Sys.time(), sep=""))
for (model in models)
{
  tic()
  print(paste("Starting: ", model, sep=""))
  train <- R_read_pickle_file(paste(data_path, "data\\rawdata\\data_train_", model, ".pkl", sep=""))
  test <- R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_", model, ".pkl", sep=""))
  # Train statistics
  train_length = nrow(train)
  train_serials = sapply(train['serial_number'], function(x) length(unique(x)))
  # Load variable importance
  vimp.order = read.csv(file=paste(data_path, "data\\rawdata\\vimp_", model,".csv", sep=""))
  vimp.top = as.character(vimp.order[1:topimpv, 1])
  vimp.top = c(vimp.top, "RUL")
  ### Sample w/ cutoff
  cutoff_test = as.data.frame((test %>% filter(failure == 0)) %>% group_by(serial_number) %>% sample_n(size = 1))
  cutoff_train = as.data.frame((train %>% filter(failure == 0)) %>% group_by(serial_number) %>% sample_n(size = 1))
  ### Median Predict Label
  median_RUL_1 = median(train$RUL)
  ### Median Predict History
  media_life_train = median(data.frame(table((train %>% filter(failure == 0))$serial_number))$Freq)
  median_life_test_until_cutoff = data.frame(table((test %>% filter(failure == 0))$serial_number))$Freq - cutoff_test$RUL
  median_RUL_2 = pmax(media_life_train - median_life_test_until_cutoff, 0)
  base1_rmse = round(rmse(median_RUL_1, cutoff_test$RUL), digits = 2)
  base2_rmse = round(rmse(median_RUL_2, cutoff_test$RUL), digits = 2)
  # Fit & tune random forest w/ cutoff data
  #rfr = tune(RUL ~ ., data = train[, names(train) %in% vimp.top],
  #           trace = TRUE, ntreeTry=ntree)
  log = read.csv(file=paste(data_path, "data\\rawdata\\my_models.csv", sep=""), sep=";", stringsAsFactors=FALSE)
  id = tail(log, 1)$ID + 1
  # Save the model to disk
  saveRDS(rfr, paste(data_path, "data\\rawdata\\rfsrc_", id, ".rds", sep=""))
  # Test statistics
  test_length = nrow(test)
  test_serials = sapply(test['serial_number'], function(x) length(unique(x)))
  test_mean_RUL = round(sapply(cutoff_test['RUL'], mean), digits=2)
  test_max_RUL = max(cutoff_test['RUL'])
  test_sd_RUL = round(sapply(cutoff_test['RUL'], sd), digits=2)
  test_sd_RUL[is.na(test_sd_RUL)] = 0
  test_sd_RUL = as.numeric(test_sd_RUL)
  # Test forest
  y_pred = predict(rfr$rf,cutoff_test[, !names(cutoff_test) %in% c("date", "failure", "capacity_bytes", "serial_number")])$predicted
  y_test = cutoff_test$RUL
  df = data.frame(y_pred, y_test, cutoff_test$serial_number)
  df$d_y = y_pred - y_test
  # Print Error
  test_rmse = round(rmse(y_pred, y_test), digits=2)
  timing = toc()
  timing = round(timing$toc - timing$tic, digits = 2)
  ### Collect statistics and output to csv
  log[nrow(log) + 1,] = list(id, as.character(Sys.Date()), model, topimpv, train_length, train_serials, ntree,
                             rfr$optimal[1], rfr$optimal[2], test_length, test_serials, test_mean_RUL,
                             test_max_RUL, test_sd_RUL, test_rmse, base1_rmse, base2_rmse, timing)
  write.table(log, file=paste(data_path, "data\\rawdata\\my_models.csv", sep=""), sep=";", row.names = FALSE)
  print(paste("Model ", model, " finished in ", timing, sep=""))
}
print(paste("Program finished at ", Sys.time(), sep=""))

