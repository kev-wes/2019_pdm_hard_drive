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
sample_size = 0.1
# 'ST4000DM000' failed! RAM Done on c5n.2xlarge
# 'ST' Done on c5n.2xlarge
# 'TOSHIBA MQ01ABF050' failed, characters
### TODO!!!!
# models = c('WDC WD800AAJS', 'HGST HMS5C4040ALE640', 'HGST HMS5C4040BLE640', 'Hitachi HDS5C3030ALA630',
#            'Hitachi HDS5C4040ALE630', 'Hitachi HDS722020ALA330', 'Hitachi HDS723030ALA640',
#            'ST320LT007', 'ST500LM012 HN', 'ST500LM030', 'ST4000DM001', 'ST4000DX000',
#            'ST6000DX000', 'ST8000DM002', 'ST8000NM0055', 'ST10000NM0086', 'ST12000NM0007',
#            'ST3160318AS', 'WDC WD30EFRX', 'WDC WD60EFRX')
models = c('TOSHIBA MQ01ABF050')
print(paste("Program started at ", Sys.time(), sep=""))
for (model in models)
{
  tic()
  print(paste("Start: ", model, sep=""))
  train <- R_read_pickle_file(paste(data_path, "data\\rawdata\\data_train_", model, ".pkl", sep=""))
  # Train statistics
  train_length = nrow(train)
  train_serials = sapply(train['serial_number'], function(x) length(unique(x)))
  # Load variable importance
  if(file_test("-f", paste(data_path, "data\\rawdata\\vimp_", model,".csv", sep=""))==FALSE){
    # Sample for VIMP
    if(sample_size != 0){
      print("Start downsampling")
      sampled_train <- train %>% group_by(serial_number) %>%  summarize(count=n())
      length = nrow(sampled_train)
      train_vimp_ser <- sampled_train %>%  sample_n(length*sample_size)
      train = train[train$serial_number %in% train_vimp_ser$serial_number,]
    }
    # Train simple RF
    print("Start simple RF training for VIMP")
    rfr = rfsrc.fast(RUL ~ ., data = train[, !names(train) %in% c("date", "failure", "capacity_bytes", "serial_number", "model")],
                     do.trace = TRUE, ntree = 100, nodesize = 1)
    print("Starting VIMP")
    vimp.df = t(data.frame(as.list(vimp(rfr, do.trace = TRUE, importance="permute")$importance)))
    vimp.order = data.frame(vimp.df[order(-vimp.df),])
    write.csv(as.data.frame(vimp.order), file=paste(data_path, "data\\rawdata\\vimp_", model,".csv", sep=""))
  }
  # Load real train set again
  train <- R_read_pickle_file(paste(data_path, "data\\rawdata\\data_train_", model, ".pkl", sep=""))
  vimp.order = read.csv(file=paste(data_path, "data\\rawdata\\vimp_", model,".csv", sep=""))
  # Select top n variables
  vimp.top = as.character(vimp.order[1:topimpv, 1])
  vimp.top = c(vimp.top, "RUL")
  # Fit & tune random forest
  print("Start tuning")
  rfr = tune(RUL ~ ., data = train[, names(train) %in% vimp.top],
             trace = TRUE, ntreeTry=ntree, nodesizeTry = c(1:9, seq(10, 25, by = 5)))
  log = read.csv(file=paste(data_path, "data\\rawdata\\my_models.csv", sep=""), sep=";", stringsAsFactors=FALSE)
  id = tail(log, 1)$ID + 1
  # Save the model to disk
  saveRDS(rfr, paste(data_path, "data\\rawdata\\rfsrc_", id, ".rds", sep=""))
  rm(train)
  # Test statistics
  test <- R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_", model, ".pkl", sep=""))
  test_length = nrow(test)
  test_serials = sapply(test['serial_number'], function(x) length(unique(x)))
  test_mean_RUL = round(sapply(test['RUL'], mean), digits=2)
  test_max_RUL = max(test['RUL'])
  test_sd_RUL = round(sapply(test['RUL'], sd), digits=2)
  test_sd_RUL[is.na(test_sd_RUL)] = 0
  test_sd_RUL = as.numeric(test_sd_RUL)
  # Test forest
  y_pred = predict(rfr$rf,test[, names(test) %in% vimp.top])$predicted
  y_test = test$RUL
  # Add baseline to dataframe
  df = data.frame(test$serial_number, y_pred, y_test, pred_error = y_pred - y_test,
                  base_y = median(train$RUL), base_error = median(train$RUL) - y_test)
  df$pred_error = y_pred - y_test
  # Print Error
  test_rmse = round(rmse(y_pred, y_test), digits=2)
  base1_rmse = round(rmse(median(train$RUL), y_test), digits = 2)
  # Plot results
  pdf(paste(data_path, "data\\rawdata\\plot_", id, ".pdf", sep=""))
  p1 = ggplot(data=df, aes(x=y_test, y=y_test, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  p2 = ggplot(data=df, aes(x=y_test, y=y_pred, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  p3 = ggplot(data=df, aes(x=y_test, y=pred_error, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  p4 = ggplot(data=df, aes(x=y_test, y=base_y, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  p5 = ggplot(data=df, aes(x=y_test, y=base_error, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  grid.arrange(p1, p2, p3, p4, p5, nrow = 5, top = model)
  dev.off()
  timing = toc()
  timing = round(timing$toc - timing$tic, digits = 2)
  ### Collect statistics and output to csv
  log[nrow(log) + 1,] = list(id, as.character(Sys.Date()), model, topimpv, train_length, train_serials, ntree,
                             rfr$optimal[1], rfr$optimal[2], test_length, test_serials, test_mean_RUL,
                             test_max_RUL, test_sd_RUL, test_rmse, base1_rmse, "", timing)
  write.table(log, file=paste(data_path, "data\\rawdata\\my_models.csv", sep=""), sep=";", row.names = FALSE)
  print(paste("Model ", model, " finished in ", timing, sep=""))
}
print(paste("Program finished at ", Sys.time(), sep=""))