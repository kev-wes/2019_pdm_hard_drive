require(reticulate)
require(randomForestSRC)
require(Metrics)
require(ggplot2)
require(gridExtra)
require(dplyr)
memory.limit(100000)
options(warn=-1)
#options(warn=0)
## set the seed to make your partition reproducible
set.seed(123)
# Set number of most important variables to use
topimpv = 30
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
# Hyperparameter tuning
split = c()

# 'ST4000DM000' failed! RAM
# 'TOSHIBA MQ01ABF050' failed, characters
models = c('WDC WD30EFRX',
           'WDC WD60EFRX', 'WDC WD800AAJS')
#models = c('WDC WD30EFRX')

start_tot <- Sys.time()
#for (nsplit in splits)
for (model in models)
  {
  start <- Sys.time()
  print(paste("Starting: ", model, sep=""))
  train <- R_read_pickle_file(paste(data_path, "data\\rawdata\\data_train_", model, ".pkl", sep=""))
  test <- R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_", model, ".pkl", sep=""))
  # Print train statistics
  print(paste("Length of train:", nrow(train)))
  print(paste("Number of train serials:", sapply(train['serial_number'], function(x) length(unique(x)))))
  # Fit random forest
  rfr = rfsrc(RUL ~ ., data = train[, !names(train) %in% c("date", "failure", "capacity_bytes", "serial_number")],
            do.trace = TRUE, ntree = 200, nsplit = 10, nodesize = 1)
  saveRDS(rfr, paste(data_path, "data\\rawdata\\rfsrc_", model, ".rds", sep=""))
  # Print test statistics
  print(paste("Length of test:", nrow(test)))
  print(paste("Number of test serials:", sapply(test['serial_number'], function(x) length(unique(x)))))
  print(paste("Mean RUL after elbow", round(sapply(test['RUL'], mean), digits=2)))
  print(paste("Min RUL", min(test['RUL'])))
  print(paste("Max RUL", max(test['RUL'])))
  print(paste("SD RUL", round(sapply(test['RUL'], sd), digits=2)))
  # Test forest
  y_pred = predict(rfr,test[, !names(test) %in% c("date", "failure", "capacity_bytes", "serial_number")])$predicted
  y_test = test$RUL
  df = data.frame(y_pred, y_test, test$serial_number)
  df$d_y = y_pred - y_test
  # Print Error
  print(paste("RMSE of prediction without VIMP:", round(rmse(y_pred, y_test), digits=2)))
  # Plot results
  pdf(paste(data_path, "data\\rawdata\\plot_", model, ".pdf", sep=""))
  p1 = ggplot(data=df, aes(x=y_test, y=y_test, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  p2 = ggplot(data=df, aes(x=y_test, y=y_pred, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  p3 = ggplot(data=df, aes(x=y_test, y=d_y, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  grid.arrange(p1, p2, p3, nrow = 3, top = model)
  dev.off()
  # Variable importance
  vimp.df = t(data.frame(as.list(vimp(rfr, do.trace = TRUE)$importance)))
  vimp.order = data.frame(vimp.df[order(-vimp.df),])
  write.csv(as.data.frame(vimp.order), file=paste(data_path, "data\\rawdata\\vimp_", model,".csv", sep=""))
  vimp.top = attributes(vimp.order)$row.names[1:topimpv]
  print(paste("The most important variables are", vimp.top))
  vimp.top = c(vimp.top, "RUL")
  # Retrain only with most important variables
  rfr = rfsrc(RUL ~ ., data = train[, names(train) %in% vimp.top],
              do.trace = TRUE, ntree = 200, nsplit = 10, nodesize = 5)
  saveRDS(rfr, paste(data_path, "data\\rawdata\\rfsrc_vimp_", model, ".rds", sep=""))
  # Test forest
  y_pred = predict(rfr,test[, names(test) %in% vimp.top])$predicted
  y_test = test$RUL
  df = data.frame(y_pred, y_test, test$serial_number)
  df$d_y = y_pred - y_test
  # Print Error
  print(paste("RMSE of prediction with VIMP:", round(rmse(y_pred, y_test), digits=2)))
  # Plot results
  pdf(paste(data_path, "data\\rawdata\\plot_vimp_", model, ".pdf", sep=""))
  p1 = ggplot(data=df, aes(x=y_test, y=y_test, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  p2 = ggplot(data=df, aes(x=y_test, y=y_pred, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  p3 = ggplot(data=df, aes(x=y_test, y=d_y, group=test.serial_number)) +
    geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
  grid.arrange(p1, p2, p3, nrow = 3, top = model)
  dev.off()
  end <- Sys.time()
  print(paste("Model ", model, " finished in ", round(end - start, digits = 2),
              "s. Total: ", round(end - start_tot, digits = 2), "s.", sep=""))
}
end <- Sys.time()
print(paste("Program finished after ", round(end - start_tot, digits=2), "s.", sep=""))

#rfr <- readRDS(paste(data_path, "data\\rawdata\\rfsrc_ST12000NM0007.rds", sep=""))



