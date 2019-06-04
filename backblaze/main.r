require(reticulate)
require(randomForestSRC)
require(Metrics)
require(ggplot2)
require(gridExtra)
## set the seed to make your partition reproducible
set.seed(123)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
train <- R_read_pickle_file(paste(data_path, "data_train_ST12000NM0007.pkl", sep=""))
test <- R_read_pickle_file(paste(data_path, "data_test_ST12000NM0007.pkl", sep=""))

# Fit random forest
rfr = rfsrc(RUL ~ ., data = train[, !names(train) %in% c("date", "failure", "capacity_bytes", "serial_number")],
            do.trace = TRUE, ntree = 1000, nsplit = 10, nodesize = 5)
# Variable importance
print(vimp(rfr, do.trace = TRUE)$importance)
# Test forest
rfr.pred = predict(rfr,test[, !names(test) %in% c("date", "failure", "capacity_bytes", "serial_number")])
# Write result to df
y_pred = rfr.pred$predicted
y_test = test$RUL
df = data.frame(y_pred, y_test, test$serial_number)
df$d_y = y_pred - y_test
# Measure accuracy
mse(y_pred, y_test)
# Plot results
p1 = ggplot(data=df, aes(x=y_test, y=y_test, group=test.serial_number)) +
  geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
p2 = ggplot(data=df, aes(x=y_test, y=y_pred, group=test.serial_number)) +
  geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")
p3 = ggplot(data=df, aes(x=y_test, y=d_y, group=test.serial_number)) +
  geom_line(aes(color=test.serial_number)) + scale_x_reverse() + theme(legend.position="none")

grid.arrange(p1, p2, p3, nrow = 3)
