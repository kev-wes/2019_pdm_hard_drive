### Globals ###
set.seed(123)
memory.limit(100000)

### SOF Plot the trajectories of all criticial measurements ###
require(reticulate)
require(ggplot2)
require(gridExtra)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
df.test <- R_read_pickle_file(paste(data_path, "data\\rawdata\\TEST.pkl", sep=""))
colnames(df.test)
str(colnames)
ggplot(data=df.test, aes(x=RUL, y=Reallocated_Sector_Count, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=Offline_Uncorrectable, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=Reported_Uncorrectable_Errors, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=Command_Timeout, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=Current_Pending_Sector_Count, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_1_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_1_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_3_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_4_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_5_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_7_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_7_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_9_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_12_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_187_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_190_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_190_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_192_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_193_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_193_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_194_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_194_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_195_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_195_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
p28 = ggplot(data=df.test, aes(x=RUL, y=smart_197_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_198_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_199_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_200_normalized, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_240_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_241_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
ggplot(data=df.test, aes(x=RUL, y=smart_242_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")

### SOF Analyze NaN share per model and smart value ###
library(xlsx)
require(reticulate)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
df = R_read_pickle_file(paste(data_path, "data\\rawdata\\data_raw.pkl", sep=""))
nan = data.frame(matrix(NA, nrow = 63))
for (col in grep("smart_", colnames(df), value = TRUE)){
  test = as.formula(paste(col, "~ model"))
  nan[[col]] = aggregate(test, data=df, function(x) {sum(is.na(x))/sum(is.na(x) | !is.na(x))}, na.action = NULL)
  print(paste(col, "added"))
}
nan[1] <- NULL
toDelete <- seq(1, 124, 2)
nan=nan2
nan <- nan[,-toDelete]
write.csv(as.data.frame(nan), file=paste(data_path, "data\\rawdata\\nan_analysis.csv", sep="")) 
### EOF Analyze NaN share per model and smart value ###

### SOF Coefficient of Variation ###
require(reticulate)
require(dplyr)
require(ggplot2)

data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
df = R_read_pickle_file(paste(data_path, "data\\rawdata\\data_preprocessed_T.pkl", sep=""))
df = bind_rows(df,R_read_pickle_file(paste(data_path, "data\\rawdata\\data_preprocessed_H.pkl", sep="")))
df = bind_rows(df,R_read_pickle_file(paste(data_path, "data\\rawdata\\data_preprocessed_W.pkl", sep="")))
df = bind_rows(df,R_read_pickle_file(paste(data_path, "data\\rawdata\\data_preprocessed_ST.pkl", sep="")))
df = df %>% 
select(matches("(raw|norm|Count|Error|Errors|Timeout|Uncorrectable|Rate|RUL)$"))

mad = sapply(df[, !names(df) %in% c("date", "failure", "capacity_bytes",
                                               "serial_number", "model", "RUL")], mad, na.rm=TRUE)
median = sapply(df[, !names(df) %in% c("date", "failure", "capacity_bytes",
                                              "serial_number", "model", "RUL")], median, na.rm=TRUE)
mean = sapply(df[df$RUL ==i, !names(df) %in% c("date", "failure", "capacity_bytes",
                                                 "serial_number", "model", "RUL")], mean, na.rm=TRUE)
sd = sapply(df[df$RUL ==i, !names(df) %in% c("date", "failure", "capacity_bytes",
                                               "serial_number", "model", "RUL")], sd, na.rm=TRUE)
cov2 = mean/sd
cov2 = mad / median

cov2[!is.finite(cov2)] <- 0
cov2 = data.frame(cov2)
cov2$measure = row.names(cov2)
ggplot(aes(x = measure, y = cov2, fill = measure), data = cov2[cov2$cov2 != 0,]) +
  geom_bar(stat = 'identity') +
  labs(x='Measurement', y='MAD/Median Ratio') +
  theme(axis.text.x = element_text(angle = 90),
        axis.text=element_text(size=20), 
        axis.title=element_text(size=20),
        legend.position = "none")
mean(cov2$cov2)
### EOF Coefficient of Variation ###

### SOF Plot All Trajectories ###
require(reticulate)
require(dplyr)
require(ggplot2)
library(gridExtra)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
df = R_read_pickle_file(paste(data_path, "data\\rawdata\\data_preprocessed_T.pkl", sep=""))
df = bind_rows(df,R_read_pickle_file(paste(data_path, "data\\rawdata\\data_preprocessed_H.pkl", sep="")))
df = bind_rows(df,R_read_pickle_file(paste(data_path, "data\\rawdata\\data_preprocessed_W.pkl", sep="")))
df = bind_rows(df,R_read_pickle_file(paste(data_path, "data\\rawdata\\data_preprocessed_ST.pkl", sep="")))
df = df %>% 
  select(matches("(raw|norm|Count|Error|Errors|Timeout|Uncorrectable|Rate|RUL|serial_number)$"))
for (col in names(df[, !names(df) %in% c("Days_In_Service_norm", "RUL", "serial_number")])){
  print(col)
  p1 = ggplot(data=df, aes_string(x="RUL", y=col, group="serial_number")) +
    ylim(NA, quantile(df[, col], probs = 0.999, na.rm=TRUE)) +
    geom_line(aes(color=serial_number)) + scale_x_reverse() + 
    theme(legend.position="none",   axis.text=element_text(size=20), axis.title=element_text(size=20))
  ggsave(paste(data_path, "data\\rawdata\\plot_col_", col, ".pdf", sep=""), p1, width=11, height=3)
}
### EOF Plot All Trajectories ###

### SOF Save Predictions to feed into Jupyter ###
require(reticulate)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
rfr = readRDS(paste(data_path, "data\\rawdata\\rfsrc_16.rds", sep=""))
test = R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_ST8000NM0055.pkl", sep=""))
y_pred = predict(rfr$rf,test)$predicted
df = data.frame(test$serial_number, y_pred, test$RUL)
write.csv(as.data.frame(df),
          file=paste(data_path, "data\\rawdata\\predictions_ST8000NM0055.csv", sep=""))
### EOF Save Predictions to feed into Jupyter ###

### SOF How long to test hard drives live? ###
require(dplyr)
models = c('WDC WD800AAJS', 'HGST HMS5C4040ALE640', 'HGST HMS5C4040BLE640', 'Hitachi HDS5C3030ALA630',
           'Hitachi HDS5C4040ALE630', 'Hitachi HDS722020ALA330', 'Hitachi HDS723030ALA640',
           'ST320LT007', 'ST500LM012 HN', 'ST500LM030', 'ST4000DM001', 'ST4000DX000',
           'ST6000DX000', 'ST8000DM002', 'ST8000NM0055', 'ST10000NM0086', 'ST12000NM0007',
           'ST3160318AS', 'WDC WD30EFRX', 'WDC WD60EFRX')
models = c('T', 'H', 'W')
model = 'T'
for (model in models){
  df = R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_", model, ".pkl", sep=""))
  df = df %>% 
    group_by(serial_number) %>% 
    summarise(n = n())
  print(paste("Mean for model ", model, ": ", mean(df$n), sep = ""))
  print(paste("Standard deviation for model ", model, ": ", sd(df$n), sep = ""))
}
### SOF How long to test hard drives live? ###

### SOF Sensitivity Analysis ###
require(reticulate)
require(randomForestSRC)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
models = c('T')
model = 'T'
for (model in models){
  rfr = readRDS(paste(data_path, "data\\rawdata\\rfsrc_23.rds", sep=""))
  test = R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_T.pkl", sep=""))
  vimp.order = read.csv(file=paste(data_path, "data\\rawdata\\vimp_", model,".csv", sep=""))
  # Select top n variables
  vimp.top = as.character(vimp.order[1:30, 1])
  vimp.top = c(vimp.top, "RUL")
  
  y_pred = predict(rfr$rf, test[, names(test) %in% vimp.top])$predicted
  df = data.frame(test$serial_number, test[, names(test) %in% vimp.top], y_pred)
  df_dist = data.frame(test$serial_number, test[, names(test) %in% vimp.top], y_pred)
  for (col in 1:dim(df)[2]){
    for (row in 2:dim(df)[1]){
        df_dist[row,col] = sqrt((df[row, col] - df[row-1, col])^2)
      }
  }
  df_dist = df_dist[-1,]
  df_dist$test.serial_number = NULL
  df_dist$RUL = NULL
  dist_means = data.frame(rowMeans(df_dist[,-which(names(df_dist) == "y_pred")]))
  df_sens = df_dist$y_pred / dist_means
  mean(df_sens$rowMeans.df_dist....which.names.df_dist......y_pred....)
}
for (model in models){
  rfr = readRDS(paste(data_path, "data\\rawdata\\rfsrc_23.rds", sep=""))
  test = R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_T.pkl", sep=""))
  vimp.order = read.csv(file=paste(data_path, "data\\rawdata\\vimp_", model,".csv", sep=""))
  # Select top n variables
  vimp.top = as.character(vimp.order[1:30, 1])
  vimp.top = c(vimp.top, "RUL")
  
  y_pred = predict(rfr$rf, test[, names(test) %in% vimp.top])$predicted
  df = data.frame(test$serial_number, test[, names(test) %in% vimp.top], y_pred)
  df_dist = data.frame(test$serial_number, test[, names(test) %in% vimp.top], y_pred)
  for (col in 1:dim(df)[2]){
    for (row in 2:dim(df)[1]){
      df_dist[row,col] = sqrt((df[row, col] - df[row-1, col])^2)
    }
  }
  df_dist = df_dist[-1,]
  df_dist$test.serial_number = NULL
  df_dist$RUL = NULL
  dist_means = data.frame(rowMeans(df_dist[,-which(names(df_dist) == "y_pred")]))
  df_sens = df_dist$y_pred / dist_means
  mean(df_sens$rowMeans.df_dist....which.names.df_dist......y_pred....)
}

