### Globals ###
set.seed(123)
memory.limit(100000)

### SOF Plot the trajectories of all criticial measurements ###
require(reticulate)
require(ggplot2)
require(gridExtra)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
df.test <- R_read_pickle_file(paste(data_path, "data\\rawdata\\data_preprocessed_ST.pkl", sep=""))
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
models = c('WDC WD800AAJS', 'HGST HMS5C4040ALE640', 'HGST HMS5C4040BLE640', 'Hitachi HDS5C3030ALA630',
           'Hitachi HDS5C4040ALE630', 'Hitachi HDS722020ALA330', 'Hitachi HDS723030ALA640',
           'ST320LT007', 'ST500LM012 HN', 'ST500LM030', 'ST4000DM001', 'ST4000DX000',
           'ST6000DX000', 'ST8000DM002', 'ST8000NM0055', 'ST10000NM0086', 'ST12000NM0007',
           'ST3160318AS', 'WDC WD30EFRX', 'ST4000DM000', 'TOSHIBA MQ01ABF050')
models = c('T', 'H', 'W', 'ST')
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
for (model in models){
  df = R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_", model, ".pkl", sep=""))
  df = df %>% 
  select(matches("(raw|norm|Count|Error|Errors|Timeout|Uncorrectable|Rate|RUL)$"))
  
  mad = sapply(df[, !names(df) %in% c("date", "failure", "capacity_bytes",
                                                 "serial_number", "model", "RUL")], mad, na.rm=TRUE)
  median = sapply(df[, !names(df) %in% c("date", "failure", "capacity_bytes",
                                                "serial_number", "model", "RUL")], median, na.rm=TRUE)
  #mean = sapply(df[df$RUL ==i, !names(df) %in% c("date", "failure", "capacity_bytes",
  #                                                 "serial_number", "model", "RUL")], mean, na.rm=TRUE)
  #sd = sapply(df[df$RUL ==i, !names(df) %in% c("date", "failure", "capacity_bytes",
  #                                               "serial_number", "model", "RUL")], sd, na.rm=TRUE)
  #cov2 = mean/sd
  cov2 = mad / median
  
  cov2[!is.finite(cov2)] <- 0
  cov2 = data.frame(cov2)
  cov2$measure = row.names(cov2)
  #ggplot(aes(x = measure, y = cov2, fill = measure), data = cov2[cov2$cov2 != 0,]) +
  #  geom_bar(stat = 'identity') +
  #  labs(x='Measurement', y='MAD/Median Ratio') +
  #  theme(axis.text.x = element_text(angle = 90),
  #        axis.text=element_text(size=20), 
  #        axis.title=element_text(size=20),
  #        legend.position = "none")
  print(model)
  print(mean(cov2$cov2))
}
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
    theme(legend.position="none",   axis.text=element_text(size=20), axis.title=element_text(size=17))
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
#models = c('WDC WD800AAJS')
models = c(#'WDC WD800AAJS',
           #'HGST HMS5C4040ALE640',
           #'HGST HMS5C4040BLE640',
           #'Hitachi HDS5C3030ALA630',
           #'Hitachi HDS5C4040ALE630',
           #'Hitachi HDS722020ALA330',
           #'Hitachi HDS723030ALA640',
           #'ST320LT007',
           #'ST500LM012 HN',
           #'ST500LM030',
           #'ST4000DM001',
           'ST4000DX000',
           'ST6000DX000',
           'ST8000DM002',
           'ST8000NM0055',
           'ST10000NM0086',
           'ST12000NM0007',
           'ST3160318AS',
           'WDC WD30EFRX',
           'WDC WD60EFRX',
           'TOSHIBA MQ01ABF050',
           'ST4000DM000',
           'T',
           'H',
           'W',
           'ST')
for (model in models){
  print(paste("Load model ", model, sep=""))
  rfr = readRDS(paste(data_path, "data\\rawdata\\rfsrc_", model, ".rds", sep=""))
  print("Load test data")
  test = R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_", model, ".pkl", sep=""))
  print("Load vimp")
  vimp.order = read.csv(file=paste(data_path, "data\\rawdata\\vimp_", model,".csv", sep=""))
  # Select top n variables
  vimp.top = as.character(vimp.order[1:30, 1])
  vimp.top = c(vimp.top, "RUL")
  print("Predict")
  y_pred = predict(rfr$rf, test[, names(test) %in% vimp.top])$predicted
  df = data.frame(test$serial_number, test[, names(test) %in% vimp.top], error = y_pred - test$RUL)
  print("Calculate sensibility")
  df_dist = df
  for (col in 1:dim(df)[2]){
    for (row in 2:dim(df)[1]){
        df_dist[row,col] = sqrt((df[row, col] - df[row-1, col])^2)
      }
  }
  df_dist = df_dist[-1,]
  df_dist$test.serial_number = NULL
  df_dist$RUL = NULL
  #df_sens = df_sens[-12685,]
  dist_means = data.frame(rowMeans(df_dist[,-which(names(df_dist) == "error")]))
  df_sens = df_dist$error / dist_means
  print(mean(df_sens$rowMeans.df_dist....which.names.df_dist......error...., na.rm = TRUE))
}
### EOF Sensitivity Analysis ###

### SOF Cox Sensitivity Analysis ###
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
models = c('W',
           'ST4000DM000',
           'ST')
for (model in models){
  print(paste("Start model ", model, sep=""))
  df = read.csv(paste(data_path, "data\\rawdata\\predictions_cox_", model, ".csv", sep=""))
  df$error = df$pred - df$RUL
  df$pred = NULL
  df$RUL = NULL
  df$X0.9 = NULL
  df$X = NULL
  df$date = NULL
  df$failure = NULL
  df$serial_number = NULL
  df$Days_In_Service = NULL
  df$model = NULL
  df$capacity_bytes = NULL
  print("Calculate sensibility")
  df_dist = df
  for (col in 1:dim(df)[2]){
    for (row in 2:dim(df)[1]){
      df_dist[row,col] = sqrt((df[row, col] - df[row-1, col])^2)
    }
  }
  df_dist = df_dist[-1,]
  dist_means = data.frame(rowMeans(df_dist[,-which(names(df_dist) == "error")]))
  df_sens = df_dist$error / dist_means
  print(mean(df_sens$rowMeans.df_dist....which.names.df_dist......error...., na.rm = TRUE))
}
### EOF Cox Sensitivity Analysis ###
  
### Plot Predictions of Cox ###
# Plot results
require(ggplot2)
library(gridExtra)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
models = c('H', 'HGST HMS5C4040ALE640', 'HGST HMS5C4040BLE640', 'Hitachi HDS5C3030ALA630',
           'Hitachi HDS722020ALA330', 'Hitachi HDS723030ALA640', 'ST320LT007', 'ST500LM012 HN',
           'ST500LM030', 'ST4000DM000', 'ST4000DX000', 'ST6000DX000', 'ST8000DM002', 'ST8000NM0055',
           'ST10000NM0086', 'ST12000NM0007', 'ST3160318AS', 'T', 'TOSHIBA MQ01ABF050', 
           'W', 'WDC WD30EFRX', 'WDC WD60EFRX', 'WDC WD800AAJS', 'ST8000NM0055')

model = 'WDC WD800AAJS'
for (model in models){
  df = read.csv(paste(data_path, "data\\rawdata\\predictions_cox_", model, ".csv", sep=""))
  df$pred_error = df$pred - df$RUL
  pdf(paste(data_path, "data\\rawdata\\plot_cox_", model, ".pdf", sep=""))
  p1 = ggplot(data=df, aes(x=RUL, y=RUL, group=serial_number)) +
    geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
  p2 = ggplot(data=df, aes(x=RUL, y=pred, group=serial_number)) +
    geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
  p3 = ggplot(data=df, aes(x=RUL, y=pred_error, group=serial_number)) +
    geom_line(aes(color=serial_number)) + scale_x_reverse() + theme(legend.position="none")
  grid.arrange(p1, p2, p3, nrow = 3, top = model)
  dev.off()
}
### EOF Plot Predictions of Cox ###

### SOF Summary of Critical SMART Values ###
require(reticulate)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
df = R_read_pickle_file(paste(data_path, "data\\rawdata\\data_raw.pkl", sep=""))
df_crit = df[,c("smart_5_raw", "smart_10_raw", "smart_184_raw","smart_187_raw",
                "smart_188_raw", "smart_196_raw","smart_197_raw", "smart_198_raw", "failure")]
df_crit_fail = df_crit[ which(df_crit$failure=='1'),
                        c("smart_5_raw", "smart_10_raw", "smart_184_raw","smart_187_raw",
                          "smart_188_raw", "smart_196_raw","smart_197_raw", "smart_198_raw")]
df_crit_nofail = df_crit[ which(df_crit$failure=='0'),
                        c("smart_5_raw", "smart_10_raw", "smart_184_raw","smart_187_raw",
                          "smart_188_raw", "smart_196_raw","smart_197_raw", "smart_198_raw")]
df_days_fail = df[ which(df_crit$failure=='1'),
                        c("smart_9_raw")]
df_days_fail = data.frame(df_days_fail)

options(digits=3)
summary(df_days_fail, na.rm = TRUE)
sapply(df_crit_nofail, max, na.rm = TRUE)
head(df_crit)

ggplot(df_days_fail, aes(x=df_days_fail)) +
  geom_histogram(aes(y=..density..), colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666") +
  labs(x="Run to Failure Times", y = "Count")

library("dplyr")
test = df %>% group_by(model, serial_number) %>% summarize(count=n())
### EOF Summary of Critical SMART Values ###

rm(list=ls())
### SOF Summary of Dataset ###
library(lubridate)
library(ggplot2)
# hdds per model
df$manu = substr(df$model,1,2)
hdds.per.model = df %>% group_by(manu, model) %>% summarize(HDDs=n_distinct(serial_number))
hdds.per.model = hdds.per.model[order(-hdds.per.model$HDDs),]
write.csv(hdds.per.model, file = paste(data_path, "data\\rawdata\\models_manu.csv", sep=""))
hdds.per.model$model <- factor(hdds.per.model$model, levels = hdds.per.model$model[order(-hdds.per.model$HDDs)])
ggplot(data=hdds.per.model, aes(x=model, y=HDDs, fill=model)) +
  geom_bar(width = 1, stat="identity") + 
  theme(legend.position="none") +
  ylab("Number of HDDs") +
  xlab("Model")
# Obs per month
df$month = paste(month(df$date, label = TRUE), year(df$date))
df$month_num = paste(year(df$date), substr(df$date, 6, 7), sep="")
obs.per.month = df %>% group_by(month_num, month, model) %>% summarize(Observations=n_distinct(serial_number))
obs.per.month$month <- factor(obs.per.month$month, levels = obs.per.month$month[order(obs.per.month$month_num)])
ggplot(data=obs.per.month,
       aes(x = month_num, y=Observations, fill=model)) +
  geom_bar(width = 1, stat="identity") + 
  theme(legend.position="none") +
  ylab("Number of HDDs") +
  ylim(0, 3100) +
  xlab("Months") +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
### SOF Summary of Dataset ###

### Plot Elbow Curve ###
library(dplyr)
new_df = df %>% 
  distinct(serial_number) %>% 
  sample_n(1, replace = FALSE) %>% 
  inner_join(df, .)
new_df$smart_9_raw = new_df$smart_9_raw/24
p1 = ggplot(data=new_df, aes(x=smart_9_raw, y=smart_5_raw, group=serial_number)) +
  geom_line(aes(color=serial_number)) +
  theme(legend.position="none", axis.text=element_text(size=20),
        axis.title=element_text(size=17)) +
  geom_vline(xintercept=125.5) +
  annotate(geom="text", x=60, y=170, label="Healthy", size=7) +
  annotate(geom="text", x=180, y=170, label="Degradation", size=7)
ggsave(paste(data_path, "data\\rawdata\\plot_smart_5_raw_ZCH072P4.pdf", sep=""), p1, width=11, height=3)


p1 = ggplot(data=new_df, aes_string(x="RUL", y=col, group="serial_number")) +
  ylim(NA, quantile(df[, col], probs = 0.999, na.rm=TRUE)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() + 
  theme(legend.position="none",   axis.text=element_text(size=20), axis.title=element_text(size=17))


### Calculate total VIMPs TODO!###
models = c( 'HGST HMS5C4040ALE640', 'HGST HMS5C4040BLE640', 'Hitachi HDS5C3030ALA630',
            'Hitachi HDS722020ALA330', 'Hitachi HDS723030ALA640', 'ST320LT007', 'ST500LM012 HN',
            'ST500LM030', 'ST4000DM000', 'ST4000DX000', 'ST6000DX000', 'ST8000DM002', 'ST8000NM0055',
            'ST10000NM0086', 'ST12000NM0007', 'ST3160318AS', 'TOSHIBA MQ01ABF050', 
            'WDC WD30EFRX', 'WDC WD60EFRX', 'WDC WD800AAJS', 'ST8000NM0055')

### Plot All Predictions ###
# Plot results
require(ggplot2)
library(gridExtra)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"

model1 = 'ST320LT007'
model2 = 'WDC WD800AAJS'
dfcox1 = read.csv(paste(data_path, "data\\rawdata\\predictions_cox_", model1, ".csv", sep=""))
dfcox1$pred_error = dfcox1$pred - dfcox1$RUL
dfcox2 = read.csv(paste(data_path, "data\\rawdata\\predictions_cox_", model2, ".csv", sep=""))
dfcox2$pred_error = dfcox2$pred - dfcox2$RUL
#pdf(paste(data_path, "data\\rawdata\\plot_cox_", model, ".pdf", sep=""))
p2 = ggplot(data=dfcox1, aes(x=RUL, y=pred, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() +
  geom_abline(aes(intercept = 0, slope = -1, lty="Ground Truth")) +
  xlab("True RUL") +
  ylab("Predicted RUL") +
  scale_linetype(name = NULL) +
  ggtitle("PrHM\n(Jardine et al. 2001)") +
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        plot.title = element_text(hjust = 0.5),
        legend.key.width=unit(1.8,"cm"),
        axis.text=element_text(size=14),
        axis.title=element_text(size=14))
p4 = ggplot(data=dfcox2, aes(x=RUL, y=pred, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() +
  geom_abline(aes(intercept = 0, slope = -1, lty="Ground Truth")) +
  xlab("True RUL") +
  ylab("Predicted RUL") +
  scale_linetype(name = NULL) +
  theme(axis.title.y=element_blank(),
        axis.text=element_text(size=14),
        axis.title=element_text(size=14))


### RSF ###
require(randomForestSRC)
require(reticulate)
data_path = "C:\\Users\\kwesendrup\\PycharmProjects\\master\\backblaze\\"
source_python(paste(data_path, "load_from_r.py", sep=""))
test1 <- R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_", model1, ".pkl", sep=""))
test2 <- R_read_pickle_file(paste(data_path, "data\\rawdata\\data_test_", model2, ".pkl", sep=""))
rfr1 = readRDS(paste(data_path, "data\\rawdata\\rfsrc_", model1, ".rds", sep=""))
rfr2 = readRDS(paste(data_path, "data\\rawdata\\rfsrc_", model2, ".rds", sep=""))
vimp.order1 = read.csv(file=paste(data_path, "data\\rawdata\\vimp_", model1,".csv", sep=""))
vimp.top1 = as.character(vimp.order1[1:30, 1])
vimp.top1 = c(vimp.top1, "RUL")
vimp.order2 = read.csv(file=paste(data_path, "data\\rawdata\\vimp_", model2,".csv", sep=""))
vimp.top2 = as.character(vimp.order2[1:30, 1])
vimp.top2 = c(vimp.top2, "RUL")
y_pred1 = predict(rfr1$rf,test1[, names(test1) %in% vimp.top1])$predicted
y_pred2 = predict(rfr2$rf,test2[, names(test2) %in% vimp.top2])$predicted
y_test1 = test1$RUL
y_test2 = test2$RUL
df1 = data.frame(serial_number = test1$serial_number, y_pred1, y_test1)
df2 = data.frame(serial_number = test2$serial_number, y_pred2, y_test2)
df1 = df1[df1$serial_number!='W0Q6KWV8',]
p1 = ggplot(data=df1, aes(x=y_test1, y=y_pred1, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() +
  geom_abline(aes(intercept = 0, slope = -1, lty="Ground Truth")) +
  xlab("True RUL") +
  ylab("Predicted RUL") +
  scale_linetype(name = NULL) +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5),
        axis.title.x=element_blank(),
        axis.text=element_text(size=14),
        axis.title=element_text(size=14)) +
  ggtitle("RSF\n(Frisk et al. 2014)")
p3 = ggplot(data=df2, aes(x=y_test2, y=y_pred2, group=serial_number)) +
  geom_line(aes(color=serial_number)) + scale_x_reverse() +
  geom_abline(aes(intercept = 0, slope = -1, lty="Ground Truth")) +
  xlab("True RUL") +
  ylab("Predicted RUL") +
  scale_linetype(name = NULL) +
  theme(legend.position="none",
        axis.text=element_text(size=14),
        axis.title=element_text(size=14))

lay <- rbind(c(1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2),
            c(3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4))
#lay <- rbind(c(1,2),
#            c(3,4))
grid.arrange(p1, p2, p3, p4, nrow = 2, layout_matrix = lay)
#dev.off()
#df = df[df$serial_number=='WD-WMAV3J877106',]

### EOF Plot All Predictions ###
