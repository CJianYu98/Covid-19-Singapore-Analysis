library(ggplot2)

cases_sent <- read.csv("/Users/chenjianyu/Library/Mobile Documents/com~apple~CloudDocs/SMU/SMU Module Materials/Y2S2/SMT203 Computational Social Sci/Covid-19-Singapore-Analysis/Data/Covid-19_cases_sentiments.csv")
names(cases_sent)

plot <- ggplot(data=cases_sent, aes(x=daily_changes)) +
  geom_histogram(binwidth = 100)
plot

SMA1 <- cases_sent$SMA1_ch
SMA3 <- cases_sent$SMA3_ch
SMA5 <- cases_sent$SMA5_ch
SMA7 <- cases_sent$SMA7_ch
SMA14 <- cases_sent$SMA14_ch
daily_changes <- cases_sent$Daily.Changes
daily_confirmed <- cases_sent$Daily.Confirmed

reg.fit1 <- lm(SMA1~daily_changes)
summary(reg.fit1)
plot(daily_changes, SMA1, main="Linear relationship between Daily Changes in Confirmed Cases and Change in 1-day Sentiment Moving Average", 
     xlab="Daily Changes", ylab="Sentiment")
abline(reg.fit1, lwd=3, col="red")
plot(daily_changes, residuals(reg.fit1), main="Relationship between Daily Changes in Confirmed Cases and Residuals", 
     xlab="Daily Changes in Confirmed Cases", ylab="Residuals")


reg.fit3 <- lm(SMA3~daily_changes)
summary(reg.fit3)
plot(daily_changes, SMA3, main="Linear relationship between 
     Daily Changes in Confirmed Cases and 3-days Sentiment Moving Average", 
     xlab="Daily Changes", ylab="Sentiment")
abline(reg.fit3, lwd=3, col="red")
resid3 <- residuals(reg.fit3)
plot(daily_changes, resid3, main="Relationship between 
     Daily Changes in Confirmed Cases and Residuals", 
     xlab="Daily Changes in Confirmed Cases", ylab="Residuals")


reg.fit5 <- lm(daily_changes~SMA5)
summary(reg.fit5)
plot(daily_changes, SMA5, main="Linear relationship between 
     Daily Changes in Confirmed Cases and 3-days Sentiment Moving Average", 
     xlab="Daily Changes", ylab="Sentiment")
abline(reg.fit5, lwd=3, col="red")
resid5 <- residuals(reg.fit5)
plot(daily_changes, resid5, main="Relationship between 
     Daily Changes in Confirmed Cases and Residuals", 
     xlab="Daily Changes in Confirmed Cases", ylab="Residuals")


reg.fit7 <- lm(daily_changes~SMA7)
summary(reg.fit7)
plot(daily_changes, SMA7, main="Linear relationship between 
     Daily Changes in Confirmed Cases and 3-days Sentiment Moving Average", 
     xlab="Daily Changes", ylab="Sentiment")
abline(reg.fit7, lwd=3, col="red")
resid7 <- residuals(reg.fit7)
plot(daily_changes, resid7, main="Relationship between 
     Daily Changes in Confirmed Cases and Residuals", 
     xlab="Daily Changes in Confirmed Cases", ylab="Residuals")


reg.fit14 <- lm(daily_changes~SMA14)
summary(reg.fit14)