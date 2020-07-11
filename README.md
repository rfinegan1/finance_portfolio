# MachineLearningFixedIncome
Machine learning techniques on Fixed Income ETFs. I used SVM, random forest classifier and logistic regression to predict tomorrow's (next trading day) adjusted close stock price. The logistic regressor had the highest confidence score. After this, I made a screener based on a couple of studies I recently read. Equities that have a high average analyst recommendation usually outperform the market. Also, a study done by AQR found that securities that had a positive yearly return will usually have another positive year. So, the screener consists of 412 fixed income ETFs that were screened with having positive annual returns in 2019 (2020 was left out for bear / unique macroeconomic news / circumstances). Those that had a positive return for the year of 2019 were then screened on their analyst recommendations of having an A- or higher. At the end, an RSI (14) signal was the last sort for the screener. The ETFs that had the lowest RSI were classified as a possible buy / long signal while the ETFs with the highest RSI were classified as a possible sell / short signal. An example was provided to show how to plug these new fixed income ETF tickers into the machine learning models. 
