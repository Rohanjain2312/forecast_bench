Having worked in the finance sector, I gravitate towards practical applications of AI in time series forecasting. In my latest project, I asked one simple question: can a pretrained AI foundation model (Chronos-2) beat classical statistics (ARIMA, HAR) at forecasting financial data?

Before running any model, I wrote down what "losing" would mean and committed it to git, so I couldn't quietly change the definition of success after seeing the results.

The AI model lost, by that rule. Classical statistics still won on a fair, data-rich comparison.

The more useful finding: cutting the training data to one year barely hurt the foundation model, while a neural network trained from scratch fell apart completely. Pretraining didn't add accuracy. It removed the need for much data.

Writeup, live demo, and code below if you want the numbers.

Writeup: https://medium.com/@rohanjain2312/testing-an-ai-forecasting-model-against-classical-statistics-037878ae00fd

Live demo: https://huggingface.co/spaces/rohanjain2312/forecastbench-demo

Code: https://github.com/Rohanjain2312/forecast_bench

#MachineLearning #DataScience #AI #Finance
