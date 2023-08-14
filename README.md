# The links for the final project

Requirements
The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:
```
numpy
sklearn
statsmodels
pandas
tensorflow
keras
xgboost
RandomForest
```

Github Link for the code -> https://github.com/zshicode/Attention-CLX-stock-prediction


Link for the research paper -> https://arxiv.org/pdf/2204.02623.pdf


## Summary of paper implementation:

The stock data undergoes initial preprocessing using an ARIMA(p=2,q=0,d=1) model. Subsequently, the preprocessed stock sequence is fed into either neural networks (NN) or the XGBoost algorithm. This marks the beginning of a pretraining-finetuning framework.

The pre-training stage involves an innovative Attention-based CNN-LSTM model operating within a sequence-to-sequence framework. The encoder component employs an Attention-based CNN to extract intricate features from the raw stock data. On the other hand, the decoder utilizes a Bidirectional LSTM to capture long-term time series features.

Following the pre-training phase, a fine-tuning step ensues. This entails the integration of an XGBoost model, which takes advantage of its ability to leverage information spanning multiple time periods. This fine-tuning process enhances the model's grasp of the intricacies of the stock market.

The amalgamation of these techniques culminates in a hybrid model named by the author as "AttCLX." Empirical findings indicate the model's superior efficacy and relatively high prediction accuracy. This predictive power offers valuable support to investors and institutions, facilitating well-informed decision-making and the pursuit of enhanced returns while mitigating risks.

Summary of contribution:

We have used the Attention-based CNN-LSTM and random forest hybrid model and compared the performance of our model to the performance of paper model (i.e. Attention-based CNN-LSTM and XGBoost hybrid model) by using evaluation metrics such as mean square error (MSE), root mean square error (RMSE), mean absolute error (MAE), and r squared (R2).

Our hybrid model has demonstrated superior performance compared to the paper's model in predicting stock prices, as indicated by the lower MSE, RMSE, MAE, and higher R2 values. This suggests that the integration of a Random Forest component into the hybrid architecture contributes positively to the prediction accuracy.

## Citation
```
@article{shi2022attclx,
    author={Zhuangwei Shi and Yang Hu and Guangliang Mo and Jian Wu},
    title={Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction},
    journal={arXiv preprint arXiv:2204.02623},
    year={2022},
}
```



