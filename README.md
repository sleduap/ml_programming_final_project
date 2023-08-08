# The links for the final project

Github Link for the code -> https://github.com/zshicode/Attention-CLX-stock-prediction


Link for the research paper -> https://arxiv.org/pdf/2204.02623.pdf


Summary of paper implementation:
The stock data undergoes initial preprocessing using an ARIMA(p=2,q=0,d=1) model. Subsequently, the preprocessed stock sequence is fed into either neural networks (NN) or the XGBoost algorithm. This marks the beginning of a pretraining-finetuning framework.

The pre-training stage involves an innovative Attention-based CNN-LSTM model operating within a sequence-to-sequence framework. The encoder component employs an Attention-based CNN to extract intricate features from the raw stock data. On the other hand, the decoder utilizes a Bidirectional LSTM to capture long-term time series features.

Following the pre-training phase, a fine-tuning step ensues. This entails the integration of an XGBoost model, which takes advantage of its ability to leverage information spanning multiple time periods. This fine-tuning process enhances the model's grasp of the intricacies of the stock market.

The amalgamation of these techniques culminates in a hybrid model named by the author as "AttCLX." Empirical findings indicate the model's superior efficacy and relatively high prediction accuracy. This predictive power offers valuable support to investors and institutions, facilitating well-informed decision-making and the pursuit of enhanced returns while mitigating risks.






