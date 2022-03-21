## **BTC Analytics Dashboard** 

### **Abstract**
This dashboard will provide real-time daily updates to Bitcoin pricing and on-chain transaction metrics from data sourced through various API's. It will also constantly update an LSTM prediction model to predict price movements

### **Design**
Download various transaction information daily from the [blockchain.com API](https://www.blockchain.com/api) and price information from [CoinbasePro API](https://developers.coinbase.com/) and store it in a MongoDB database. The dashboard program will query this information and update the prediction model. The Steamlit application will display this information and some visualizations for the user to interact it.  I also plan to incorporate news feeds and other sources of information if possible.

### **Data**
The following information will be captured:
BTC Prices
Block Size
Average Block Size
Transactions per Block
Total Transactions
Median Confirmation Time
Average Confirmation Time

### **Algorithms/Models**
An LSTM Model will be used on the historical pricing information to predict future prices

Model Peformance
MSE: .388 
RMSE: .623
MAE: .272 
MAPE: 370.017
MSLE: .047  

### **Tools**
MongoDB for data storage
Numpy and Pandas for data manipulation
Scikit-learn and Keras for modeling
Atlair for plotting
Steamlit for deployment