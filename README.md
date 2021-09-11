# Online Search with Best-Price and Query-Based Predictions
This is the implementation about our Best-Price prediction and Query-Based prediction algorithms for the Online Search problem.

We implement `ORA`, `Robust`, `RLIS` and `RLIS-H` algorithms. 

## Experiment Data
The experiment data can be download from https://eatradingacademy.com/software/forex-historical-data/.

We included `ETHUSD`, `BTCUSD`, `CADJPY` and `EURUSD` as experment data in `Data` folder.

## Prerequisites
* Python 3.8 
* Numpy >= 1.19.2 
* matploylib >= 3.3.2 
* pandas >= 1.1.5

## Reproducing Results
* `BTC-to-USD`
  1. run `python query_based_prediction.py BTCUSD`
  2. run `python best_price_prediction.py BTCUSD`

* `Ethereum-to-USD`
  1. run `python query_based_prediction.py ETHUSD`
  2. run `python best_price_prediction.py ETHUSD`

* `Yen-to-CAD`
  1. run `python query_based_prediction.py CADJPY`
  2. run `python best_price_prediction.py CADJPY`

* `Euro-to-USD`
  1. run `python query_based_prediction.py EURUSD`
  2. run `python best_price_prediction.py EURUSD`

## Experiment Result:
The figures and csv files can be found in `Experiment_result` folder.

