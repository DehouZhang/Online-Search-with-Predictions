# Online Algorithm with Predictions for Trading Problems
This is the implementation about our Best-Price prediction and Query-Based prediction algorithms for the Online Search problem. And Also the Query-Based prediction algorithms for the Interval Search problem

We implement `ORA`, `ROBUST-MIX`, `RLIS` and `LIS` algorithms. 

## Experiment Data
The experiment data can be download from https://eatradingacademy.com/software/forex-historical-data/.

We included `ETHUSD`, `BTCUSD`, `CADJPY`, `EURUSD`, `GBPUSD` and `AUDCHF` as experment data in `Data` folder.

All dataset have same entries: [`Time`, `Open`, `High`, `Low`, `Close`, `Volumn`].

[`Close`] was used in all experiments.

## Prerequisites
* Python 3.8 
* Numpy >= 1.19.2 
* matploylib >= 3.3.2 
* pandas >= 1.1.5

## Reproducing Results
* `BTC-to-USD`
  1. run `python query_based_prediction.py BTCUSD`
  2. run `python best_price_prediction.py BTCUSD`
  3. run `python interval_search.py BTCUSD`

* `Ethereum-to-USD`
  1. run `python query_based_prediction.py ETHUSD`
  2. run `python best_price_prediction.py ETHUSD`
  3. 3. run `python interval_search.py ETHUSD`

* `CAD-to-Yen`
  1. run `python query_based_prediction.py CADJPY`
  2. run `python best_price_prediction.py CADJPY`
  3. run `python interval_search.py CADJPY`

* `Euro-to-USD`
  1. run `python query_based_prediction.py EURUSD`
  2. run `python best_price_prediction.py EURUSD`
  3. run `python interval_search.py EURUSD`

* `GBP-to-USD`
  1. run `python query_based_prediction.py GBPUSD`
  2. run `python best_price_prediction.py GBPUSD`
  3. run `python interval_search.py GBPUSD`

* `AUD-to-CHF`
  1. run `python query_based_prediction.py AUDCHF`
  2. run `python best_price_prediction.py AUDCHF`
  3. run `python interval_search.py AUDCHF`

## Experiment Result:
The figures and csv files can be found in `Experiment_result` folder.

