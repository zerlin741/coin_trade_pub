import collections
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import statsmodels.api as sm
from sklearn import ensemble, preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from Coin.Strategy.util.logic import gen_x_feature_multi
from Strategy.util.query_base import get_kline_util
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')



OKEX_BTC_1H_KLINE = '/home/zelinchen/Private/okex_btc_usdt_kline_1h.csv'


def read_from_local_csv(market_type, exchange, symbol, period):
	kline_df = pd.read_csv(OKEX_BTC_1H_KLINE)
	kline_df = kline_df[(kline_df['market_type']==market_type)
	                    &(kline_df['exchange']==exchange)
	                    &(kline_df['symbol']==symbol)
	                    &(kline_df['period']==period)]
	result = []
	cols = ['trading_date', 'exchange', 'symbol', 'period', 'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']
	for idx in kline_df.index:
		kline_dict = json.loads(kline_df.loc[idx, 'kline_dict'])
		trading_date = kline_df.loc[idx, 'trading_date']
		for kline in kline_dict['klines']:
			result.append([trading_date, exchange, symbol, period, int(kline['kline_timestamp'])/1e+9, datetime.datetime.fromtimestamp(int(kline['kline_timestamp'])/1e+9),
			             kline['open'], kline['high'], kline['low'], kline['close'], kline['volume']])
	result_df = pd.DataFrame(result, columns=cols)
	result_df.sort_values('timestamp', inplace=True)
	result_df.index = range(len(result_df))
	return result_df

# kline_df = read_from_local_csv(market_type='Spot', exchange='Okex', symbol='BTC-USDT', period='1h')
kline_df = get_kline_util('Spot', 'Okex', 'BTC-USDT', '1h')

kline_df['y'] = kline_df['high']/kline_df['low']-1
kline_df['pct'] = kline_df['close']/kline_df['open']-1
tradeable_fetch_logic1 = (kline_df['high'] - kline_df['low']) / abs(kline_df['open'] - kline_df['close']) < 2.5
tradeable_fetch_logic2 = abs(kline_df['pct']) > abs(kline_df['pct']).mean()
kline_df['y_label'] = 0
kline_df['y_label'][kline_df['pct'] > 0] = 1
kline_df['y_label'][kline_df['pct'] < 0] = -1
kline_df['y_label'][tradeable_fetch_logic2 & (kline_df['pct'] > 0)] = 2
kline_df['y_label'][tradeable_fetch_logic2 & (kline_df['pct'] < 0)] = -2

## predict model
short_n = 2
mid_n = 20
long_n = 24
x_label = gen_x_feature_multi(kline_df, range(short_n, long_n))
y_label = kline_df['y_label']


assert len(x_label) == len(y_label)
x_label.dropna(inplace=True, axis=0, how='any')
y_label = y_label.loc[x_label.index]
y_label_index = np.array(y_label.index)
x_feature = np.array(x_label)
y_label = np.array(y_label)

train_pct = 0.95
train_num = int(len(x_feature)*train_pct)
train_x_feature = x_feature[:train_num-1]
train_y_label = y_label[1:train_num]
test_x_feature = x_feature[train_num-1:-1]
test_y_label = y_label[train_num:]
test_y_label_index = y_label_index[train_num:]
print("test length :%s" % len(test_y_label))

y_label_weight = collections.Counter(train_y_label)
y_label_weight = {k: len(train_y_label)/v for k, v in y_label_weight.items()}

def cal_ret(kline_df, test_y_label_index, y_pred):
	ret = [0]
	for idx, y in zip(test_y_label_index, y_pred):
		r = kline_df.loc[idx, 'pct'] * np.sign(y)
		last_y = y_pred[test_y_label_index.tolist().index(idx)-1]
		fee = 0.0001
		if abs(y) >= 1 and ret[-1] in (-fee, 0):
			ret.append(r-fee)
		elif last_y * y > 0 and ret[-1] not in (-fee, 0):
			ret.append(r)
		elif last_y * y < 0 and ret[-1] not in (-fee, 0):
			if abs(y) >= 1:
				ret.append(r-2*fee)
			else:
				ret.append(-fee)
		else:
			ret.append(0)
	nav = np.cumprod([1+r for r in ret])
	return nav



# random forest
from Strategy.util.logic import human_gridsearch_rm, gen_y_pred_from_proba

# clf, iTrees = human_gridsearch_rm(ensemble.RandomForestClassifier, train_x_feature, train_y_label, 0.95, range(100, 2000, 100))
# y_pred = gen_y_pred_from_proba(clf.predict_proba(test_x_feature), np.array([-2 , -1, 0, 1, 2]))
# correct = accuracy_score(test_y_label, y_pred)
# n1 = len([y for y in y_pred if abs(y) > 1]) / len(y_pred)
# p1 = len([y for i, y in enumerate(y_pred) if abs(y) > 1 and y * test_y_label[i] > 0]) / len(
# 	[y for i, y in enumerate(y_pred) if abs(y) > 1])
# p2 = len([y for i, y in enumerate(y_pred) if abs(y) >= 1 and y * test_y_label[i] > 0]) / len(
# 	[y for i, y in enumerate(y_pred) if abs(y) >= 1])
# nav = cal_ret(kline_df, test_y_label_index, y_pred)
# print("iTrees: %s,  rate: %s, precision: %s %s %s, nav: %s" % (iTrees, n1, correct, p1, p2, nav[-1]))


nav_df = pd.DataFrame()

nTreeList = range(1000, 1100, 100)
for iTrees in nTreeList:
	model = ensemble.RandomForestClassifier(n_estimators=iTrees, class_weight='balanced', oob_score=True, random_state=100, n_jobs=-1)
	model.fit(train_x_feature, train_y_label)
	y_pred = model.predict(test_x_feature)
	y_pred = gen_y_pred_from_proba(model.predict_proba(test_x_feature), np.array([-2,-1,0,1,2]))
	correct = accuracy_score(test_y_label, y_pred)
	n1 = len([y for y in y_pred if abs(y) > 1])/len(y_pred)
	p1 = len([y for i, y in enumerate(y_pred) if abs(y) > 1 and y * test_y_label[i] > 0]) / len([y for i, y in enumerate(y_pred) if abs(y) > 1])
	p2 = len([y for i, y in enumerate(y_pred) if abs(y) >= 1 and y * test_y_label[i] > 0]) / len([y for i, y in enumerate(y_pred) if abs(y) >= 1])
	nav = cal_ret(kline_df, test_y_label_index, y_pred)
	nav_df[iTrees] = nav
	print("iTrees: %s,  rate: %s, precision: %s %s %s, nav: %s" % (iTrees, n1, correct, p1, p2, nav[-1]))









