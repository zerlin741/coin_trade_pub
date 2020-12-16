import os
import datetime
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from bayes_opt import BayesianOptimization
from sklearn import ensemble
import joblib

from Strategy.util.query_base import get_kline_util
from Strategy.util.logic import gen_y_label_rm1, human_gridsearch_rm, gen_y_pred_from_proba, gen_x_feature_multi

class TradeableModel(object):
	def __init__(self, *, exchange, symbol, kline_period, params):
		self._train_kline_info = {'exchange': exchange, 'symbol': symbol, 'period': kline_period}
		self._model = None
		self._model_params = params
		self._x_feature, self._y_label, self._last_dt = self.gen_latest_x_y()

	def internal_score(self, y_true, y_pred):
		y_tradeable_pred = y_pred == 2
		if y_tradeable_pred.sum() == 0:
			return 0
		else:
			return (y_true[y_tradeable_pred]>0).sum() / y_tradeable_pred.sum()

	def gen_latest_x_y(self):
		exchange = self._train_kline_info['exchange']
		symbol = self._train_kline_info['symbol']
		period = self._train_kline_info['period']
		kline_df = get_kline_util('Spot', exchange, symbol, period, query_new=True, save_new=True)
		kline_df['pct'] = kline_df['close']/kline_df['open']-1
		short_n = self._model_params['short_n']
		long_n = self._model_params['long_n']
		x_feature = gen_x_feature_multi(kline_df, list(range(short_n, long_n)))
		y_label = gen_y_label_rm1(kline_df)
		total_train_data = pd.concat([x_feature, y_label], axis=1).iloc[:-1]
		assert not total_train_data.iloc[-1].isna().values.any() and not np.isinf(total_train_data.iloc[-1]).any(), \
			"Na/Inf in lastest raw train data, please check kline"
		total_train_data = total_train_data.replace([np.inf, -np.inf], np.NAN).dropna()
		x_feature = np.array(total_train_data)[:, :-1]
		y_label = np.array(total_train_data)[:, -1]
		return x_feature, y_label, kline_df['datetime'].iloc[-1]

	def get_latest_x_y(self):
		current_dt = datetime.datetime.utcnow()
		last_dt = datetime.datetime.strptime(self._last_dt, "%Y-%m-%d %H:%M:%S")
		if current_dt - last_dt > datetime.timedelta(hours=1):
			self._x_feature, self._y_label, self._last_dt = self.gen_latest_x_y()
		return self._x_feature, self._y_label

	def train(self, grid_search=False, load_model_path=None):
		x_feature, y_label = self.get_latest_x_y()
		searched_params = {
			'n_estimators': [400, 500],
			'max_features': [0.4, 0.8]
		}
		fixed_params = {
			'class_weight': 'balanced',
		  'random_state': np.random.RandomState(1),
			'n_jobs': -1,
		}
		if grid_search:
			clf = human_gridsearch_rm(
				ensemble.RandomForestClassifier,
				x_feature[:-1],
				y_label[1:],
				train_pct=0.99,
				searched_params=searched_params,
				fixed_params=fixed_params,
				show_detail=True
			)
			self._model = clf
		elif load_model_path is not None:
			assert os.path.exists(load_model_path), load_model_path
			self._model = joblib.load(load_model_path)
		else:
			clf = ensemble.RandomForestClassifier(
				n_estimators=400,
				max_features=0.8,
				**fixed_params
			)
			clf.fit(x_feature[:-1, :], y_label[1:])
			self._model = clf

	def save_model(self, save_path):
		joblib.dump(self._model, save_path)

	def current_predict(self):
		if self._model:
			x_feature, y_label = self.get_latest_x_y()
			y_prob = self._model.predict_proba(x_feature[-1:, :])
			y_pred = gen_y_pred_from_proba(y_prob, np.array([-2, -1, 0, 1, 2]))
			return y_pred[0]
		else:
			return None


if __name__ == '__main__':
	params = {
		'short_n': 2,
		'long_n': 24,
	}
	model = TradeableModel(exchange='Okex', symbol='BTC-USDT', kline_period='1h', params=params)
	model.train(grid_search=True)
	# model.current_predict()
	# def opt_model(short_n, mid_n, long_n):
	# 	params = {
	# 		'short_n': int(short_n),
	# 		'mid_n': int(mid_n),
	# 		'long_n': int(long_n),
	# 	}
	# 	model = TradeableModel(exchange='Okex', symbol='BTC-USDT', kline_period='1h', params=params)
	# 	try:
	# 		score = model.train(grid_search=True)
	# 	except Exception as e:
	# 		print(type(e), e, params)
	# 		score = 0
	# 	return score
	# bayes_opt = BayesianOptimization(
	# 	opt_model,
	# 	{
	# 		'short_n': (3, 7),
	# 		'mid_n': (14, 20),
	# 		'long_n': (14, 30)
	# 	}
	# )
	# bayes_opt.maximize(n_iter=50)

	# model.current_predict()
	# model.save_model(save_path='Data/okex_rf.joblib')
