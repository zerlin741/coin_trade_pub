import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm


def tag_y_label_by_extremum_point(df, threshold, max_lenth=24):
	df['extremum_point'] = ''
	df['extremum_point'][(df['close'] > df['close'].shift(1)) & (df['close'] > df['close'].shift(-1))] = 'max'
	df['extremum_point'][(df['close'] < df['close'].shift(1)) & (df['close'] < df['close'].shift(-1))] = 'min'
	df['inflection_point'] = None
	for i, idx in enumerate(df.index):
		if df.loc[idx, 'extremum_point'] == 'max':
			condition_max_right = df.ix[i:i+max_lenth][df.ix[i:i+max_lenth, 'close'] < df.ix[i, 'close'] * (1 - threshold)]
			condition_max_left = df.ix[i-max_lenth:i][df.ix[i-max_lenth:i, 'close'] < df.ix[i, 'close'] * (1 - threshold)]
			max_point_left = df.ix[i-max_lenth: i, 'close'].max() <= df.loc[idx, 'close']
			max_point_right = df.ix[i: i+max_lenth, 'close'].max() <= df.loc[idx, 'close']
			if not condition_max_left.empty and not condition_max_right.empty and max_point_left and max_point_right:
				df.loc[idx, 'inflection_point'] = 'start_-1'
			elif condition_max_left.empty and not condition_max_right.empty and max_point_right:
				df.loc[idx, 'inflection_point'] = 'start_-1'
			elif not condition_max_left.empty and condition_max_right.empty and max_point_left:
				df.loc[idx, 'inflection_point'] = 'start_0'
		elif df.loc[idx, 'extremum_point'] == 'min':
			condition_min_right = df.ix[i:i+max_lenth][df.ix[i:i+max_lenth, 'close'] > df.ix[i, 'close'] * (1 + threshold)]
			condition_min_left = df.ix[i-max_lenth:i][df.ix[i-max_lenth:i, 'close'] > df.ix[i, 'close'] * (1 + threshold)]
			min_point_left = df.ix[i-max_lenth: i, 'close'].min() >= df.loc[idx, 'close']
			min_point_right = df.ix[i: i+max_lenth, 'close'].min() >= df.loc[idx, 'close']
			if not condition_min_left.empty and not condition_min_right.empty and min_point_left and min_point_right:
				df.loc[idx, 'inflection_point'] = 'start_1'
			elif condition_min_left.empty and not condition_min_right.empty and min_point_right:
				df.loc[idx, 'inflection_point'] = 'start_1'
			elif not condition_min_left.empty and condition_min_right.empty and min_point_left:
				df.loc[idx, 'inflection_point'] = 'start_0'
	df['label'] = None
	for i, idx in enumerate(df.index):
		if i <= 0:
			continue
		elif df.ix[i-1, 'inflection_point'] is not None:
			df.loc[idx, 'label'] = int(df.ix[i-1, 'inflection_point'].split('_')[-1])
		elif df.ix[i-1, 'label'] is not None:
			df.loc[idx, 'label'] = df.ix[i-1, 'label']
		else:
			pass
	df['label'].fillna(-df['label'].dropna().iloc[0], inplace=True)
	return df

def _plot_x_label(df):
	from matplotlib.pylab import plt
	import datetime
	plt.scatter([datetime.datetime.fromtimestamp(x.timestamp()) for x in df['datetime'][df['label'] == 1]], df['close'][df['label'] == 1], color='red', s=1)
	plt.scatter([datetime.datetime.fromtimestamp(x.timestamp()) for x in df['datetime'][df['label'] == -1]], df['close'][df['label'] == -1], color='green', s=1)
	plt.scatter([datetime.datetime.fromtimestamp(x.timestamp()) for x in df['datetime'][df['label'] == 0]], df['close'][df['label'] == 0], color='yellow', s=1)
	plt.show()

def rsi(x):
	if pd.isna(x[x>0].mean()):
		return 0
	elif pd.isna(x[x<0].mean()):
		return 1
	else:
		return x[x > 0].mean() / (x[x > 0].mean() + abs(x[x < 0].mean()))

def simple_OLS(x, y, offset=False):
	if offset:
		x = sm.add_constant(x)
	try:
		model = sm.OLS(y, x)
		results = model.fit()
		return results.params
	except:
		return [np.nan, np.nan]

def cal_adx(df, N, M):
	hd = df['high'].diff()
	ld = -df['low'].diff()
	dmp = pd.DataFrame({'dmp': [0] * len(hd)}, index=hd.index)
	dmp[(hd > 0) & (ld < 0)] = hd
	dmp = dmp.rolling(N).sum()
	dmm = pd.DataFrame({'dmm': [0] * len(ld)}, index=ld.index)
	dmm[(hd < 0) & (ld > 0)] = ld
	dmm = dmm.rolling(N).sum()
	temp = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), \
										abs(df['low'] - df['close'].shift(1))], axis=1)
	tr = temp.max(axis=1)

	s_index = dmm.index & tr.index & dmp.index
	dmp = dmp.loc[s_index]
	dmm = dmm.loc[s_index]
	tr = tr.loc[s_index]
	pdi = 100 * dmp['dmp'] / tr
	mdi = dmm['dmm'] * 100 / tr

	dx = abs(pdi - mdi) / (pdi + mdi) * 100
	adx = dx.rolling(M).mean()
	adx = pd.DataFrame(adx, columns=['adx'])
	return tr, adx

def cal_kdj(df, N, M):
	df['l_low'] = df['low'].rolling(N).min()
	df['h_high'] = df['high'].rolling(N).max()
	df['rsv'] = (df['close'] - df['l_low']) / (df['h_high'] - df['l_low'])
	df['k'] = df['rsv'].ewm(adjust=False, alpha=1 / M).mean()
	df['d'] = df['k'].ewm(adjust=False, alpha=1 / M).mean()
	df['j'] = 3 * df['k'] - 2 * df['d']
	return df['j']

def gen_x_feature(kline_df, short_n, mid_n, long_n):
	x_label = pd.DataFrame()
	x_label['pct_short'] = kline_df['close'].rolling(window=short_n).apply(lambda x: x[-1]/x[0]-1)
	x_label['pct_mid'] = kline_df['close'].rolling(window=mid_n).apply(lambda x: x[-1]/x[0]-1)
	x_label['pct_long'] = kline_df['close'].rolling(window=long_n).apply(lambda x: x[-1]/x[0]-1)
	x_label['ma_short_distance_pct'] = kline_df['close'] / kline_df['close'].rolling(window=short_n).mean() - 1
	x_label['ma_long_distance_pct'] = kline_df['close'] / kline_df['close'].rolling(window=long_n).mean() - 1
	x_label['RSI_short'] = kline_df['pct'].rolling(window=short_n).apply(rsi)
	x_label['RSI_long'] = kline_df['pct'].rolling(window=long_n).apply(rsi)

	x_label['slope_short'] = kline_df['close'].rolling(window=short_n).apply(lambda x: simple_OLS(range(short_n), x, offset=True)[-1])
	x_label['slope_long'] = kline_df['close'].rolling(window=long_n).apply(lambda x: simple_OLS(range(long_n), x, offset=True)[-1])
	x_label['double_ma'] = kline_df['close'].rolling(window=short_n).mean()/kline_df['close'].rolling(window=long_n).mean() - 1
	x_label['donchian_high_distance_pct'] = (kline_df['high'] - kline_df['low'].shift(1).rolling(window=mid_n).min()) \
	                                        / (kline_df['high'].shift(1).rolling(window=mid_n).max() - kline_df['low'].shift(1).rolling(window=mid_n).min())
	x_label['donchian_low_distance_pct'] = (kline_df['low'] - kline_df['low'].shift(1).rolling(window=mid_n).min()) \
	                                        / (kline_df['high'].shift(1).rolling(window=mid_n).max() - kline_df['low'].shift(1).rolling(window=mid_n).min())
	x_label['donchian_distance_pct'] = (kline_df['close'] - kline_df['low'].shift(1).rolling(window=mid_n).min()) \
	                                    / (kline_df['high'].shift(1).rolling(window=mid_n).max() - kline_df['low'].shift(1).rolling(window=mid_n).min())
	x_label['booling_distance_pct'] = (kline_df['close'] - (kline_df['close'].rolling(window=mid_n).mean() - kline_df['close'].rolling(window=mid_n).std()*2)) \
																		 / (kline_df['close'].rolling(window=mid_n).std()*4)

	x_label['TR'], x_label['ADX'] = cal_adx(kline_df, N=mid_n, M=short_n)
	x_label['kdj'] = cal_kdj(kline_df, N=mid_n, M=short_n)
	x_label['dif'] = kline_df['close'].ewm(span=mid_n, adjust=False).mean() - kline_df['close'].ewm(span=long_n, adjust=False).mean()
	x_label['dea'] = x_label['dif'].rolling(window=short_n).mean()
	x_label['hist'] = (x_label['dif'] - x_label['dea']) * 2
	x_label['osc'] = kline_df['close']/kline_df['close'].rolling(short_n).mean()
	x_label['vr'] = ((kline_df['pct']>0).astype(int) * kline_df['volume']).rolling(short_n).sum() / ((kline_df['pct']<0).astype(int) * kline_df['volume']).rolling(long_n).sum()
	x_label['vr'][x_label['vr'] > 500] = 500
	x_label['wr'] = (kline_df['high'].rolling(short_n).max() - kline_df['close']) / (kline_df['high'].rolling(short_n).max() - kline_df['low'].rolling(short_n).min())
	x_label['vosc'] = kline_df['volume'].rolling(long_n).mean() - kline_df['volume'].rolling(short_n).mean()
	x_label['cvlt'] = (kline_df['high'] - kline_df['low']).ewm(span=short_n).mean() / (kline_df['high'] - kline_df['low']).ewm(span=short_n).mean().shift(1) - 1
	return x_label

def gen_x_feature_impl(kline_df, n):
	x_label = pd.DataFrame()
	x_label['pct'] = kline_df['close'].rolling(window=n).apply(lambda x: x[-1]/x[0]-1)
	x_label['ma_distance_pct'] = kline_df['close'] / kline_df['close'].rolling(window=n).mean() - 1
	x_label['RSI'] = kline_df['pct'].rolling(window=n).apply(rsi)
	x_label['slope'] = kline_df['close'].rolling(window=n).apply(lambda x: simple_OLS(range(n), x, offset=True)[-1])
	x_label['double_ma'] = kline_df['close'].rolling(window=n).mean()/kline_df['close'].rolling(window=2*n).mean() - 1
	x_label['donchian_high_distance_pct'] = (kline_df['high'] - kline_df['low'].shift(1).rolling(window=n).min()) \
	                                        / (kline_df['high'].shift(1).rolling(window=n).max() - kline_df['low'].shift(1).rolling(window=n).min())
	x_label['donchian_low_distance_pct'] = (kline_df['low'] - kline_df['low'].shift(1).rolling(window=n).min()) \
	                                        / (kline_df['high'].shift(1).rolling(window=n).max() - kline_df['low'].shift(1).rolling(window=n).min())
	x_label['donchian_distance_pct'] = (kline_df['close'] - kline_df['low'].shift(1).rolling(window=n).min()) \
	                                    / (kline_df['high'].shift(1).rolling(window=n).max() - kline_df['low'].shift(1).rolling(window=n).min())
	x_label['booling_distance_pct'] = (kline_df['close'] - (kline_df['close'].rolling(window=n).mean() - kline_df['close'].rolling(window=n).std()*2)) \
																		 / (kline_df['close'].rolling(window=n).std()*4)

	x_label['TR'], x_label['ADX'] = cal_adx(kline_df, N=2*n, M=n)
	x_label['kdj'] = cal_kdj(kline_df, N=2*n, M=n)
	x_label['dif'] = kline_df['close'].ewm(span=n, adjust=False).mean() - kline_df['close'].ewm(span=2*n, adjust=False).mean()
	x_label['dea'] = x_label['dif'].rolling(window=n).mean()
	x_label['hist'] = (x_label['dif'] - x_label['dea']) * 2
	x_label['osc'] = kline_df['close']/kline_df['close'].rolling(n).mean()
	x_label['vr'] = ((kline_df['pct']>0).astype(int) * kline_df['volume']).rolling(n).sum() / ((kline_df['pct']<0).astype(int) * kline_df['volume']).rolling(2*n).sum()
	x_label['vr'][x_label['vr'] > 500] = 500
	x_label['wr'] = (kline_df['high'].rolling(n).max() - kline_df['close']) / (kline_df['high'].rolling(n).max() - kline_df['low'].rolling(n).min())
	x_label['vosc'] = kline_df['volume'].rolling(2*n).mean() - kline_df['volume'].rolling(n).mean()
	x_label['cvlt'] = (kline_df['high'] - kline_df['low']).ewm(span=n).mean() / (kline_df['high'] - kline_df['low']).ewm(span=n).mean().shift(1) - 1
	return x_label

def gen_x_feature_multi(kline_df, n_list):
	x_label_group = pd.DataFrame()
	for n in n_list:
		x_label = gen_x_feature_impl(kline_df, n)
		x_label.columns = [col+'_%s' % n for col in x_label.columns]
		x_label_group = pd.concat([x_label_group, x_label], axis=1)
	return x_label_group

def gen_y_label_rm1(kline_df, min_tradeable_pct=0.0005):
	kline_df['pct'] = kline_df['close'] / kline_df['open'] - 1
	tradeable_fetch_logic = abs(kline_df['pct']) > abs(kline_df['pct']).mean()
	kline_df['y_label'] = 0
	kline_df['y_label'][kline_df['pct'] >= min_tradeable_pct] = 1
	kline_df['y_label'][kline_df['pct'] <= -min_tradeable_pct] = -1
	kline_df['y_label'][tradeable_fetch_logic & (kline_df['pct'] >= min_tradeable_pct)] = 2
	kline_df['y_label'][tradeable_fetch_logic & (kline_df['pct'] <= -min_tradeable_pct)] = -2
	return kline_df['y_label']

def gen_y_pred_from_proba(y_pred_proba, y_label_index, threshold=0.4):
	y_pred = []
	for array in y_pred_proba:
		label = y_label_index[array > threshold]
		if len(label) == 0:
			y_pred.append(0)
		else:
			y_pred.append(y_label_index[array.argmax()])
	return np.array(y_pred)

# def gen_y_pred_from_proba(y_pred_proba, y_label_index):
# 	y_pred = []
# 	for array in y_pred_proba:
# 		prods = pd.Series(array, index=y_label_index).sort_values(ascending=False)
# 		if prods.index[0] in (-2, 2) and prods.index[0] * prods.index[1] > 0:
# 			y_pred.append(prods.index[0])
# 		elif prods.iloc[0] > 0.5:
# 			y_pred.append(prods.index[0])
# 		else:
# 			y_pred.append(0)
# 	return np.array(y_pred)

def human_gridsearch_rm(clf, x_feature, y_label, train_pct, searched_params, fixed_params, show_detail=False):
	train_num = int(len(x_feature) * train_pct)
	train_x_feature = x_feature[:train_num]
	train_y_label = y_label[:train_num]
	test_x_feature = x_feature[train_num:]
	test_y_label = y_label[train_num:]
	if show_detail:
		print("train length: %s,   test length: %s" % (len(train_y_label), len(test_y_label)))
	score_list = []
	gridsearch_params = [dict(zip(searched_params.keys(), k)) for k in itertools.product(*searched_params.values())]
	for param in gridsearch_params:
		model = clf(**param, **fixed_params)
		model.fit(train_x_feature, train_y_label)
		pred_prob = model.predict_proba(test_x_feature)
		y_pred = gen_y_pred_from_proba(pred_prob, np.array([-2,-1,0,1,2]))
		l1 = len([y for i, y in enumerate(y_pred) if abs(y) >= 1 and y * test_y_label[i] > 0])
		l2 = len([y for i, y in enumerate(y_pred) if abs(y) >= 1 and y * test_y_label[i] <= 0])
		p2 = l1 / (l1 + l2)
		score_list.append(p2)
		if show_detail:
			print("searched_param: %s   l1: %s   l2: %s   score: %s" % (param, l1, l2, p2))
	param = gridsearch_params[score_list.index(max(score_list))]
	model = clf(**param, **fixed_params)
	model.fit(x_feature, y_label)
	return model
