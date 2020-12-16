import datetime
import json
import pandas as pd
import pytz

from Okex.public_client import OkexPublicClient

OKEX_BTC_1H_KLINE = '/home/zelinchen/Private/okex_btc_usdt_kline_1h.csv'

KLINE_CSV_PATH = 'Data/kline_info.csv'

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
			result.append([trading_date, exchange, symbol, period, int(kline['kline_timestamp'])/1e+9, datetime.datetime.utcfromtimestamp(int(kline['kline_timestamp'])/1e+9),
			             kline['open'], kline['high'], kline['low'], kline['close'], kline['volume']])
	result_df = pd.DataFrame(result, columns=cols)
	result_df.sort_values('timestamp', inplace=True)
	result_df.index = range(len(result_df))
	return result_df


def get_public_client(market_type, exchange):
	if exchange == 'Okex':
		return OkexPublicClient()
	else:
		raise NotImplementedError()


def parse_kline_period(exchange, period):
	kline_period_map = {
		'Okex': {'1h': 3600}
	}
	return kline_period_map[exchange][period]


def get_kline_util(market_type, exchange, symbol, period, start_dt=None, end_dt=None,
                   query_new=False, kline_csv_path=None, save_new=False):
	kline_csv_path = kline_csv_path or KLINE_CSV_PATH
	kline_df = pd.read_csv(kline_csv_path)
	kline_df = kline_df[(kline_df['exchange'] == exchange)
	                    & (kline_df['symbol'] == symbol)
											& (kline_df['period'] == period)]
	if start_dt and end_dt:
		kline_df = kline_df[(kline_df['datetime']>=start_dt) & (kline_df['datetime']<=end_dt)]
	extend_klines = []
	if query_new:
		start_dt = datetime.datetime.strptime(kline_df['datetime'].max(), '%Y-%m-%d %H:%M:%S')
		end_dt = datetime.datetime.utcnow()
		if end_dt - start_dt <= datetime.timedelta(hours=0):
			return kline_df
		kline_period = parse_kline_period(exchange, period)
		client = get_public_client(market_type, exchange)
		start_dt = (start_dt-datetime.timedelta(hours=2)).isoformat() + 'Z'
		end_dt = end_dt.isoformat() + 'Z'
		extend_klines = client.query_full_kline(symbol, kline_period, start_dt, end_dt)
	if extend_klines:
		extend_kline_df = []
		for kline in extend_klines:
			iso_dt = kline[0]
			dt = datetime.datetime.strptime(iso_dt, "%Y-%m-%dT%H:%M:%S.%fZ")
			dt = dt.replace(tzinfo=pytz.UTC)
			trading_date = dt.date().strftime('%Y-%m-%d')
			timestamp = dt.timestamp()
			extend_kline_df.append([trading_date, exchange, symbol, period, timestamp, dt.strftime("%Y-%m-%d %H:%M:%S"),
			                        float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]), float(kline[5])])
		extend_kline_df = pd.DataFrame(extend_kline_df, columns=kline_df.columns)
		extend_kline_df.sort_values('datetime', inplace=True)
		kline_df = kline_df.append(extend_kline_df)
		kline_df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
		if save_new:
			raw_kline_df = pd.read_csv(kline_csv_path)
			raw_kline_df = raw_kline_df.append(extend_kline_df)
			raw_kline_df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
			raw_kline_df.to_csv(kline_csv_path, index=False)
		kline_df.index = range(len(kline_df))
	return kline_df
