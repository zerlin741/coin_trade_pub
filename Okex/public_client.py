import json
import logging
import requests
import urllib

OKEX_API_HOST = 'https://www.okex.com'


class OkexPublicClient(object):
	def __init__(self, api_host=None, timeout=10, logger=None):
		self._api_host = api_host or OKEX_API_HOST
		self._timeout = timeout
		self._logger = logger or logging.getLogger(__name__)

	def query(self, method, path, params=None):
		assert method in ('GET', 'POST'), method
		if params is not None:
			params = {k: str(v) for k, v in params.items() if v if not None}
			params = params or None
		url = urllib.parse.urljoin(self._api_host, path)
		body = json.dumps(params) if method == "POST" else ""
		if method != 'GET':
			params = None
		try:
			response = requests.request(method,
			                            url,
			                            params=params,
			                            data=body,
			                            timeout=self._timeout)
			content = response.json()
			return content
		except Exception as e:
			self._logger.exception('Exception %s: %s', type(e), e)
			self._logger.info(response.text)


	def query_klines(self, symbol, period, start=None, end=None):
		method = 'GET'
		path = '/api/spot/v3/instruments/%s/candles' % symbol
		params = {
			'start': start,
			'end': end,
			'granularity': period
		}
		return self.query(method, path, params)

	def query_full_kline(self, symbol, period, start, end):
		result = []
		klines = self.query_klines(symbol, period, start, end)
		result.extend(klines)
		while len(klines) == 200:
			klines = self.query_klines(symbol, period, start, klines[-1][0])
			result.extend(klines[1:])
		return result

if __name__ == '__main__':
	client = OkexPublicClient()
	print(client.query_klines('btc_usdt', 3600))