import hmac
import base64
import requests
import logging
import json
import urllib

from Tools.auth_base import AuthKey
from Tools.utils import get_unix_timestamp

OKEX_API_HOST = 'https://www.okex.com'
CONTENT_TYPE = 'Content-Type'
OK_ACCESS_KEY = 'OK-ACCESS-KEY'
OK_ACCESS_SIGN = 'OK-ACCESS-SIGN'
OK_ACCESS_TIMESTAMP = 'OK-ACCESS-TIMESTAMP'
OK_ACCESS_PASSPHRASE = 'OK-ACCESS-PASSPHRASE'
APPLICATION_JSON = 'application/json'


# signature
def signature(timestamp, method, request_path, body, secret_key):
    if str(body) == '{}' or str(body) == 'None':
        body = ''
    message = str(timestamp) + str.upper(method) + request_path + str(body)
    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    d = mac.digest()
    return base64.b64encode(d)


# set request header
def get_header(api_key, sign, timestamp, passphrase):
    header = dict()
    header[CONTENT_TYPE] = APPLICATION_JSON
    header[OK_ACCESS_KEY] = api_key
    header[OK_ACCESS_SIGN] = sign
    header[OK_ACCESS_TIMESTAMP] = str(timestamp)
    header[OK_ACCESS_PASSPHRASE] = passphrase
    return header


class OkexPrivateClientBase(object):
	def __init__(self, key_file, *, timeout=10, simulated=False, api_host=None, logger=None):
		self._api_host = api_host or OKEX_API_HOST
		self._auth_key = AuthKey.from_file(key_file)
		self._timeout = timeout
		self._simulated = simulated
		self._logger = logger or logging.getLogger(__name__)

	def _get_header(self, method, path, body):
		timestamp_str = get_unix_timestamp()
		api_key = self._auth_key.api_key
		secert_key = self._auth_key.secret_key
		passphrase = self._auth_key.passphrase
		sig = signature(timestamp_str, method, path, body, secert_key)
		header = get_header(api_key, sig, timestamp_str, passphrase)
		return header

	def query(self, method, path, params=None):
		assert method in ('GET', 'POST'), method
		if params is not None:
			params = {k: v for k, v in params.items() if v is not None}
			params = params or None
		url = urllib.parse.urljoin(self._api_host, path)
		body = json.dumps(params) if method == "POST" else ""
		if method == 'GET' and params:
			path = path + '?' + urllib.parse.urlencode(params)
		header = self._get_header(method, path, body)
		if self._simulated:
			header['x-simulated-trading'] = "1"
		response = None
		if method != 'GET':
			params = None
		try:
			response = requests.request(method,
			                            url,
			                            params=params,
			                            data=body,
			                            headers=header,
			                            timeout=self._timeout)
			content = response.json()
			return content
		except Exception as e:
			self._logger.exception('Exception %s: %s', type(e), e)
			self._logger.info(response.text)


class OkexSpotPrivateClient(OkexPrivateClientBase):
	def query_account_info(self):
		method = 'GET'
		path = '/api/spot/v3/accounts'
		return self.query(method, path)


class OKexSwapPrivateClient(OkexPrivateClientBase):
	def query_account_info(self):
		method = 'GET'
		path = '/api/swap/v3/accounts'
		return self.query(method, path)

	def query_product_account(self, product):
		method = 'GET'
		path = '/api/swap/v3/%s/accounts' % product
		return self.query(method, path)

	def query_position_info(self):
		method = 'GET'
		path = '/api/swap/v3/position'
		return self.query(method, path)

	def query_product_position(self, product):
		method = 'GET'
		if self._simulated:
			product = 'MN' + product
		path = '/api/swap/v3/%s/position' % product
		return self.query(method, path)

	def query_product_orders(self, product, state, *, from_id=None, to_id=None, limit=None):
		method = 'GET'
		if self._simulated:
			product = 'MN' + product
		path = '/api/swap/v3/orders/%s' % product
		params = {
			'state': state,
			'after': to_id,
			'before': from_id,
			'limit': limit
		}
		return self.query(method, path, params)

	def cancel_batch_product_orders(self, product, client_oids=None, order_ids=None):
		method = 'POST'
		if self._simulated:
			product = 'MN' + product
		path = '/api/swap/v3/cancel_batch_orders/%s' % product
		if order_ids:
			params = {'ids': order_ids}
		elif client_oids:
			params = {'client_oids': client_oids}
		return self.query(method, path, params)

	def place_order(
			self,
			*,
			size,
			order_side,
			instrument_id,
			client_oid=None,
			order_type=None,
			price=None,
			match_price=None,
	):
		method = 'POST'
		path = '/api/swap/v3/order'
		params = {
			'client_oid': client_oid,
			'type': order_side,
			'size': size,
			'order_type': order_type,
			'price': price,
			'match_price': match_price,
			'instrument_id': instrument_id,
		}
		if self._simulated:
			params['instrument_id'] = 'MN' + params['instrument_id']
		return self.query(method, path, params)


if __name__ == '__main__':
	key_file = 'Okex/sim_trade_key.json'
	client = OKexSwapPrivateClient(key_file, simluated=True)
	# print(client.place_order(size="1", order_side="1", instrument_id='MNBTC-USD-SWAP', price="10000")	print(client.query_product_position("MNBTC-USD-SWAP"))
	r = client.query_account_info()
	# client = OkexSpotPrivateClient(key_file, simluated=True)
	# print(client.query_account_info())
