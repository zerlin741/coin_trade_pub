import collections
import hjson
import os

_AuthKey = collections.namedtuple("_AuthKey", [
	'owner',
	'access_key',
	'api_key',
	'secret_key',
	'passphrase',
	'raw',
	'key_file'
])

class  AuthKey(_AuthKey):
	def __new__(cls, owner=None, access_key=None, api_key=None, secret_key=None, passphrase=None, raw=None, key_file=None):
		return super(AuthKey, cls).__new__(cls, owner, access_key, api_key, secret_key, passphrase, raw, key_file)

	@staticmethod
	def from_file(key_file):
		assert os.path.exists(key_file), key_file
		raw_key = {}
		with open(key_file) as f:
			raw_key = hjson.loads(f.read())
		return AuthKey(raw=raw_key, key_file=key_file, **raw_key)

if __name__ == '__main__':
	key_file = '../Okex/sim_trade_key.json'
	auth_key = AuthKey.from_file(key_file)
