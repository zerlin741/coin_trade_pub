


class OkexProduct(object):
	def __init__(self, coin):
		self._coin = coin

	@staticmethod
	def FromCoin(coin):
		return OkexProduct(coin=coin)

	@property
	def native_spot_symbol(self):
		return '%s-USDT' % self._coin.upper()

	@property
	def spot_symbol(self):
		return self.native_spot_symbol

	@property
	def coin(self):
		return self._coin.upper()

	@property
	def native_swap_symbol(self, quote='USD'):
		return '%s-%s-SWAP' % (self._coin.upper(), quote)
