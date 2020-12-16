import datetime
import functools
import logging
from tornado.ioloop import IOLoop, PeriodicCallback
import warnings
warnings.filterwarnings('ignore')

from Okex.event_handler import OkexEventHandler
from Okex.private_client import OKexSwapPrivateClient
from Okex.product import OkexProduct
from Okex.okex_executor import OkexSwapExecutor
from Strategy.random_forest_classifier.model import TradeableModel


class Strategy(object):
  def __init__(self, *, strategy_config, key_file, simulated=False, ioloop=None):
    self.strategy_config = strategy_config
    self.ioloop = ioloop or IOLoop.current()
    self.logger = logging.getLogger('OkexStrategy')
    self.event_handler = OkexEventHandler(ioloop=ioloop)
    trade_client = OKexSwapPrivateClient(key_file=key_file, simulated=simulated, logger=self.logger)
    self.executor = OkexSwapExecutor(trade_client, strategy_config, self.logger)
    self._model = {}

  def trade(self, product):
    pred = self._model[product.coin].current_predict()
    ticker = self.event_handler.book.get_ticker(product.native_swap_symbol)
    if datetime.datetime.now().timestamp() - ticker.timestamp > 60:
      self.event_handler.ticker_recovery()
      self.logger.info("resubscribe ticker: %s" % product.coin)
      return
    ask0 = ticker.ask0
    bid0 = ticker.bid0
    self.logger.info("current pred: %s" % pred)
    self.logger.info("current ask: %s, bid: %s" % (ask0, bid0))
    if pred >= 1:
      self.executor.buy_all(product)
    elif pred <= -1:
      self.executor.sell_all(product)
    else:
      self.executor.close_all(product)

  def prepare_model(self, product):
    params = self.strategy_config[product.coin]['model']
    model = TradeableModel(exchange='Okex', symbol=product.spot_symbol, kline_period='1h', params=params)
    self.logger.info("begin to train %s" % product.coin)
    model.train(grid_search=False, load_model_path=params.get('model_path'))
    self._model.update({product.coin: model})

  def prepare(self):
    for coin in self.strategy_config.keys():
      product = OkexProduct.FromCoin(coin)
      self.prepare_model(product)
      self.event_handler.subscribe_depth('Swap', product.native_swap_symbol)
      callback = functools.partial(self.trade, product)
      PeriodicCallback(callback, 30 * 1000).start()
    PeriodicCallback(self.executor.print_balance, 60 * 1000).start()
    PeriodicCallback(self.executor.print_position, 60 * 1000).start()

  def start(self, exit_after_min=None):
    self.logger.info("Start strategy, begin to trade: %s" % str(self.strategy_config.keys()))
    if exit_after_min:
      self.ioloop.add_timeout(datetime.timedelta(minutes=exit_after_min), self.ioloop.stop)
    self.prepare()
    self.ioloop.start()
    self.logger.info("Exit all")


if __name__ == '__main__':
  logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(name)s - [%(lineno)d] - %(message)s'
  )
  sim_key_file = 'Okex/sim_trade_key.json'
  real_key_file = 'Okex/trade_key.json'
  strategy_config = {
    'BTC': {
      'model': {
        'short_n': 2,
        'long_n': 24,
      },
      'trade': {
        'max_pos': 30
      }
    }
  }
  strategy = Strategy(strategy_config=strategy_config, key_file=real_key_file, simulated=False)
  strategy.start(exit_after_min=24*60)
