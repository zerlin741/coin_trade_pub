import collections
import datetime
import pytz
import json
import zlib
from tornado import gen

from Okex.websocket_client import OkexWebsocketClient

SymbolTicker = collections.namedtuple(
  'SymbolTicker', ['timestamp', 'ask0', 'bid0'])

SymbolKlines = collections.namedtuple(
  'Symbolklines', ['timestamp', 'period', 'open', 'close', 'high', 'low', 'volume']
)


class BookMap(object):
  def __init__(self):
    self._tickers = {}
    self._klines = {}

  def update_ticker(self, *, symbol, timestamp, ask0, bid0):
    self._tickers.update({symbol: SymbolTicker(timestamp, ask0, bid0)})

  def update_kline(self, *, symbol, timestamp, period, open, close, high, low, amount, **kwargs):
    self._klines.update({symbol: SymbolKlines(timestamp, period, open, close, high, low, amount)})

  def get_ticker(self, symbol):
    return self._tickers.get(symbol)

class OkexEventHandler(OkexWebsocketClient):
  def __init__(self, ioloop=None, logger=None):
    super(OkexEventHandler, self).__init__(ioloop=ioloop, logger=logger)
    self.book = BookMap()
    self.event_universe = collections.defaultdict(list)

  def subscribe_kline(self, symbol, period):
    assert period in ('60s', '180s', '300s', '900s', '1800s', '3600s', '7200s', '14400s', '21600s', '43200s', '86400s', '604800s'), period
    topic = {"op": "subscribe", "args": ["spot/candle%s:%s" % (period, symbol)]}
    self._message_list.append(json.dumps(topic))

  def subscribe_ticker(self, market_type, symbol):
    topic = {"op": "subscribe", "args": ["%s/ticker:%s" % (market_type.lower(), symbol)]}
    self._message_list.append(json.dumps(topic))
    self.event_universe['ticker'].append(topic)

  def subscribe_depth(self, market_type, symbol):
    topic = {"op": "subscribe", "args": ["%s/depth5:%s" % (market_type.lower(), symbol)]}
    self._message_list.append(json.dumps(topic))
    self.event_universe['ticker'].append(topic)

  def ticker_recovery(self):
    self._connect()
    for topic in self.event_universe['ticker']:
      self._ws.write_message(json.dumps(topic))

  def subscribe_event(self, event_type, callback):
    if event_type == 'trade':
      self.event_universe[event_type].append(callback)
    else:
      raise NotImplementedError(event_type)

  def publish_event(self, event_type):
    for callback in self.event_universe.get(event_type, []):
      callback()

  def _read_msg(self, msg):
    market_type, channel = msg['table'].split('/')
    if channel == 'ticker':
      data = msg['data']
      assert len(data) == 1, msg
      symbol = data[0]['instrument_id']
      ask0 = data[0]['best_ask']
      bid0 = data[0]['best_bid']
      timestamp = datetime.datetime.strptime(data[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.UTC).timestamp()
      self.book.update_ticker(symbol=symbol, timestamp=timestamp, ask0=ask0, bid0=bid0)
    elif channel == 'depth5':
      data = msg['data']
      assert len(data) == 1, msg
      symbol = data[0]['instrument_id']
      ask0 = data[0]['asks'][0][0]
      bid0 = data[0]['bids'][0][0]
      timestamp = datetime.datetime.strptime(data[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.UTC).timestamp()
      self.book.update_ticker(symbol=symbol, timestamp=timestamp, ask0=ask0, bid0=bid0)
    else:
      raise NotImplementedError(channel)



if __name__ == '__main__':
  feedpublisher = OkexEventHandler()
  feedpublisher.subscribe_ticker('Swap', 'BTC-USD-SWAP')
  feedpublisher.start()
