import collections
import datetime
import logging
import pandas as pd


class OkexSwapExecutor(object):
  def __init__(self, client, strategy_config, logger=None):
    self.client = client
    self.logger = logger or logging.getLogger(__name__)
    self._strategy_config = strategy_config
    self._last_trade = {}
    self._oid_list = []

  def print_position(self):
    client = self.client
    response = client.query_position_info()
    result_position = []
    for margin_pos in response:
      margin_mode = margin_pos['margin_mode']
      for position_info in margin_pos['holding']:
        position = float(position_info['position'])
        instrument_id = position_info['instrument_id']
        liquidation_price = position_info['liquidation_price']
        last = position_info['last']
        side = position_info['side']
        avg_cost = position_info['avg_cost']
        result_position.append(
            collections.OrderedDict({
                'margin_mode': margin_mode,
                'instrument_id': instrument_id,
                'position': position,
                'side': side,
                'last_price': last,
                'avg_cost': avg_cost,
                'liquidation_price': liquidation_price
            }))
    if result_position:
      result_position = pd.DataFrame(result_position)
      self.logger.info('POSITION:\n' + result_position.to_string())
    return

  def print_balance(self):
    client = self.client
    response = client.query_account_info()
    balance_infos = response['info']
    result_balance = []
    for balance_info in balance_infos:
      currency = balance_info['currency']
      total = float(balance_info['equity'])
      if total == 0:
        continue
      result_balance.append(
          collections.OrderedDict({
              'currency': currency,
              'total': total,
          }))
    if result_balance:
      result_balance = pd.DataFrame(result_balance)
      self.logger.info('BALANCE:\n' + result_balance.to_string())

  def clean_all_orders(self, product):
    response = self.client.query_product_orders(product.native_swap_symbol, state=0)
    order_ids = []
    for order_info in response['order_info']:
      order_ids.append(order_info['order_id'])
    if order_ids:
      self.client.cancel_batch_product_orders(product.native_swap_symbol, order_ids=order_ids)

  def buy_all(self, product, *, ask0=None, bid0=None):
    self.clean_all_orders(product)
    response = self.client.query_product_position(product.native_swap_symbol)
    max_pos = self._strategy_config[product.coin]['trade']['max_pos']
    for position_info in response['holding']:
      curr_position = float(position_info['position'])
      if position_info['side'] == 'long':
        if curr_position > max_pos:
          self.close_long(product=product, price=ask0, size=curr_position-max_pos)
        elif curr_position < max_pos:
          self.open_long(product=product, price=bid0, size=max_pos-curr_position)
      elif position_info['side'] == 'short':
        self.close_short(product=product, price=bid0, size=curr_position)

  def sell_all(self, product, *, ask0=None, bid0=None):
    self.clean_all_orders(product)
    response = self.client.query_product_position(product.native_swap_symbol)
    max_pos = self._strategy_config[product.coin]['trade']['max_pos']
    for position_info in response['holding']:
      curr_position = float(position_info['position'])
      if position_info['side'] == 'long':
        self.close_long(product=product, price=ask0, size=curr_position)
      elif position_info['side'] == 'short':
        if curr_position > max_pos:
          self.close_short(product=product, price=bid0, size=curr_position-max_pos)
        elif curr_position < max_pos:
          self.open_short(product=product, price=ask0, size=max_pos - curr_position)

  def close_all(self, product, *, ask0=None, bid0=None):
    self.clean_all_orders(product)
    response = self.client.query_product_position(product.native_swap_symbol)
    for position_info in response['holding']:
      curr_position = float(position_info['position'])
      if position_info['side'] == 'long' and curr_position > 0:
        self.close_long(product=product, price=ask0, size=curr_position)
      elif position_info['side'] == 'short' and curr_position > 0:
        self.close_short(product=product, price=bid0, size=curr_position)

  def open_long(self, *, product, price, size):
    direction = 1
    self.send_order(product=product,
                    price=price,
                    size=size,
                    direction=direction)

  def close_long(self, *, product, price, size):
    direction = 3
    self.send_order(product=product,
                    price=price,
                    size=size,
                    direction=direction)

  def open_short(self, *, product, price, size):
    direction = 2
    self.send_order(product=product,
                    price=price,
                    size=size,
                    direction=direction)

  def close_short(self, *, product, price, size):
    direction = 4
    self.send_order(product=product,
                    price=price,
                    size=size,
                    direction=direction)

  def send_order(self, *, product, price, size, direction):
    if float(size) <= 0:
      return
    self.logger.info("send order: symbol:%s, price:%s, size:%s, direction:%s" %
                     (product.native_swap_symbol, price, size, direction))
    client = self.client
    if price is None:
      match_price = 1
      order_type = 0
    else:
      match_price = 0
      order_type = 1
    client_oid = product.coin + str(int(datetime.datetime.now().timestamp()))
    self._oid_list.append(client_oid)
    res = client.place_order(instrument_id=product.native_swap_symbol,
                             client_oid=client_oid,
                             price=price,
                             size=int(size),
                             order_side=direction,
                             match_price=match_price,
                             order_type=order_type)
    self._last_trade.update({product.native_swap_symbol: datetime.datetime.utcnow()})
    return res

if __name__ == '__main__':
  from Okex.private_client import OKexSwapPrivateClient
  trade_client = OKexSwapPrivateClient(key_file='Okex/trade_key.json', simulated=False)
  strategy_config = {
    'BTC': {
      'model': {
        'short_n': 7,
        'mid_n': 14,
        'long_n': 30
      },
      'trade': {
        'max_pos': 80
      }
    }
  }
  executor = OkexSwapExecutor(trade_client, strategy_config)
  from Okex.product import OkexProduct
  p = OkexProduct.FromCoin('BTC')
  res = executor.print_balance()
