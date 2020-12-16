import requests
import json
import zlib
import logging

from websocket import create_connection
import tornado
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado import gen
from tornado.websocket import websocket_connect

OkexWSApiUrl = 'wss://real.okex.com:8443/ws/v3'


def inflate(data):
  decompress = zlib.decompressobj(
    -zlib.MAX_WBITS
  )
  inflated = decompress.decompress(data)
  inflated += decompress.flush()
  return inflated


class OkexWebsocketClient(object):
  def __init__(self, api_host=None, logger=None, ioloop=None, **kargs):
    self._url = api_host or OkexWSApiUrl
    self._ioloop = ioloop or IOLoop.current()
    self.logger = logger or logging.getLogger(__name__)
    self._ws = None
    self._message_list = []
    self._ioloop.add_callback(self._connect)

  def _on_message(self, msg):
    data = json.loads(inflate(msg).decode('utf-8'))
    if data.get('event'):
      self.logger.info(data)
    elif data.get('table'):
      self._read_msg(data)
    else:
      self.logger.info(data)

  def _read_msg(self, msg):
    raise NotImplementedError()

  @gen.coroutine
  def _run(self):
    while True:
      for msg in self._message_list:
        self._ws.write_message(msg)
      msg = yield self._ws.read_message()
      if msg is None:
        continue
      self._on_message(msg)
      self._message_list = []

  @gen.coroutine
  def _connect(self):
    self._ws = yield websocket_connect(self._url)
    self._ioloop.add_callback(self._run)

  def _keep_alive(self):
    self._ws.write_message('ping')

  def start(self):
    PeriodicCallback(self._keep_alive, 10*1000).start()
    self._ioloop.start()
