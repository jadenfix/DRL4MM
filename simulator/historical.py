"""Strict local order-event ingestion for chronological book replay."""
import csv
from dataclasses import dataclass
import math
from pathlib import Path

from .lob_simulator import Order, OrderSide, OrderType


@dataclass(frozen=True)
class MarketEvent:
    timestamp: float
    event: str
    order_id: str
    side: str = ''
    price: float = 0.0
    quantity: int = 0

    def apply(self, book):
        """Apply one external event without impersonating the agent's orders."""
        identity = 'feed:' + self.order_id
        if self.event == 'cancel':
            if not book.cancel_order(identity):
                raise ValueError(f'Cancel references an inactive external order: {self.order_id}')
            return
        if identity in book.orders:
            raise ValueError(f'Duplicate active external order: {self.order_id}')
        book.add_order(Order(
            order_id=identity, symbol=book.symbol,
            side=OrderSide.BUY if self.side == 'buy' else OrderSide.SELL,
            price=self.price, quantity=self.quantity, timestamp=self.timestamp,
            order_type=OrderType.MARKET if self.event == 'market' else OrderType.LIMIT,
        ))


def load_events(path, *, max_events=100000):
    """Read validated add/cancel/market rows in supplied order; never sort or fill gaps."""
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Replay CSV exceeds the 64 MiB input limit')
    events = []
    previous = -math.inf
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'timestamp', 'event', 'order_id', 'side', 'price', 'quantity'}
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError('Replay CSV has duplicate column names')
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Replay CSV requires columns: {", ".join(sorted(required))}')
        for line, row in enumerate(reader, start=2):
            try:
                if None in row:
                    raise ValueError('row contains more values than the CSV header')
                timestamp = float(row['timestamp'])
                kind = row['event'].strip().lower()
                identity = row['order_id'].strip()
                if not math.isfinite(timestamp) or timestamp < 0 or timestamp < previous:
                    raise ValueError('timestamps must be finite, nonnegative and chronological')
                if kind not in {'add', 'cancel', 'market'}:
                    raise ValueError('event must be add, cancel or market')
                if not identity or len(identity) > 256:
                    raise ValueError('order_id must contain 1–256 characters')
                side, price, quantity = '', 0.0, 0
                if kind != 'cancel':
                    side = row['side'].strip().lower()
                    price = float(row['price']) if kind == 'add' else 0.0
                    quantity = int(row['quantity'])
                    if side not in {'buy', 'sell'} or quantity <= 0:
                        raise ValueError('orders require buy/sell and a positive integer quantity')
                    if kind == 'add' and (not math.isfinite(price) or price <= 0):
                        raise ValueError('add events require a finite positive limit price')
                events.append(MarketEvent(timestamp, kind, identity, side, price, quantity))
                if len(events) > max_events:
                    raise ValueError(f'Replay exceeds {max_events} events')
                previous = timestamp
            except (ValueError, TypeError, AttributeError, KeyError) as error:
                raise ValueError(f'Invalid replay row {line}: {error}') from error
    if not events:
        raise ValueError('Replay CSV has no market events')
    return events
