"""A small but realistic multi-module repository used as a deterministic retrieval benchmark."""

from __future__ import annotations

from pathlib import Path

FILES: dict[str, str] = {
    "README.md": """# ShopAPI

ShopAPI is a small e-commerce backend with a TypeScript cart frontend. It exposes a product
catalog, order creation with payment and inventory reservation, and JWT authentication.

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env
python -m shopapi.server
```

## Configuration

| Variable | Purpose |
| --- | --- |
| `SHOP_DB_URL` | Database connection string |
| `PAYMENT_API_KEY` | Key for the payment provider |
| `LEGACY_CACHE_TTL` | Cache lifetime in seconds |

## API

- `GET /products` lists the catalog.
- `POST /orders` creates an order.
- `DELETE /orders/{order_id}` cancels an order.

See [the architecture notes](docs/architecture.md) and [contributing](CONTRIBUTING.md).
""",
    ".env.example": "SHOP_DB_URL=sqlite:///shop.db\nPAYMENT_API_KEY=your_key_here\n",
    "requirements.txt": "fastapi==0.110.0\npydantic>=2\nrequests==2.19.0\n",
    "Dockerfile": "FROM python:3.11-slim\nCOPY . /app\nRUN pip install -r /app/requirements.txt\nCMD [\"python\", \"-m\", \"shopapi.server\"]\n",
    "shopapi/__init__.py": '"""ShopAPI package."""\n',
    "shopapi/server.py": '''import os

from fastapi import FastAPI

from shopapi.catalog import Catalog
from shopapi.orders import create_order

app = FastAPI()
catalog = Catalog()
DATABASE_URL = os.getenv("SHOP_DB_URL", "sqlite:///shop.db")


@app.get("/products")
def list_products_endpoint():
    """HTTP endpoint returning every product in the catalog."""
    return catalog.list_products()


@app.post("/orders")
def create_order_endpoint(payload: dict):
    """HTTP endpoint that places an order for the requested items."""
    return create_order(payload["items"], payload["card_token"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
''',
    "shopapi/catalog.py": '''from shopapi.utils.money import round_currency

PRODUCTS = {"sku-1": {"name": "Mug", "price": 12.5}, "sku-2": {"name": "Shirt", "price": 20.0}}


class Catalog:
    """In-memory product catalog."""

    def list_products(self):
        return [{"sku": sku, **data} for sku, data in PRODUCTS.items()]

    def find_product(self, sku):
        """Look up a single product by its SKU; raises KeyError when it does not exist."""
        if sku not in PRODUCTS:
            raise KeyError(f"unknown sku {sku}")
        return PRODUCTS[sku]


def apply_discount(price, percent):
    """Reduce a price by a percentage discount and round to currency precision."""
    if not 0 <= percent <= 100:
        raise ValueError("discount percent must be between 0 and 100")
    return round_currency(price * (1 - percent / 100))
''',
    "shopapi/orders.py": '''from shopapi.catalog import Catalog
from shopapi.inventory import reserve_stock
from shopapi.payments import charge_card
from shopapi.utils.money import round_currency


def compute_total(items, catalog):
    """Sum line prices (unit price x quantity) for an order."""
    total = 0.0
    for item in items:
        product = catalog.find_product(item["sku"])
        total += product["price"] * item["quantity"]
    return round_currency(total)


def create_order(items, card_token):
    """Create an order: price the items, reserve inventory, then charge the card."""
    catalog = Catalog()
    total = compute_total(items, catalog)
    for item in items:
        reserve_stock(item["sku"], item["quantity"])
    receipt = charge_card(total, card_token)
    return {"items": items, "total": total, "receipt": receipt}
''',
    "shopapi/payments.py": '''import os
import time

import requests

PAYMENT_URL = "https://payments.example.com/charge"


def _retry(operation, attempts=3, base_delay=0.5):
    """Run an operation with exponential backoff between failed attempts."""
    for attempt in range(attempts):
        try:
            return operation()
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def charge_card(amount, card_token):
    """Charge a card through the payment provider, retrying transient failures."""
    api_key = os.getenv("PAYMENT_API_KEY")
    return _retry(lambda: requests.post(PAYMENT_URL, json={"amount": amount, "token": card_token}, headers={"Authorization": api_key}).json())
''',
    "shopapi/inventory.py": '''class StockLedger:
    """Tracks reserved and available stock per SKU."""

    def __init__(self):
        self.available = {"sku-1": 10, "sku-2": 5}
        self.reserved = {}


LEDGER = StockLedger()


def reserve_stock(sku, quantity):
    """Move quantity from available to reserved; fails when stock is insufficient."""
    if LEDGER.available.get(sku, 0) < quantity:
        raise RuntimeError("insufficient stock")
    LEDGER.available[sku] -= quantity
    LEDGER.reserved[sku] = LEDGER.reserved.get(sku, 0) + quantity


def release_stock(sku, quantity):
    """Return previously reserved units to available stock (e.g. after a cancelled order)."""
    LEDGER.reserved[sku] -= quantity
    LEDGER.available[sku] += quantity
''',
    "shopapi/auth.py": '''import hashlib
import hmac
import time


def verify_jwt_token(token, secret):
    """Validate a JWT: check the HMAC signature and reject expired tokens."""
    header, payload, signature = token.split(".")
    expected = hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise PermissionError("bad signature")
    if int(payload.split(":")[-1]) < time.time():
        raise PermissionError("token expired")
    return payload


def hash_password(password, salt):
    """Derive a password hash with PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()
''',
    "shopapi/utils/__init__.py": "",
    "shopapi/utils/money.py": '''def round_currency(value):
    """Round a float to two decimal places for display and storage."""
    return round(value + 1e-9, 2)


def to_cents(dollars):
    """Convert a dollar amount into integer cents."""
    return int(round(dollars * 100))
''',
    "web/src/api.ts": '''export async function fetchProducts(): Promise<any[]> {
  const response = await fetch("/products");
  return response.json();
}

export async function submitOrder(items: any[], cardToken: string) {
  const response = await fetch("/orders", { method: "POST", body: JSON.stringify({ items, card_token: cardToken }) });
  return response.json();
}
''',
    "web/src/cart.ts": '''import { submitOrder } from "./api";

export class Cart {
  items: any[] = [];

  addItem(sku: string, quantity: number) {
    this.items.push({ sku, quantity });
  }

  async checkout(cardToken: string) {
    const receipt = await submitOrder(this.items, cardToken);
    this.items = [];
    return receipt;
  }
}
''',
    "tests/test_orders.py": '''from shopapi.orders import compute_total
from shopapi.catalog import Catalog


def test_compute_total_sums_line_items():
    assert compute_total([{"sku": "sku-1", "quantity": 2}], Catalog()) == 25.0
''',
}

# (question, gold files, gold symbol or None)
QUESTIONS: list[tuple[str, set[str], str | None]] = [
    ("How is the JWT token verified?", {"shopapi/auth.py"}, "verify_jwt_token"),
    ("Where is the order total computed?", {"shopapi/orders.py"}, "compute_total"),
    ("How does payment retry work when a card charge fails?", {"shopapi/payments.py"}, "_retry"),
    ("What happens when an order is created?", {"shopapi/orders.py"}, "create_order"),
    ("Which endpoint lists products?", {"shopapi/server.py"}, "list_products_endpoint"),
    ("How do I run the server locally?", {"README.md"}, None),
    ("How are discounts applied to prices?", {"shopapi/catalog.py"}, "apply_discount"),
    ("How is stock reserved for an order?", {"shopapi/inventory.py"}, "reserve_stock"),
    ("What does the Cart checkout do in the frontend?", {"web/src/cart.ts"}, "Cart.checkout"),
    ("How does the frontend fetch products?", {"web/src/api.ts"}, "fetchProducts"),
    ("How are money amounts rounded?", {"shopapi/utils/money.py"}, "round_currency"),
    ("Which environment variable configures the database?", {"shopapi/server.py", "README.md", ".env.example"}, None),
    ("Which tests cover order totals?", {"tests/test_orders.py"}, None),
    ("what calls reserve_stock?", {"shopapi/orders.py", "shopapi/inventory.py"}, "reserve_stock"),
    ("How are passwords hashed?", {"shopapi/auth.py"}, "hash_password"),
    ("Where is the StockLedger class defined?", {"shopapi/inventory.py"}, "StockLedger"),
    ("How is a product looked up by SKU?", {"shopapi/catalog.py"}, "Catalog.find_product"),
    ("convert dollars to cents", {"shopapi/utils/money.py"}, "to_cents"),
    ("What does release_stock do?", {"shopapi/inventory.py"}, "release_stock"),
    ("How is the payment API key used?", {"shopapi/payments.py"}, "charge_card"),
]

# Docs-vs-code drift the detector is expected to find in this fixture.
EXPECTED_DRIFT = {
    "route_missing": "DELETE /orders/{order_id}",
    "env_unused": "LEGACY_CACHE_TTL",
    "broken_link": "docs/architecture.md",
}


def build_fixture(root: Path) -> Path:
    for relative, content in FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root
