import os
import re
import json
import argparse
import logging
import requests
from collections import defaultdict, Counter

from utils import login
from config import VICKI_APP

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("grace_period.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_name(name: str) -> str:
    """Keep only alphabetic characters, lowercased."""
    return ''.join(re.findall(r'[a-zA-Z]+', name)).lower()


def get_access_token(base_url: str, machine_id: str, machine_token: str, machine_api_key: str) -> str:
    return login.get_current_access_token(base_url, machine_id, machine_token, machine_api_key, logger)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------
def get_current_line_items(base_url: str, transaction_id: str, access_token: str) -> dict:
    """Fetch the current order from the loyalty API."""
    url = f"{base_url}/loyalty/orders/{transaction_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    logger.info(f"Fetched current line items for transaction {transaction_id}")
    return response.json()


def get_planogram(base_url: str, access_token: str) -> dict:
    """Fetch the machine's planogram so we can locate (row, column) for
    products that don't currently appear in the order's line items."""
    url = f"{base_url}/loyalty/machines/planogram"
    headers = {'Authorization': f'Bearer {access_token}'}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    logger.info("Fetched planogram")
    return response.json()


def change_line_items(access_token: str, base_url: str, transaction_id: str, body: dict) -> bool:
    """PATCH updated line items to the loyalty API."""
    url = f"{base_url}/loyalty/orders/{transaction_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    logger.debug(f"PATCH {url} — body: {body}")
    response = requests.patch(url, headers=headers, json=body)
    if response.status_code == 200:
        logger.info("Grace Period: PATCH successful")
        return True
    else:
        logger.error(f"PATCH failed: {response.status_code} {response.text}")
        return False


def get_updated_line_items(base_url: str, transaction_id: str, access_token: str) -> None:
    """Log the line items after an update for verification."""
    url = f"{base_url}/loyalty/orders/{transaction_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    logger.info(f"Updated line items: {response.json()}")


# ---------------------------------------------------------------------------
# Planogram lookup
# ---------------------------------------------------------------------------
def build_planogram_lookup(planogram_json: dict) -> dict:
    """
    Build a {cleaned_product_name: (row, column)} lookup from the planogram
    response.

    Some products occupy multiple shelf slots (e.g. several columns of the
    same drink). We keep the FIRST slot encountered for each product, since
    that's enough to construct an "add" line-item op — the loyalty API just
    needs *a* valid shelf location for the product, not a specific one tied
    to physical pick location.
    """
    lookup = {}
    shelves = planogram_json.get("planogram", {}).get("shelves", [])

    for shelf in shelves:
        product = shelf.get("product") or {}
        tray = shelf.get("tray") or {}
        name_clean = clean_name(product.get("name", ""))

        if not name_clean:
            continue
        if name_clean in lookup:
            continue  # already have a slot for this product, keep the first

        lookup[name_clean] = {
            "shelf_row": tray.get("row"),
            "shelf_column": tray.get("column"),
            "product_id": product.get("id"),
            "price": product.get("pricing", {}).get("effective_price"),
        }

    return lookup


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def compute_picked_products(cv_activities: list[dict]) -> Counter:
    """
    Given CV activity events, compute the net picked products
    (USER_PICKUP - USER_PUTBACK per product).
    """
    product_counts = defaultdict(lambda: {'USER_PICKUP': 0, 'USER_PUTBACK': 0})

    for item in cv_activities:
        for action, product in item.items():
            cleaned = clean_name(product)
            if action in product_counts[cleaned]:
                product_counts[cleaned][action] += 1

    picked = []
    for product, actions in product_counts.items():
        net = actions['USER_PICKUP'] - actions['USER_PUTBACK']
        if net > 0:
            picked.extend([product] * net)

    return Counter(picked)


def compute_bought_products(line_items: list[dict], valid_products_cleaned: list[str]) -> Counter:
    """
    Expand line items into a Counter of cleaned product names,
    filtered to only those in the planogram config.
    """
    bought = [
        clean_name(item["product_name"])
        for item in line_items
        for _ in range(item["quantity"])
    ]
    filtered = [p for p in bought if p in valid_products_cleaned]
    return Counter(filtered)


def build_downcharge_ops(line_items: list[dict], picked_counter: Counter, valid_products_cleaned: list[str]) -> list[dict]:
    """
    Build PATCH operations to align line_items with what was actually picked.

    - If a product was NOT picked at all  → delete op (reduce to zero)
    - If a product was picked LESS than charged → update op (set to picked qty)
    - If quantities already match → no op
    """
    ops = []
    for li in line_items:
        pname_clean = clean_name(li.get("product_name", ""))
        if pname_clean not in valid_products_cleaned:
            continue

        current_qty = int(li.get("quantity", 0))
        target_qty  = picked_counter.get(pname_clean, 0)
        row = li.get("shelf_row")
        col = li.get("shelf_column")

        if target_qty == 0 and current_qty > 0:
            ops.append({
                "quantity": current_qty,
                "shelf_row": row,
                "shelf_column": col,
                "op_type": "delete",
            })
        elif 0 < target_qty < current_qty:
            ops.append({
                "quantity": target_qty,
                "shelf_row": row,
                "shelf_column": col,
                "op_type": "update",
            })

    return ops


def build_upcharge_ops(
    line_items: list[dict],
    picked_counter: Counter,
    valid_products_cleaned: list[str],
    planogram_lookup: dict,
) -> list[dict]:
    """
    Build PATCH operations for products that were picked MORE than what's
    currently on the order — either a brand-new product not in line_items
    at all, or an existing line item where picked qty > charged qty.

    Since a brand-new product has no shelf_row/shelf_column on the order
    yet, we pull its location from the planogram lookup.

    - Product not in line_items at all, but picked → add op (full picked qty)
    - Product in line_items, but picked MORE than charged → add op (the delta)
    """
    ops = []

    # Map cleaned name -> current charged qty, for products already on the order
    charged_qty_by_product = defaultdict(int)
    for li in line_items:
        pname_clean = clean_name(li.get("product_name", ""))
        if pname_clean in valid_products_cleaned:
            charged_qty_by_product[pname_clean] += int(li.get("quantity", 0))

    for product, picked_qty in picked_counter.items():
        if product not in valid_products_cleaned:
            continue  # not a real/known product, skip

        current_qty = charged_qty_by_product.get(product, 0)

        if picked_qty <= current_qty:
            continue  # no upcharge needed for this product

        delta_qty = picked_qty - current_qty

        slot = planogram_lookup.get(product)
        if not slot:
            logger.warning(
                f"Cannot upcharge '{product}': picked {picked_qty}, charged {current_qty}, "
                f"but no planogram slot found for this product. Skipping."
            )
            continue

        ops.append({
            "quantity": delta_qty,
            "shelf_row": slot["shelf_row"],
            "shelf_column": slot["shelf_column"],
            "op_type": "add",
        })

    return ops


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_grace_period_check(
    transaction_id: str,
    json_file: str,
    cv_activities: list[dict],
    all_config_products: list,
) -> bool:
    """
    Full grace-period check:
      1. Load user activity JSON
      2. Compare CV picks vs purchased line items
      3. Downcharge / upcharge if mismatch, send alert

    Args:
        transaction_id:  The order/transaction UUID.
        json_file:       Path to the user_activity JSON file.
        cv_activities:   List of CV event dicts e.g. [{"USER_PICKUP": "Quest Chips"}]
        all_config_products: List of product name strings from the planogram config.

    Returns:
        True if alert was sent (mismatch), False if everything matched.
    """
    with open(json_file, 'r') as f:
        activity = json.load(f)

    report_id  = activity['user_activity_instance']['report_id']
    line_items = activity.get('line_items') or activity.get('user_activity_instance', {}).get('line_items', [])

    # Build the valid product list from planogram config
    valid_cleaned = [clean_name(p) for p in all_config_products]

    # Counters
    picked_counter = compute_picked_products(cv_activities)
    bought_counter = compute_bought_products(line_items, valid_cleaned)

    logger.info(f"Picked products : {picked_counter}")
    logger.info(f"Bought products : {bought_counter}")

    if picked_counter == bought_counter:
        logger.info("✅ Picked products match bought products. No action needed.")
        return False

    # --- Mismatch: build both downcharge and upcharge ops ---
    logger.info("⚠️  Mismatch detected. Building downcharge/upcharge operations...")

    base_url, machine_id, machine_token, machine_api_key = login.get_custom_machine_settings(VICKI_APP, logger)
    access_token = get_access_token(base_url, machine_id, machine_token, machine_api_key)

    downcharge_ops = build_downcharge_ops(line_items, picked_counter, valid_cleaned)

    # Only fetch the planogram if we might need it for an upcharge —
    # i.e. something was picked more than it was charged for.
    upcharge_ops = []
    needs_upcharge = any(
        picked_counter.get(p, 0) > bought_counter.get(p, 0)
        for p in valid_cleaned
    )

    if needs_upcharge:
        planogram_json = get_planogram(base_url, access_token)
        planogram_lookup = build_planogram_lookup(planogram_json)
        upcharge_ops = build_upcharge_ops(line_items, picked_counter, valid_cleaned, planogram_lookup)

    ops = downcharge_ops + upcharge_ops
    print(ops)
    if ops:
        body = {"line_items": ops}
        logger.info(f"Downcharge ops: {downcharge_ops}")
        logger.info(f"Upcharge ops: {upcharge_ops}")
        change_line_items(access_token, base_url, transaction_id, body)
        get_updated_line_items(base_url, transaction_id, access_token)
    else:
        logger.info("No downcharge/upcharge operations required (quantities already correct).")

    return True


