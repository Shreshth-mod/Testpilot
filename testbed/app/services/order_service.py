"""
Business logic and order calculation services for the TestPilot testbed.
Contains intentional, realistic boundary defects designed for Phase 1
automated prompt engineering and Spec-as-Oracle testing evaluation.
"""

from typing import Optional

from testbed.app.models import CartItem, OrderStatus, OrderTotals

TAX_RATE = 0.0825  # 8.25% Sales Tax
FREE_SHIPPING_THRESHOLD = 50.00
STANDARD_SHIPPING_FEE = 5.99

VALID_COUPONS = {
    "SAVE10": {"type": "percentage", "value": 0.10},
    "SAVE20": {"type": "percentage", "value": 0.20},
    "FLAT50": {"type": "fixed", "value": 50.00},
    "WELCOME5": {"type": "fixed", "value": 5.00},
}


class OrderService:
    """Core domain service responsible for calculating order totals and lifecycle transitions."""

    @staticmethod
    def calculate_subtotal(items: list[CartItem | dict]) -> float:
        """Calculate total sum of items in cart."""
        subtotal = 0.0
        for item in items:
            if isinstance(item, dict):
                price = item.get("unit_price", item.get("price", 0.0))
                qty = item.get("quantity", item.get("qty", 1))
            else:
                price = item.unit_price
                qty = item.quantity
            subtotal += price * qty
        return round(subtotal, 2)

    @staticmethod
    def calculate_discount(coupon_code: Optional[str], subtotal: float) -> float:
        """
        Calculate discount based on coupon rules.
        """
        if not coupon_code:
            return 0.0
        code = coupon_code.upper().strip()
        if code not in VALID_COUPONS:
            return 0.0

        rule = VALID_COUPONS[code]
        if rule["type"] == "percentage":
            discount = subtotal * rule["value"]
            return round(discount, 2)
        elif rule["type"] == "fixed":
            # BUG 1 (Discount Deficit Bug):
            # When fixed discount > subtotal, it returns full fixed value without clamping to subtotal,
            # which will propagate into negative net totals when calculating total payable amount!
            return min(subtotal, rule["value"])
        return 0.0

    @staticmethod
    def calculate_tax(taxable_amount: float) -> float:
        """
        Calculate state sales tax on the taxable amount.
        """
        if taxable_amount <= 0.0:
            return 0.0

        # BUG 2 (Tax Truncation Precision Bug):
        # Truncates instead of round-half-up, causing boundary cent calculation discrepancies.
        # e.g., $10.06 * 0.0825 = 0.82995 -> should round to 0.83, but int truncation makes it 0.82!
        truncated_tax = int(taxable_amount * TAX_RATE * 100) / 100.0
        return truncated_tax

    @classmethod
    def calculate_order_totals(cls, items: list[CartItem], coupon_code: Optional[str] = None) -> OrderTotals:
        """
        Calculate full breakdown of order charges: subtotal, discount, tax, shipping, and total.
        """
        subtotal = cls.calculate_subtotal(items)
        discount = cls.calculate_discount(coupon_code, subtotal)

        taxable_amount = max(0.0, subtotal - discount)
        tax = cls.calculate_tax(taxable_amount)

        if not items or subtotal <= 0.0:
            shipping = 0.0
        else:
            shipping = 0.0 if subtotal >= FREE_SHIPPING_THRESHOLD else STANDARD_SHIPPING_FEE

        # Net calculation inherits BUG 1 when discount exceeds subtotal
        total = max(0.0, round((subtotal - discount) + tax + shipping, 2))

        return OrderTotals(
            subtotal=subtotal,
            discount=discount,
            tax=tax,
            shipping=shipping,
            total=total,
        )

    @staticmethod
    def transition_order_status(current: OrderStatus, requested: OrderStatus) -> tuple[bool, Optional[str]]:
        """
        Validates state machine transitions.
        Valid transitions:
          PENDING -> PAID | CANCELLED
          PAID -> SHIPPED | CANCELLED
          SHIPPED -> DELIVERED
          DELIVERED -> COMPLETED
          CANCELLED -> [Terminal, no transitions permitted]
          COMPLETED -> [Terminal, no transitions permitted]
        """
        if current in (OrderStatus.CANCELLED, OrderStatus.COMPLETED):
            # BUG 3 (Illegal State Transition):
            # Logic error: erroneously allows CANCELLED orders to become COMPLETED directly!
            if requested == OrderStatus.COMPLETED:
                return True, None
            return False, f"Cannot transition from terminal state {current.value}"

        allowed_map = {
            OrderStatus.PENDING: [OrderStatus.PAID, OrderStatus.CANCELLED],
            OrderStatus.PAID: [OrderStatus.SHIPPED, OrderStatus.CANCELLED],
            OrderStatus.SHIPPED: [OrderStatus.DELIVERED],
            OrderStatus.DELIVERED: [OrderStatus.COMPLETED],
        }

        permitted = allowed_map.get(current, [])
        if requested in permitted:
            return True, None
        return False, f"Illegal transition from {current.value} to {requested.value}"
