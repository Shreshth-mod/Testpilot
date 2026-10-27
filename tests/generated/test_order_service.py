"""
TestPilot AI: Autonomous Boundary Test Suite for testbed/app/services/order_service.py
Prompt Technique: COT
Spec-as-Oracle Grounded Assertions targeting edge cases and boundary limits.
"""

import pytest
from pydantic import ValidationError
from testbed.app.models import CartItem, OrderStatus, OrderTotals
from testbed.app.services.order_service import (
    FREE_SHIPPING_THRESHOLD,
    STANDARD_SHIPPING_FEE,
    TAX_RATE,
    VALID_COUPONS,
    OrderService,
)

# Reasoning Trace Summary: The function `calculate_order_totals` needs to be thoroughly tested to ensure it handles various boundary conditions correctly. The boundary condition...

def test_empty_cart():
    from testbed.app.services.order_service import OrderService
    result = OrderService.calculate_order_totals(items=[], coupon_code='SAVE10')
    assert result.subtotal == 0.0
    assert result.discount == 0.0
    assert result.shipping == 0.0
    assert result.total == 0.0

def test_subtotal_below_free_shipping_threshold():
    from testbed.app.services.order_service import OrderService, CartItem
    result = OrderService.calculate_order_totals(items=[CartItem(item_id='item-1', name='Product', unit_price=49.99, quantity=1)], coupon_code='SAVE10')
    assert result.subtotal == 49.99
    assert result.discount == 5.0
    assert result.tax == 3.71
    assert result.shipping == 5.99
    assert result.total == 54.69

def test_subtotal_at_free_shipping_threshold():
    from testbed.app.services.order_service import OrderService, CartItem
    result = OrderService.calculate_order_totals(items=[CartItem(item_id='item-1', name='Product', unit_price=50.00, quantity=1)], coupon_code='SAVE10')
    assert result.subtotal == 50.00
    assert result.discount == 5.00
    assert result.tax == 3.71
    assert result.shipping == 0.0
    assert result.total == 48.71

def test_subtotal_above_free_shipping_threshold():
    from testbed.app.services.order_service import OrderService, CartItem
    result = OrderService.calculate_order_totals(items=[CartItem(item_id='item-1', name='Product', unit_price=50.01, quantity=1)], coupon_code='SAVE10')
    assert result.subtotal == 50.01
    assert result.discount == 5.0
    assert result.tax == 3.71
    assert result.shipping == 0.0
    assert result.total == 48.72

def test_valid_coupon():
    from testbed.app.services.order_service import OrderService, CartItem
    result = OrderService.calculate_order_totals(items=[CartItem(item_id='item-1', name='Product', unit_price=100.00, quantity=1)], coupon_code='SAVE20')
    assert result.subtotal == 100.00
    assert result.discount == 20.00
    assert result.tax == 6.60
    assert result.shipping == 0.0
    assert result.total == 86.60

def test_invalid_coupon():
    from testbed.app.services.order_service import OrderService, CartItem
    result = OrderService.calculate_order_totals(items=[CartItem(item_id='item-1', name='Product', unit_price=100.00, quantity=1)], coupon_code='UNKNOWN')
    assert result.subtotal == 100.00
    assert result.discount == 0.0
    assert result.tax == 8.25
    assert result.shipping == 0.0
    assert result.total == 108.25

def test_negative_price():
    import pytest
    from pydantic import ValidationError
    from testbed.app.models import CartItem
    with pytest.raises(ValidationError):
        CartItem(item_id='item-err', name='Bad', unit_price=-1.0, quantity=1)
