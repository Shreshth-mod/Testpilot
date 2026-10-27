function formatCurrency(amount, currency = "USD") {
  return `${currency} ${Number(amount).toFixed(2)}`;
}

module.exports = { formatCurrency };
