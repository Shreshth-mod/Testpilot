const { formatCurrency } = require("@monorepo/core");

function renderPrice(val) {
  return `<div>${formatCurrency(val)}</div>`;
}

module.exports = { renderPrice };
