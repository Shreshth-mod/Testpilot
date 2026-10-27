const { formatCurrency } = require("../src/index");

test("formats USD currency correctly", () => {
  expect(formatCurrency(123.456)).toBe("USD 123.46");
});

test("handles custom currency symbol", () => {
  expect(formatCurrency(50, "EUR")).toBe("EUR 50.00");
});
