import { multiply, power } from "../src/math";

describe("Math functions", () => {
  test("multiplies positive integers correctly", () => {
    expect(multiply(3, 4)).toBe(12);
  });

  it("calculates exponential power", () => {
    expect(power(2, 3)).toBe(8);
  });

  test("handles zero multiplier", () => {
    expect(multiply(5, 0)).toBe(0);
  });
});
