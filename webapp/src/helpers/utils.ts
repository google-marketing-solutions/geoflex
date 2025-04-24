export function assertIsError(e: unknown): asserts e is Error {
  if (!(e instanceof Error)) throw new Error('e is not an Error');
}

export function formatNumber(value: number): string {
  if (value === undefined || value === null) return 'N/A';
  if (value === 0) return '0';
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (!Number.isFinite(value)) return value as any;

  // Format based on the value range
  if (Math.abs(value) < 0.01) return value.toExponential(2);
  if (Math.abs(value) < 1) return value.toFixed(2);
  if (Math.abs(value) < 100) return value.toFixed(1);
  if (Math.abs(value) < 10000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toLocaleString(undefined, { notation: 'compact', maximumFractionDigits: 1 });
}
