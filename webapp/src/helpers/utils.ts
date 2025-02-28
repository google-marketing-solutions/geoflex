export function assertIsError(e: unknown): asserts e is Error {
  if (!(e instanceof Error)) throw new Error('e is not an Error');
}
