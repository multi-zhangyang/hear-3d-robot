export interface MutationFence {
  runMutation<T>(operation: () => Promise<T>): Promise<T>;
}

export async function runFencedMutation<T>(
  fence: MutationFence | undefined,
  operation: () => Promise<T>
): Promise<T> {
  return fence ? fence.runMutation(operation) : operation();
}
