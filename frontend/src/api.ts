export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const result = await fetch(`/api${path}`, options)
  if (!result.ok) throw new Error(result.status === 404 ? 'This report is pending publication.' : 'Could not load this request. Try again.')
  return result.json() as Promise<T>
}
