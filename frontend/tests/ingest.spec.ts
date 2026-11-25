import { randomUUID } from 'node:crypto'
import { expect, test } from '@playwright/test'

test('authenticates, deduplicates, indexes, and recommends a zero-rating movie within 60 seconds', async ({ request }) => {
  const token = process.env.FILMMATCH_E2E_ADMIN_TOKEN
  test.skip(!token || test.info().project.name !== 'desktop', 'Run against the isolated Postgres test stack.')
  test.setTimeout(65_000)
  const key = randomUUID()
  const movie = { title: `Ingest check ${key}`, year: 2199, genres: ['Sci-Fi'],
    description: 'A hacker escapes a virtual simulation built by machines in a dystopian reality.' }
  const headers = { Authorization: `Bearer ${token}`, 'Idempotency-Key': key }
  const anchors = await (await request.get('/api/search?q=matrix')).json()
  const anchor = anchors.find((result: { year: number }) => result.year === 1999)
  expect(anchor).toBeTruthy()
  expect((await request.post('/api/admin/movies', { headers: { 'Idempotency-Key': key }, data: movie })).status()).toBe(401)
  const started = Date.now()
  const submission = await request.post('/api/admin/movies', { headers, data: movie })
  expect(submission.status()).toBe(202)
  const { id } = await submission.json()
  const retry = await request.post('/api/admin/movies', { headers, data: movie })
  expect((await retry.json()).id).toBe(id)
  const conflict = await request.post('/api/admin/movies', { headers, data: { ...movie, year: 2198 } })
  expect(conflict.status()).toBe(409)
  await expect.poll(async () => {
    const result = await (await request.get(`/api/search?q=${encodeURIComponent(movie.title)}`)).json()
    return result.map((match: { id: number }) => match.id)
  }, { timeout: 59_000 }).toEqual([id])
  const picks = await (await request.post('/api/recommend', { data: {
    ratings: [{ movie_id: anchor.id, stars: 5 }], year_from: movie.year, year_to: movie.year,
  } })).json()
  expect(picks.some((pick: { movie: { id: number; rating_count: number }; signal: string }) =>
    pick.movie.id === id && pick.movie.rating_count === 0 && pick.signal === 'content')).toBe(true)
  expect(Date.now() - started).toBeLessThan(60_000)
})
