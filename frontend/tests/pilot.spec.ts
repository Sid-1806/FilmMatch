import { expect, test } from '@playwright/test'

test('pilot link explains participation before a session starts', async ({ page }) => {
  await page.goto('/?pilot=1')
  await expect(page.getByRole('heading', { name: 'FilmMatch pilot' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Agree and start' })).toBeEnabled()
})

test('completes a paired pilot with a reload and an idempotent answer', async ({ page, request }) => {
  test.skip(!process.env.FILMMATCH_E2E_ADMIN_TOKEN || test.info().project.name !== 'desktop', 'Synthetic responses belong on the isolated Postgres test service.')
  test.setTimeout(50_000)
  const before = await (await request.get('/api/pilot/summary')).json()
  await page.goto('/?pilot=1')
  await page.getByRole('button', { name: 'Agree and start' }).click()
  await expect(page.getByText(/Round 1 of 2.*0 \/ 5/)).toBeVisible()
  const token = await page.evaluate(() => localStorage.getItem('filmmatch.pilot.five-prompts.v1'))
  expect(token).toBeTruthy()
  const headers = { 'X-Pilot-Session': token! }
  const initial = await (await request.get('/api/pilot/session', { headers })).json()
  const data = { action: 'rate', revision: initial.revision, movie_id: initial.prompt.id, stars: 5 }
  const first = await request.post('/api/pilot/event', { headers, data })
  const duplicate = await request.post('/api/pilot/event', { headers, data })
  expect((await duplicate.json()).revision).toBe((await first.json()).revision)
  expect((await duplicate.json()).answered).toBe(1)
  await page.reload()
  await expect(page.getByText(/Round 1 of 2.*1 \/ 5/)).toBeVisible()
  for (let i = 2; i <= 5; i++) {
    await page.getByRole('button', { name: 'Haven’t seen it' }).click()
    await expect(page.getByText(new RegExp(`Round 1 of 2.*${i} / 5`))).toBeVisible()
  }
  await page.locator('.pilot-pick').first().click()
  await page.getByRole('button', { name: 'Start round two' }).click()
  await expect(page.getByText(/Round 2 of 2.*0 \/ 5/)).toBeVisible()
  for (let i = 1; i <= 5; i++) {
    await page.getByRole('button', { name: 'Haven’t seen it' }).click()
    await expect(page.getByText(new RegExp(`Round 2 of 2.*${i} / 5`))).toBeVisible()
  }
  await page.getByRole('button', { name: 'None of these fit' }).click()
  await page.getByRole('button', { name: 'No preference' }).click()
  await expect(page.getByRole('heading', { name: 'Thanks for taking part' })).toBeVisible()
  const after = await (await request.get('/api/pilot/summary')).json()
  expect(after.started - before.started).toBe(1)
  expect(after.paired - before.paired).toBe(1)
  expect(after.completed - before.completed).toBe(1)
  expect(after.preferences.tie - (before.preferences?.tie ?? 0)).toBe(1)
})
