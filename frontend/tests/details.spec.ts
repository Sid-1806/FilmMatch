import { expect, test } from '@playwright/test'

test('opens title details and shows distinct IMDb and TMDB scores', async ({ page }) => {
  await page.route('**/api/movies/*', route => route.fulfill({ json: {
    overview: 'A story about a lifelong friendship.',
    release_date: '1994-07-06',
    runtime_minutes: 142,
    directors: ['Example Director'],
    cast: ['Example Actor'],
    imdb_rating: 8.8,
    tmdb_rating: 8.4,
    imdb_url: 'https://www.imdb.com/title/tt0109830/',
    poster_url: null,
  } }))
  await page.goto('/')
  const card = page.getByRole('button', { name: 'Forrest Gump', exact: true })
  await card.click()
  const dialog = page.getByRole('dialog', { name: 'Forrest Gump' })
  await expect(dialog.getByText('A story about a lifelong friendship.')).toBeVisible()
  await expect(dialog.getByText('8.8/10')).toBeVisible()
  await expect(dialog.getByText('8.4/10')).toBeVisible()
  await expect(dialog.getByText('142 min')).toBeVisible()
  await expect(dialog.getByRole('link', { name: 'View on IMDb' })).toHaveAttribute('href', 'https://www.imdb.com/title/tt0109830/')
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
  await expect(card).toBeFocused()
})
