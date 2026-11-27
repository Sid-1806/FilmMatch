import { expect, test } from '@playwright/test'

test('finishes the mood quiz, fixes an incompatible answer, and starts fresh', async ({ page }) => {
  await page.goto('/?mood=1')
  await expect(page.getByRole('heading', { name: 'What are you in the mood for?' })).toBeVisible()
  await page.getByRole('button', { name: 'Lighter and in a better mood' }).click()
  await page.getByRole('button', { name: 'Back', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Lighter and in a better mood' })).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('button', { name: 'Lighter and in a better mood' }).click()
  await page.getByRole('button', { name: 'Keep it light' }).click()
  await page.getByRole('button', { name: 'Horror', exact: true }).click()
  for (let question = 4; question <= 11; question++) {
    await expect(page.getByText(`Question ${question} of 12`, { exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'No preference' }).click()
  }
  await page.getByRole('button', { name: 'Horror', exact: true }).click()
  await expect(page.getByText('No films match these filters.', { exact: false })).toBeVisible()
  await page.getByText('Edit answers', { exact: true }).click()
  await page.getByRole('listitem').filter({ hasText: 'Is there a genre you want tonight?' })
    .getByRole('button', { name: /Horror\s*Edit/ }).click()
  await page.getByRole('button', { name: 'Comedy', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Picks for tonight' })).toBeFocused()
  await expect(page.locator('.mood-results .pick-grid > li')).toHaveCount(10)
  await expect(page.locator('.mood-results .pick-grid').getByText('Tagged Comedy').first()).toBeVisible()
  await page.getByRole('button', { name: 'Start a new quiz' }).click()
  await expect(page.getByText('Question 1 of 12')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Lighter and in a better mood' })).toHaveAttribute('aria-pressed', 'false')
})

test('offers a superhero follow-up and clears it when the story answer changes', async ({ page }) => {
  await page.goto('/?mood=1')
  for (const choice of ['Buzzing with excitement', 'No preference', 'Action', 'No preference', 'A superhero story']) {
    await page.getByRole('button', { name: choice, exact: true }).click()
  }
  await expect(page.getByRole('heading', { name: 'Which superhero sounds right tonight?' })).toBeVisible()
  await page.getByRole('button', { name: 'Spider-Man', exact: true }).click()
  for (const choice of ['No preference', 'No preference', 'No preference', '2000s', 'Leave out animation', 'No preference']) {
    await page.getByRole('button', { name: choice, exact: true }).click()
  }
  await expect(page.getByRole('heading', { name: 'Spider-Man', exact: true })).toBeVisible()
  await expect(page.getByText('Title mentions Spider-Man').first()).toBeVisible()
  await page.getByText('Edit answers', { exact: true }).click()
  await page.getByRole('button', { name: /A superhero story\s*Edit/ }).click()
  await page.getByRole('button', { name: 'A mission that could go wrong' }).click()
  await expect(page.getByRole('heading', { name: 'Picks for tonight' })).toBeFocused()
  await page.getByText('Edit answers', { exact: true }).click()
  await expect(page.getByRole('listitem').filter({ hasText: 'Who would you like to spend the story with?' })
    .getByRole('button', { name: /No preference\s*Edit/ })).toBeVisible()
})
