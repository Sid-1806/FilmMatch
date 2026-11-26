import { expect, test } from '@playwright/test'

test('compares two models and then edits one rating', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Model Lab', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Model Lab' })).toBeVisible()
  await page.getByRole('button', { name: 'Use example ratings' }).click()
  await page.getByRole('button', { name: 'Compare top ten' }).click()
  await expect(page.getByRole('table', { name: 'Top-ten changes' })).toBeVisible()
  await expect(page.getByRole('table', { name: 'Top-ten changes' }).getByRole('row')).not.toHaveCount(1)
  await page.locator('#compare-mode').selectOption('rating')
  await page.locator('#edit-stars').selectOption('1')
  await page.getByRole('button', { name: 'Compare top ten' }).click()
  await expect(page.getByRole('table', { name: 'Top-ten changes' })).toBeVisible()
})
