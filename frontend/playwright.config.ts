import { defineConfig, devices } from '@playwright/test'

const baseURL = process.env.FILMMATCH_E2E_URL || 'http://127.0.0.1:8000'
const channel = process.env.FILMMATCH_BROWSER_CHANNEL

export default defineConfig({
  testDir: './tests',
  timeout: 25_000,
  use: { baseURL, channel, trace: 'retain-on-failure' },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile', use: { ...devices['Pixel 7'] } },
  ],
})
