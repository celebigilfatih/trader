#!/usr/bin/env node
import { existsSync } from "node:fs"
import path from "node:path"
import { pathToFileURL } from "node:url"

const args = process.argv.slice(2)

const parsed = {
  apiBase: "http://127.0.0.1:8010",
  frontBase: "http://127.0.0.1:3010",
  timeoutSeconds: 20,
  symbol: "THYAO.IS",
  email: "smoke_test_ui@example.com",
  quantity: "2",
  averageCost: "45.5",
}
const RSI_WARNING_DETAILS_STORAGE_KEY = "trader_rsi_warning_details_visible"

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--api-base") parsed.apiBase = args[i + 1] || parsed.apiBase
  if (args[i] === "--front-base") parsed.frontBase = args[i + 1] || parsed.frontBase
  if (args[i] === "--timeout") parsed.timeoutSeconds = Number(args[i + 1] || parsed.timeoutSeconds)
  if (args[i] === "--symbol") parsed.symbol = args[i + 1] || parsed.symbol
  if (args[i] === "--email") parsed.email = args[i + 1] || parsed.email
}

function log(message) {
  const timestamp = new Date().toISOString().replace(/T/, " ").replace(/\..+/, "")
  console.log(`[UI-SMOKE] ${timestamp} ${message}`)
}

async function waitForUrl(url, timeoutSeconds) {
  const deadline = Date.now() + timeoutSeconds * 1000
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url)
      if (response.status === 200) {
        return true
      }
    } catch (_error) {
      // bekleniyor
    }
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  return false
}

async function requestJson(url, method, payload, headers = {}) {
  const response = await fetch(url, {
    method,
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      ...headers,
    },
    body: payload ? JSON.stringify(payload) : undefined,
  })

  const text = await response.text()
  const data = text ? safeJson(text) : {}
  return { status: response.status, data }
}

function exitWithFailure(message) {
  log(`Hata: ${message}`)
  process.exit(1)
}

function safeJson(response) {
  try {
    return response ? JSON.parse(response) : {}
  } catch (_error) {
    return {}
  }
}

async function waitForLocalStorageValue(page, key, expected, timeoutSeconds) {
  const deadline = Date.now() + timeoutSeconds * 1000
  while (Date.now() < deadline) {
    const currentValue = await page.evaluate((storageKey) => window.localStorage.getItem(storageKey), key)
    if (currentValue === expected) {
      return true
    }
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  return false
}

let chromium
try {
  const playwrightPackage = await import("playwright")
  chromium = playwrightPackage.chromium
} catch (_error) {
  const fallbackPath = path.resolve(process.cwd(), "frontend/node_modules/playwright")
  if (!existsSync(fallbackPath)) {
    log("Playwright bağımlılığı bulunamadı. Kurulum: npm i -D playwright && npx playwright install")
    process.exit(0)
  }
  try {
    const fallbackEsModulePath = path.join(fallbackPath, "index.mjs")
    log(`Playwright fallback yükleme denemesi: ${fallbackEsModulePath}`)
    const playwrightModuleUrl = pathToFileURL(fallbackEsModulePath).href
    const playwrightPackage = await import(playwrightModuleUrl)
    chromium = playwrightPackage.chromium
  } catch (_fallbackError) {
    log(
      `Playwright modülü bulundu ancak yüklenemedi: ${_fallbackError?.message || _fallbackError}. Kurulum: npm i -D playwright && npx playwright install`
    )
    process.exit(0)
  }
}

const readyBackend = await waitForUrl(`${parsed.apiBase}/api/v1/health`, parsed.timeoutSeconds)
if (!readyBackend) {
  exitWithFailure(`Backend hazır değil: ${parsed.apiBase}`)
}

const readyFrontend = await waitForUrl(parsed.frontBase, parsed.timeoutSeconds)
if (!readyFrontend) {
  exitWithFailure(`Frontend hazır değil: ${parsed.frontBase}`)
}

const invite = await requestJson(
  `${parsed.apiBase}/api/v1/auth/invite`,
  "POST",
  { email: parsed.email }
)

if (invite.status !== 200) {
  exitWithFailure(`Invite isteği başarısız: status=${invite.status}`)
}

const token = invite.data?.data?.magic_link_token
if (!token) {
  exitWithFailure("Invite token bulunamadı")
}

const magic = await requestJson(
  `${parsed.apiBase}/api/v1/auth/magic`,
  "POST",
  { token }
)

if (magic.status !== 200) {
  exitWithFailure(`Magic token tüketimi başarısız: status=${magic.status}`)
}

const sessionToken = magic.data?.data?.session_token
if (!sessionToken) {
  exitWithFailure("Session token alınamadı")
}

const browser = await chromium.launch({ headless: true })
const context = await browser.newContext()

await context.addInitScript(
  ({ tokenValue, emailValue }) => {
    localStorage.setItem("trader_session_token", tokenValue)
    localStorage.setItem("trader_session_email", emailValue)
  },
  {
    tokenValue: sessionToken,
    emailValue: parsed.email,
  }
)

const page = await context.newPage()

await page.goto(parsed.frontBase, { waitUntil: "networkidle" })

try {
  await page.waitForSelector('text="Trader Cockpit (MVP)"', { timeout: 5000 })
} catch (_error) {
  await browser.close()
  exitWithFailure("Frontend başlığı bulunamadı")
}

try {
  await page.getByTestId("watchlist-symbol-input").fill(parsed.symbol)
  await Promise.all([
    page.waitForResponse((response) => {
      return (
        response.url().includes(`/api/v1/watchlist/${encodeURIComponent(parsed.symbol)}/add`) &&
        response.request().method() === "POST" &&
        response.status() === 200
      )
    }),
    page.getByTestId("watchlist-add-button").click(),
  ])
  await page.getByText(parsed.symbol).first().waitFor({ timeout: 5000 })
} catch (_error) {
  await browser.close()
  exitWithFailure("Watchlist ekleme UI akışı başarısız")
}

try {
  const rsiRequest = page.waitForResponse((response) => {
    return (
      response.url().includes("/api/v1/screeners/rsi") &&
      response.request().method() === "POST" &&
      response.status() === 200
    )
  })

  await page.getByTestId("rsi-screen-button").click()
  const response = await rsiRequest
  const payload = await response.json().catch(() => ({}))

  if (typeof payload?.data?.count !== "number") {
    await browser.close()
    exitWithFailure("RSI tarama yanıtı beklenen biçimde gelmedi")
  }

  if (!Array.isArray(payload?.data?.skipped_symbols)) {
    await browser.close()
    exitWithFailure("RSI tarama yanıtında skipped_symbols alanı beklenen biçimde gelmedi")
  }

  if (!Array.isArray(payload?.data?.warnings)) {
    await browser.close()
    exitWithFailure("RSI tarama yanıtında warnings alanı beklenen biçimde gelmedi")
  }

  for (const header of [
    "Sembol",
    "RSI",
    "Fiyat",
    "Veri Zamanı",
    "Kaynak",
    "Çekilme Zamanı",
    "Yaşlanma",
  ]) {
    await page.locator(`th:has-text("${header}")`).first().waitFor({ timeout: 5000 })
  }

  if (payload.data.count > 0 && Array.isArray(payload.data.results) && payload.data.results.length) {
    const firstSymbol = payload.data.results[0]?.symbol
    if (firstSymbol) {
      await page
        .getByTestId("rsi-results-table")
        .getByText(firstSymbol, { exact: false })
        .first()
        .waitFor({ timeout: 5000 })
    }
  }

  const warningPanel = page.getByTestId("rsi-warning-panel")
  if ((await warningPanel.count()) > 0) {
    const warningToggle = page.getByTestId("rsi-warning-toggle-button")

    const toggleText = (await warningToggle.innerText()).trim()
    if (toggleText === "Detayları göster") {
      await warningToggle.click()
    }

    await page.getByTestId("rsi-warning-detail-item").first().waitFor({ timeout: 5000 })

    const warningPreference = await page.evaluate(
      (key) => window.localStorage.getItem(key),
      RSI_WARNING_DETAILS_STORAGE_KEY
    )
    if (warningPreference !== "true") {
      await browser.close()
      exitWithFailure("RSI uyarı detay tercihi localStorage'da kalıcı olarak saklanmadı")
    }

    await page.reload({ waitUntil: "networkidle" })

    const persistedPreference = await waitForLocalStorageValue(page, RSI_WARNING_DETAILS_STORAGE_KEY, "true", 5)
    if (!persistedPreference) {
      await browser.close()
      exitWithFailure("RSI uyarı detay tercihi sayfa yenilendiğinde korunmadı")
    }

    await page.getByTestId("rsi-screen-button").click()
    const rsiRequestAfterRefresh = page.waitForResponse((response) => {
      return (
        response.url().includes("/api/v1/screeners/rsi") &&
        response.request().method() === "POST" &&
        response.status() === 200
      )
    })
    const responseAfterRefresh = await rsiRequestAfterRefresh
    const payloadAfterRefresh = await responseAfterRefresh.json().catch(() => ({}))

    if (typeof payloadAfterRefresh?.data?.count !== "number") {
      await browser.close()
      exitWithFailure("Yenilenmiş sayfada RSI tarama yanıtı beklenen biçimde gelmedi")
    }

    if (
      Array.isArray(payloadAfterRefresh?.data?.warnings) &&
      payloadAfterRefresh.data.warnings.length > 0
    ) {
      await page.getByTestId("rsi-warning-panel").waitFor({ timeout: 5000 })
      const toggleAfterReload = await page.getByTestId("rsi-warning-toggle-button").innerText()
      if (toggleAfterReload.trim() !== "Detayları gizle") {
        await browser.close()
        exitWithFailure("RSI uyarı paneli yenilenmiş oturumda açık kalmadı")
      }
      await page.getByTestId("rsi-warning-detail-item").first().waitFor({ timeout: 5000 })
    }
  }
} catch (_error) {
  await browser.close()
  exitWithFailure("RSI tarama UI akışı başarısız")
}

try {
  const portfolioSymbolInput = page.getByTestId("portfolio-symbol-input")
  const portfolioQuantityInput = page.getByTestId("portfolio-qty-input")
  const portfolioCostInput = page.getByTestId("portfolio-cost-input")

  await portfolioSymbolInput.fill(parsed.symbol)
  await portfolioQuantityInput.fill(parsed.quantity)
  await portfolioCostInput.fill(parsed.averageCost)

  await Promise.all([
    page.waitForResponse((response) => {
      return (
        response.url().endsWith("/api/v1/portfolio") &&
        response.request().method() === "POST" &&
        response.status() === 200
      )
    }),
    page.getByTestId("portfolio-add-button").click(),
  ])

  await page.getByTestId(`portfolio-delete-${parsed.symbol}`).waitFor({ timeout: 5000 })
} catch (_error) {
  await browser.close()
  exitWithFailure("Portföy ekleme UI akışı başarısız")
}

try {
  await Promise.all([
    page.waitForResponse((response) => {
      return (
        response.url().includes(`/api/v1/portfolio/${encodeURIComponent(parsed.symbol)}`) &&
        response.request().method() === "DELETE" &&
        response.status() === 200
      )
    }),
    page.getByTestId(`portfolio-delete-${parsed.symbol}`).click(),
  ])

  await page.getByTestId(`portfolio-delete-${parsed.symbol}`).waitFor({
    state: "detached",
    timeout: 5000,
  })
} catch (_error) {
  await browser.close()
  exitWithFailure("Portföy silme UI akışı başarısız")
}

await page.getByTestId("logout-button").click()
await browser.close()
log("UI smoke akışı başarılı")
process.exit(0)
