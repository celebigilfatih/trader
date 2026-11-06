// Test-only route fixtures for the real cockpit. No backend or provider calls.
import assert from "node:assert/strict"

const EMAIL = "fixture@example.com"
const MAGIC_TOKEN = "fixture-magic-token"
const SESSION_TOKEN = "fixture-session-token"
const PREFERENCE_KEY = "trader_rsi_warning_details_visible"
const AS_OF = "2026-11-05T15:20:00+00:00"
const FETCHED_AT = "2026-11-05T15:21:00+00:00"
const ROW = {
  symbol: "THYAO.IS", rsi: 35.25, price: 123.45,
  source: "test-fixture", as_of: AS_OF, fetched_at: FETCHED_AT, staleness_seconds: 60,
}
const WARNINGS = [
  { symbol: "GARAN.IS", code: "market_data_unavailable", reason: "Fixture: OHLCV alınamadı." },
  { symbol: "ASELS.IS", code: "insufficient_ohlcv_for_rsi", reason: "Fixture: RSI için veri yetersiz." },
]
const EMPTY = { count: 0, results: [], skipped_symbols: [], warnings: [] }
const SUCCESS = { ...EMPTY, count: 1, results: [ROW] }
const PARTIAL = { ...SUCCESS, skipped_symbols: ["GARAN.IS", "ASELS.IS"], warnings: WARNINGS }

function envelope(data, meta = {}) {
  return { data, meta: { correlation_id: "deterministic-ui-fixture", ...meta } }
}

async function afterClick(page, pathname, method, locator) {
  // Subscribe before the action: even an immediate fixture response is observed.
  const responsePromise = page.waitForResponse((response) =>
    new URL(response.url()).pathname === pathname && response.request().method() === method)
  const [response] = await Promise.all([responsePromise, locator.click()])
  await response.finished()
  return response
}

async function scan(page) {
  const response = await afterClick(page, "/api/v1/screeners/rsi", "POST",
    page.getByTestId("rsi-screen-button"))
  await page.waitForFunction(() =>
    !document.querySelector('[data-testid="rsi-screen-button"]').disabled)
  return response
}

async function login(page) {
  await page.getByTestId("auth-email-input").fill(EMAIL)
  await afterClick(page, "/api/v1/auth/invite", "POST",
    page.getByRole("button", { name: "Davet token üret", exact: true }))
  await page.getByText(MAGIC_TOKEN, { exact: true }).waitFor()
  await page.getByTestId("auth-magic-input").fill(MAGIC_TOKEN)
  await afterClick(page, "/api/v1/auth/magic", "POST",
    page.getByRole("button", { name: "Magic Link kullan", exact: true }))
  await page.getByText("Hazır", { exact: true }).waitFor()
  assert.equal(await page.getByTestId("rsi-screen-button").isEnabled(), true)
}

async function assertRows(page, expectedCount) {
  assert.equal(await page.getByTestId("rsi-results-table").locator("tbody tr").count(), expectedCount)
}

async function assertSuccessfulRow(page) {
  const table = page.getByTestId("rsi-results-table")
  const row = table.locator("tbody tr").filter({ hasText: "THYAO.IS" })
  await row.waitFor()
  assert.deepEqual(await table.locator("th").allTextContents(),
    ["Sembol", "RSI", "Fiyat", "Veri zamanı (son bar)", "Kaynak", "Veri niteliği", "Çekilme zamanı", "Veri yaşı (çekilme anında)"])
  assert.deepEqual(await row.locator("td").allTextContents(),
    ["THYAO.IS", "35.25", "123.45", "05.11.2026 18:20:00", "test-fixture", "Demo / sentetik veri — gerçek piyasa verisi değildir.", "05.11.2026 18:21:00", "60 sn"])
}

async function fixturePage(browser, frontBase, timeout, run, watchlist = ["THYAO.IS"]) {
  const context = await browser.newContext({ locale: "tr-TR", timezoneId: "Europe/Istanbul", serviceWorkers: "block" })
  context.setDefaultTimeout(timeout)
  const failures = []
  const state = { watchlist: [...watchlist], holdings: [], rsi: SUCCESS, scanCount: 0,
    analysisMeta: { source: "yahoo_finance", as_of: AS_OF, fetched_at: FETCHED_AT, staleness_seconds: 60 },
    analysisError: false, analysisGate: null }
  const origin = new URL(frontBase).origin
  await context.route("**/*", async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (!url.pathname.startsWith("/api/v1/")) {
      if (url.origin === origin) return route.continue()
      failures.push(`Dış istek: ${request.method()} ${url.origin}${url.pathname}`)
      return route.abort("blockedbyclient")
    }
    const method = request.method()
    const endpoint = `${method} ${url.pathname}`
    const json = (data, status = 200) => route.fulfill({
      status, contentType: "application/json", body: JSON.stringify(data),
      headers: { "Access-Control-Allow-Origin": "*" },
    })
    try {
      if (endpoint === "POST /api/v1/auth/invite") {
        assert.deepEqual(request.postDataJSON(), { email: EMAIL })
        return json(envelope({ email: EMAIL, magic_link_token: MAGIC_TOKEN }))
      }
      if (endpoint === "POST /api/v1/auth/magic") {
        assert.deepEqual(request.postDataJSON(), { token: MAGIC_TOKEN })
        return json(envelope({ email: EMAIL, session_token: SESSION_TOKEN, expires_at: "2099-01-01T00:00:00Z" }))
      }
      assert.equal(request.headers().authorization, `Bearer ${SESSION_TOKEN}`, endpoint)
      switch (endpoint) {
        case "GET /api/v1/symbols":
          return json(envelope({ symbols: ["THYAO.IS", "BIMAS.IS", "GARAN.IS", "ASELS.IS"].map((symbol) => ({ symbol, name: "Synthetic fixture" })) }))
        case "GET /api/v1/watchlist":
          return json(envelope({ symbols: state.watchlist }))
        case "POST /api/v1/watchlist/BIMAS.IS/add":
          state.watchlist = [...new Set([...state.watchlist, "BIMAS.IS"])]
          return json(envelope({ symbols: state.watchlist }))
        case "GET /api/v1/analysis/THYAO.IS":
        case "GET /api/v1/analysis/BIMAS.IS": {
          assert.equal(url.searchParams.get("period"), "6mo")
          assert.equal(url.searchParams.get("interval"), "1d")
          const reply = envelope({ symbol: url.pathname.split("/").at(-1), price: { close: ROW.price }, indicators: { rsi: ROW.rsi } }, state.analysisMeta)
          const error = state.analysisError
          if (state.analysisGate) await state.analysisGate
          return error ? json({ detail: { message: "Fixture analysis unavailable" } }, 503) : json(reply)
        }
        case "GET /api/v1/portfolio":
          return json(envelope({ holdings: state.holdings }))
        case "POST /api/v1/portfolio": {
          const holding = request.postDataJSON()
          assert.deepEqual(holding, { symbol: "THYAO.IS", quantity: 2, average_cost: 45.5 })
          state.holdings = [{ ...holding, updated_at: FETCHED_AT }]
          return json(envelope({ holding: state.holdings[0] }))
        }
        case "DELETE /api/v1/portfolio/THYAO.IS":
          state.holdings = []
          return json(envelope({ removed: true, symbol: "THYAO.IS" }))
        case "POST /api/v1/screeners/rsi":
          assert.deepEqual(request.postDataJSON(), { symbols: state.watchlist })
          state.scanCount += 1
          return state.rsi.error
            ? json({ detail: { error_code: "fixture_unavailable", message: "Fixture service unavailable" } }, 503)
            : json(envelope(state.rsi))
        default:
          throw new Error(`Beklenmeyen API isteği: ${endpoint}`)
      }
    } catch (error) {
      failures.push(error.message)
      return json({ message: "Fixture contract failed" }, 500)
    }
  })
  const page = await context.newPage()
  page.on("pageerror", (error) => failures.push(error.message))
  try {
    await page.goto(frontBase, { waitUntil: "networkidle" })
    await login(page)
    await run(page, state)
    assert.deepEqual(failures, [], "Fixture kontratı, dış ağ veya browser hatası")
  } finally {
    await context.close()
  }
}

export async function runDeterministicScenarios({ chromium, frontBase, timeout, browserChannel }) {
  const browser = await chromium.launch({ headless: true, channel: browserChannel })
  const run = async (name, test, watchlist) => {
    await fixturePage(browser, frontBase, timeout, test, watchlist)
    console.log(`[UI-DETERMINISTIC] PASS ${name}`)
  }
  try {
    await run("analiz kaynağı, veri niteliği ve eksik metadata", async (page, state) => {
      await page.getByText("Araştırma ve karar desteği içindir. Yatırım tavsiyesi değildir.", { exact: true }).waitFor()
      for (const [source, label] of [
        ["yahoo_finance", "Sağlayıcı verisi — Yahoo Finance"],
        ["mock", "Demo / sentetik veri — gerçek piyasa verisi değildir."],
        ["unverified-vendor", "Veri niteliği belirtilmedi"],
        [null, "Veri niteliği belirtilmedi"],
      ]) {
        state.analysisMeta = source === null ? {} : { source, as_of: AS_OF, fetched_at: FETCHED_AT, staleness_seconds: 60 }
        await afterClick(page, "/api/v1/analysis/THYAO.IS", "GET", page.getByRole("button", { name: "Analizi getir", exact: true }))
        const panel = page.getByTestId("analysis-provenance")
        await panel.getByText(label, { exact: true }).waitFor()
        await panel.getByText(source || "Kaynak belirtilmedi", { exact: true }).waitFor()
        if (source === null) {
          assert.equal(await panel.getByText("Kaydedilmemiş", { exact: true }).count(), 3)
        } else {
          await panel.getByText("05.11.2026 18:20:00", { exact: true }).waitFor()
          await panel.getByText("05.11.2026 18:21:00", { exact: true }).waitFor()
          await panel.getByText("60 sn", { exact: true }).waitFor()
        }
      }
      for (const malformed of [
        { as_of: true, fetched_at: 1, staleness_seconds: false },
        { as_of: "invalid", fetched_at: {}, staleness_seconds: " " },
        { as_of: [], fetched_at: true, staleness_seconds: [] },
      ]) {
        state.analysisMeta = { source: "yahoo_finance", ...malformed }
        await afterClick(page, "/api/v1/analysis/THYAO.IS", "GET", page.getByRole("button", { name: "Analizi getir", exact: true }))
        const panel = page.getByTestId("analysis-provenance")
        await page.getByText("Analiz hazır.", { exact: true }).waitFor()
        await panel.getByText("Sağlayıcı verisi — Yahoo Finance", { exact: true }).waitFor()
        assert.equal(await panel.getByText("Kaydedilmemiş", { exact: true }).count(), 3)
        assert.equal(await panel.getByText("0 sn", { exact: true }).count(), 0)
      }
    })

    await run("analiz hatası ve sembol değişimi eski sonucu temizler", async (page, state) => {
      const button = page.getByRole("button", { name: "Analizi getir", exact: true })
      await afterClick(page, "/api/v1/analysis/THYAO.IS", "GET", button)
      await page.locator("pre").waitFor()
      state.analysisError = true
      await afterClick(page, "/api/v1/analysis/THYAO.IS", "GET", button)
      await page.getByText("Analiz alınamadı. Yeniden deneyin.", { exact: true }).waitFor()
      assert.equal(await page.locator("pre").count(), 0)
      assert.equal(await page.getByTestId("analysis-provenance").count(), 0)
      state.analysisError = false
      await afterClick(page, "/api/v1/analysis/THYAO.IS", "GET", button)
      await page.locator("pre").waitFor()
      await page.locator("select").selectOption("BIMAS.IS")
      assert.equal(await page.locator("pre").count(), 0)
    }, ["THYAO.IS", "BIMAS.IS"])

    await run("geç gelen eski analiz yeni sembol sonucunu değiştirmez", async (page, state) => {
      let release
      state.analysisGate = new Promise((resolve) => { release = resolve })
      const started = page.waitForRequest((request) => new URL(request.url()).pathname === "/api/v1/analysis/THYAO.IS")
      await page.getByRole("button", { name: "Analizi getir", exact: true }).click()
      await started
      await page.locator("select").selectOption("BIMAS.IS")
      state.analysisGate = null
      await afterClick(page, "/api/v1/analysis/BIMAS.IS", "GET", page.getByRole("button", { name: "Analizi getir", exact: true }))
      await page.locator("pre").filter({ hasText: "BIMAS.IS" }).waitFor()
      const lateResponse = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/v1/analysis/THYAO.IS")
      release()
      await (await lateResponse).finished()
      await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))))
      assert.equal(JSON.parse(await page.locator("pre").innerText()).symbol, "BIMAS.IS")
      await page.getByTestId("logout-button").click()
      await page.getByTestId("auth-email-input").waitFor()
      await login(page)
      assert.equal(await page.locator("pre").count(), 0)
    }, ["THYAO.IS", "BIMAS.IS"])

    await run("RSI kaynak türleri ve eksik değerler", async (page, state) => {
      await page.getByText("Henüz tarama yapılmadı.", { exact: true }).waitFor()
      for (const [source, label] of [
        ["yahoo_finance", "Sağlayıcı verisi — Yahoo Finance"],
        ["demo", "Demo / sentetik veri — gerçek piyasa verisi değildir."],
        ["another-source", "Veri niteliği belirtilmedi"],
        [null, "Veri niteliği belirtilmedi"],
      ]) {
        state.rsi = { ...SUCCESS, results: [{ ...ROW, source, ...(source === null ? { as_of: null, fetched_at: null, staleness_seconds: null } : {}) }] }
        await scan(page)
        const row = page.getByTestId("rsi-results-table").locator("tbody tr")
        await row.getByText(label, { exact: true }).waitFor()
        if (source === null) {
          assert.equal(await row.getByText("Kaydedilmemiş", { exact: true }).count(), 3)
          await row.getByText("Kaynak belirtilmedi", { exact: true }).waitFor()
        }
      }
    })

    await run("çıkıştan önceki geç yanıt yeni oturuma taşınmaz", async (page, state) => {
      let release
      state.analysisGate = new Promise((resolve) => { release = resolve })
      try {
        const started = page.waitForRequest((request) => new URL(request.url()).pathname === "/api/v1/analysis/THYAO.IS")
        await page.getByRole("button", { name: "Analizi getir", exact: true }).click()
        await started
        await page.getByTestId("logout-button").click()
        await page.getByTestId("auth-email-input").waitFor()
        await login(page)
        const lateResponse = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/v1/analysis/THYAO.IS")
        release()
        await (await lateResponse).finished()
        await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))))
        assert.equal(await page.locator("pre").count(), 0)
        assert.equal(await page.getByTestId("analysis-provenance").count(), 0)
      } finally {
        release()
      }
    })

    await run("auth, watchlist, analiz, paper, RSI başarı ve logout", async (page, state) => {
      await page.getByRole("button", { name: "Analizi getir", exact: true }).waitFor()
      await afterClick(page, "/api/v1/analysis/THYAO.IS", "GET",
        page.getByRole("button", { name: "Analizi getir", exact: true }))
      await page.locator("pre").filter({ hasText: '"close": 123.45' }).waitFor()
      assert.deepEqual(JSON.parse(await page.locator("pre").innerText()),
        { symbol: "THYAO.IS", price: { close: 123.45 }, indicators: { rsi: 35.25 } })
      await scan(page)
      await assertSuccessfulRow(page)
      await page.getByTestId("watchlist-symbol-input").fill("bimas.is")
      await afterClick(page, "/api/v1/watchlist/BIMAS.IS/add", "POST", page.getByTestId("watchlist-add-button"))
      await page.getByText("THYAO.IS, BIMAS.IS", { exact: true }).waitFor()
      assert.deepEqual(state.watchlist, ["THYAO.IS", "BIMAS.IS"])
      await page.getByTestId("portfolio-symbol-input").fill("thyao.is")
      await page.getByTestId("portfolio-qty-input").fill("2")
      await page.getByTestId("portfolio-cost-input").fill("45.5")
      await afterClick(page, "/api/v1/portfolio", "POST", page.getByTestId("portfolio-add-button"))
      await page.getByTestId("portfolio-delete-THYAO.IS").waitFor()
      const holdingRow = page.locator("tr").filter({ has: page.getByTestId("portfolio-delete-THYAO.IS") })
      assert.deepEqual((await holdingRow.locator("td").allTextContents()).slice(0, 4),
        ["THYAO.IS", "2", "45.5", FETCHED_AT])
      await afterClick(page, "/api/v1/portfolio/THYAO.IS", "DELETE", page.getByTestId("portfolio-delete-THYAO.IS"))
      await page.getByText("Henüz portföy kaydı yok.", { exact: true }).waitFor()
      await page.getByTestId("logout-button").click()
      await page.getByTestId("auth-email-input").waitFor()
      assert.equal(await page.evaluate(() => localStorage.getItem("trader_session_token")), null)
      assert.equal(await page.getByTestId("rsi-screen-button").count(), 0)
    })

    await run("boş sonuç önceki satırları temizler", async (page, state) => {
      await scan(page)
      await assertSuccessfulRow(page)
      state.rsi = EMPTY
      await scan(page)
      await assertRows(page, 0)
      assert.equal(await page.getByTestId("rsi-warning-panel").count(), 0)
      await page.getByText("Tarama tamamlandı; sonuç bulunamadı.", { exact: true }).waitFor()
      assert.equal(state.scanCount, 2)
    })

    await run("kısmi veri ve açık/kapalı tercih yenilemede korunur", async (page, state) => {
      state.rsi = PARTIAL
      await scan(page)
      await assertSuccessfulRow(page)
      const panel = page.getByTestId("rsi-warning-panel")
      await panel.getByText("2 sembol taramaya dahil edilemedi (ASELS.IS, GARAN.IS)", { exact: true }).waitFor()
      await panel.getByText("Veri alınamadı (1)", { exact: false }).waitFor()
      await panel.getByText("RSI hesaplanamadı (1)", { exact: false }).waitFor()
      assert.equal(await page.getByTestId("rsi-warning-detail-item").count(), 0)
      for (const visible of [true, false]) {
        await page.getByTestId("rsi-warning-toggle-button").click()
        await page.waitForFunction(({ key, value }) => localStorage.getItem(key) === value,
          { key: PREFERENCE_KEY, value: String(visible) })
        assert.equal(await page.getByTestId("rsi-warning-detail-item").count(), visible ? 2 : 0)
        await page.reload({ waitUntil: "networkidle" })
        await page.getByText("Hazır", { exact: true }).waitFor()
        await scan(page)
        await page.getByTestId("rsi-warning-toggle-button").filter({ hasText: visible ? "Detayları gizle" : "Detayları göster" }).waitFor()
        assert.equal(await page.getByTestId("rsi-warning-detail-item").count(), visible ? 2 : 0)
        assert.equal(await page.evaluate((key) => localStorage.getItem(key), PREFERENCE_KEY), String(visible))
        if (visible) {
          for (const warning of WARNINGS) {
            await page.getByTestId("rsi-warning-detail-item").filter({ hasText: warning.reason }).waitFor()
          }
        }
      }
      assert.equal(state.scanCount, 3)
    }, ["THYAO.IS", "GARAN.IS", "ASELS.IS"])

    await run("tüm sembollerde veri yokluğu eski satırları temizler", async (page, state) => {
      await scan(page)
      await assertSuccessfulRow(page)
      state.rsi = { ...EMPTY, skipped_symbols: ["THYAO.IS"],
        warnings: [{ ...WARNINGS[0], symbol: "THYAO.IS" }] }
      await scan(page)
      await assertRows(page, 0)
      await page.getByText("Kullanılabilir tarama sonucu yok; atlanan sembolleri kontrol edin.", { exact: true }).waitFor()
      await page.getByTestId("rsi-warning-panel").getByText("1 sembol taramaya dahil edilemedi (THYAO.IS)", { exact: true }).waitFor()
      assert.equal(await page.getByTestId("rsi-warning-toggle-button").innerText(), "Detayları göster")
      await page.getByTestId("rsi-warning-toggle-button").click()
      await page.getByTestId("rsi-warning-detail-item").filter({ hasText: "Fixture: OHLCV alınamadı." }).waitFor()
    })

    await run("503 eski sonuç/uyarıyı temizler, yeniden deneme başarılı", async (page, state) => {
      state.rsi = PARTIAL
      await scan(page)
      await assertSuccessfulRow(page)
      await page.getByTestId("rsi-warning-panel").waitFor()
      state.rsi = { error: true }
      assert.equal((await scan(page)).status(), 503)
      await page.getByText("Tarama tamamlanamadı. Yeniden deneyin.", { exact: true }).first().waitFor()
      await assertRows(page, 0)
      assert.equal(await page.getByTestId("rsi-warning-panel").count(), 0)
      state.rsi = SUCCESS
      await scan(page)
      await assertSuccessfulRow(page)
      await page.getByText("RSI tarama tamam.", { exact: true }).waitFor()
      assert.equal(await page.getByText("Tarama tamamlanamadı. Yeniden deneyin.", { exact: true }).count(), 0)
      assert.equal(state.scanCount, 3)
    }, ["THYAO.IS", "GARAN.IS", "ASELS.IS"])
    console.log("[UI-DETERMINISTIC] 10/10 senaryo geçti; API yanıtları fixture, dış istek 0")
  } finally {
    await browser.close()
  }
}
