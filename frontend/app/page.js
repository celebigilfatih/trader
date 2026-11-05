"use client"

import { useEffect, useMemo, useState } from "react"

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"
const RSI_WARNING_DETAILS_STORAGE_KEY = "trader_rsi_warning_details_visible"

function useSessionStorage() {
  const [session, setSession] = useState(null)
  const [sessionEmail, setSessionEmail] = useState("")

  useEffect(() => {
    const token = window.localStorage.getItem("trader_session_token")
    const email = window.localStorage.getItem("trader_session_email")
    if (token) {
      setSession(token)
      setSessionEmail(email || "")
    }
  }, [])

  const saveSession = (token, email) => {
    window.localStorage.setItem("trader_session_token", token)
    window.localStorage.setItem("trader_session_email", email)
    setSession(token)
    setSessionEmail(email)
  }

  const clearSession = () => {
    window.localStorage.removeItem("trader_session_token")
    window.localStorage.removeItem("trader_session_email")
    setSession(null)
    setSessionEmail("")
  }

  return { session, sessionEmail, setSession: saveSession, clearSession }
}

export default function Home() {
  const { session, sessionEmail, setSession, clearSession } = useSessionStorage()
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE)
  const [status, setStatus] = useState("")

  const [email, setEmail] = useState("")
  const [magicCode, setMagicCode] = useState("")
  const [inviteToken, setInviteToken] = useState("")

  const [symbols, setSymbols] = useState([])
  const [watchlist, setWatchlist] = useState([])
  const [selectedSymbol, setSelectedSymbol] = useState("")

  const [newSymbol, setNewSymbol] = useState("")
  const [analysis, setAnalysis] = useState(null)
  const [portfolio, setPortfolio] = useState([])
  const [rsiRows, setRsiRows] = useState([])
  const [rsiSkippedSymbols, setRsiSkippedSymbols] = useState([])
  const [rsiWarnings, setRsiWarnings] = useState([])
  const [portfolioSymbol, setPortfolioSymbol] = useState("")
  const [portfolioQuantity, setPortfolioQuantity] = useState("")
  const [portfolioAverageCost, setPortfolioAverageCost] = useState("")

  const [loading, setLoading] = useState(false)

  const formatDateTime = (value) => {
    if (!value) return "-"
    const parsed = new Date(value)
    if (Number.isNaN(parsed.getTime())) return value
    return `${parsed.toLocaleDateString("tr-TR")} ${parsed.toLocaleTimeString("tr-TR")}`
  }

  const formatStaleness = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) return "-"
    const numeric = typeof value === "number" ? value : Number(value)
    if (Number.isNaN(numeric)) return "-"
    return `${numeric.toFixed(0)} sn`
  }

  const warningMetadataByCode = (code) => {
    switch (code) {
      case "market_data_unavailable":
        return {
          label: "Veri alınamadı",
          icon: "⚠️",
          tone: "soft",
        }
      case "insufficient_ohlcv_for_rsi":
        return {
          label: "RSI hesaplanamadı",
          icon: "⚠️",
          tone: "soft",
        }
      default:
        return {
          label: code || "Genel uyarı",
          icon: "ℹ️",
          tone: "info",
        }
    }
  }

  const warningToneStyle = (tone) => {
    if (tone === "soft") {
      return { color: "#854d0e", backgroundColor: "#fef3c7", borderColor: "#f59e0b" }
    }
    if (tone === "info") {
      return { color: "#0369a1", backgroundColor: "#e0f2fe", borderColor: "#0ea5e9" }
    }
    return { color: "#b91c1c", backgroundColor: "#fee2e2", borderColor: "#f87171" }
  }

  const [showRsiWarningDetails, setShowRsiWarningDetails] = useState(() => {
    try {
      const stored = window.localStorage.getItem(RSI_WARNING_DETAILS_STORAGE_KEY)
      if (stored === "true" || stored === "false") {
        return stored === "true"
      }
    } catch (_error) {
      // no-op: localStorage erişimi her zaman garanti değil
    }
    return false
  })

  useEffect(() => {
    try {
      window.localStorage.setItem(RSI_WARNING_DETAILS_STORAGE_KEY, String(showRsiWarningDetails))
    } catch (_error) {
      // no-op: localStorage erişimi her zaman garanti değil
    }
  }, [showRsiWarningDetails])

  const rsiWarningSummaryByCode = useMemo(() => {
    const byCode = {}
    for (const warning of rsiWarnings) {
      const code = warning?.code || "unknown"
      byCode[code] = (byCode[code] || 0) + 1
    }
    return Object.entries(byCode)
      .map(([code, count]) => ({
        ...warningMetadataByCode(code),
        code,
        count,
      }))
      .sort((left, right) => {
        if (right.count !== left.count) {
          return right.count - left.count
        }
        return left.label.localeCompare(right.label)
      })
  }, [rsiWarnings])

  const rsiSkippedSymbolsLabel = useMemo(() => {
    const uniqueSymbols = [...new Set(rsiSkippedSymbols)]
    uniqueSymbols.sort()
    return uniqueSymbols.join(", ")
  }, [rsiSkippedSymbols])

  const authHeaders = useMemo(
    () =>
      session
        ? {
            Authorization: `Bearer ${session}`,
            "Content-Type": "application/json",
          }
        : {},
    [session]
  )

  const runRequest = async (path, options = {}) => {
    const response = await fetch(`${apiBase}${path}`, {
      ...options,
      headers: {
        ...authHeaders,
        ...options.headers,
      },
    })
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(data?.message || data?.error_code || "İstek hatası")
    }
    return data
  }

  const fetchAuthRequiredData = async () => {
    if (!session) return
    setLoading(true)
    setStatus("Veriler yükleniyor...")
    try {
      const symbolsResponse = await runRequest("/api/v1/symbols")
      const wlResponse = await runRequest("/api/v1/watchlist")
      const portfolioResponse = await runRequest("/api/v1/portfolio")
      const list = symbolsResponse?.data?.symbols || []
      const wl = wlResponse?.data?.symbols || []
      const holdings = portfolioResponse?.data?.holdings || []
      setSymbols(list)
      setWatchlist(wl)
      setPortfolio(holdings)
      setSelectedSymbol(wl[0] || (list[0]?.symbol || ""))
      setStatus("Hazır")
    } catch (error) {
      setStatus(error.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (session) {
      fetchAuthRequiredData()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session])

  const handleInvite = async () => {
    if (!email) return
    try {
      const payload = await runRequest("/api/v1/auth/invite", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      })
      setInviteToken(payload?.data?.magic_link_token || "")
      setStatus("Demo invite token üretildi. API ekranından veya burada token alanına yapıştır.")
    } catch (error) {
      setStatus(error.message)
    }
  }

  const handleConsume = async () => {
    if (!magicCode) return
    setLoading(true)
    setStatus("Oturum açılıyor...")
    try {
      const response = await runRequest("/api/v1/auth/magic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: magicCode }),
      })
      const token = response?.data?.session_token
      const userEmail = response?.data?.email || ""
      if (!token) throw new Error("Token alınamadı")
      setSession(token, userEmail)
      setStatus("Oturum açıldı.")
      setMagicCode("")
    } catch (error) {
      setStatus(error.message)
    } finally {
      setLoading(false)
    }
  }

  const analyzeSymbol = async () => {
    if (!selectedSymbol) return
    setLoading(true)
    setStatus("Analiz çekiliyor...")
    try {
      const response = await runRequest(
        `/api/v1/analysis/${encodeURIComponent(selectedSymbol)}?period=6mo&interval=1d`
      )
      setAnalysis(response?.data || null)
      setStatus("Analiz hazır.")
    } catch (error) {
      setStatus(error.message)
    } finally {
      setLoading(false)
    }
  }

  const runRSIScreen = async () => {
    setLoading(true)
    setStatus("RSI tarayıcı çalışıyor...")
    try {
      const response = await runRequest("/api/v1/screeners/rsi", {
        method: "POST",
        body: JSON.stringify({ symbols: watchlist }),
      })
      setRsiRows(response?.data?.results || [])
      setRsiSkippedSymbols(response?.data?.skipped_symbols || [])
      setRsiWarnings(response?.data?.warnings || [])
      setStatus("RSI tarama tamam.")
    } catch (error) {
      setRsiRows([])
      setRsiSkippedSymbols([])
      setRsiWarnings([])
      setStatus(error.message)
    } finally {
      setLoading(false)
    }
  }

  const addWatchlistSymbol = async () => {
    if (!newSymbol) return
    try {
      const response = await runRequest(`/api/v1/watchlist/${encodeURIComponent(newSymbol)}/add`, {
        method: "POST",
      })
      setWatchlist(response?.data?.symbols || watchlist)
      setNewSymbol("")
      setStatus("Sembol eklendi.")
    } catch (error) {
      setStatus(error.message)
    }
  }

  const refreshPortfolio = async () => {
    if (!session) return
    const response = await runRequest("/api/v1/portfolio")
    setPortfolio(response?.data?.holdings || [])
  }

  const addPortfolioHolding = async () => {
    if (!portfolioSymbol || !portfolioQuantity || !portfolioAverageCost) return
    const quantity = Number(portfolioQuantity)
    const averageCost = Number(portfolioAverageCost)
    if (!Number.isFinite(quantity) || !Number.isFinite(averageCost) || quantity <= 0 || averageCost < 0) {
      setStatus("Geçerli adet ve ortalama maliyet girin.")
      return
    }
    setLoading(true)
    setStatus("Portföy ekleniyor...")
    try {
      await runRequest("/api/v1/portfolio", {
        method: "POST",
        body: JSON.stringify({
          symbol: portfolioSymbol.toUpperCase(),
          quantity,
          average_cost: averageCost,
        }),
      })
      await refreshPortfolio()
      setPortfolioSymbol("")
      setPortfolioQuantity("")
      setPortfolioAverageCost("")
      setStatus("Portföy kaydı eklendi.")
    } catch (error) {
      setStatus(error.message)
    } finally {
      setLoading(false)
    }
  }

  const removePortfolioHolding = async (symbol) => {
    setLoading(true)
    setStatus("Portföy satırı siliniyor...")
    try {
      await runRequest(`/api/v1/portfolio/${encodeURIComponent(symbol)}`, {
        method: "DELETE",
      })
      await refreshPortfolio()
      setStatus("Portföy satırı kaldırıldı.")
    } catch (error) {
      setStatus(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>Trader Cockpit (MVP)</h1>
      <p className="muted">FastAPI backend ile çalışan karar destek kabuğu</p>

      <div className="card">
        <h2>Genel</h2>
        <div className="row">
          <label>
            API Base URL:
            <input
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
              style={{ marginLeft: 8, width: 260 }}
            />
          </label>
          <span className="muted">{status || "Hazır"}</span>
        </div>
      </div>

      {!session ? (
        <div className="card">
          <h2>Giriş</h2>
          <div className="row">
              <input
                placeholder="email@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                data-testid="auth-email-input"
              />
              <button onClick={handleInvite} disabled={loading || !email}>
                Davet token üret
              </button>
          </div>
          {inviteToken && (
            <p className="muted">
              Demo token: <strong>{inviteToken}</strong>
            </p>
          )}
          <div className="row" style={{ marginTop: 8 }}>
            <input
              placeholder="invite token"
              value={magicCode}
              onChange={(e) => setMagicCode(e.target.value)}
              data-testid="auth-magic-input"
            />
            <button onClick={handleConsume} disabled={loading || !magicCode}>
              Magic Link kullan
            </button>
          </div>
        </div>
      ) : (
        <div className="card">
          <h2>Oturum</h2>
              <p className="muted">{sessionEmail || "Bilinen e-posta yok"} ({session.slice(0, 10)}...)</p>
          <button onClick={clearSession} data-testid="logout-button">
            Çıkış yap
          </button>
        </div>
      )}

      {session && (
        <>
          <div className="card">
            <h2>Watchlist</h2>
            <div className="row">
              <input
                placeholder="örn: THYAO.IS"
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                data-testid="watchlist-symbol-input"
              />
              <button onClick={addWatchlistSymbol} data-testid="watchlist-add-button">
                Ekle
              </button>
            </div>
            <p className="muted">
              {watchlist.length ? watchlist.join(", ") : "Watchlist boş"}
            </p>
          </div>

          <div className="card">
            <h2>Teknik analiz</h2>
            <div className="row">
              <select value={selectedSymbol} onChange={(e) => setSelectedSymbol(e.target.value)}>
                {watchlist.length
                  ? watchlist.map((symbol) => <option key={symbol}>{symbol}</option>)
                  : symbols.map((row) => <option key={row.symbol}>{row.symbol}</option>)}
              </select>
              <button onClick={analyzeSymbol} disabled={loading || !selectedSymbol}>
                Analizi getir
              </button>
              <button
                onClick={runRSIScreen}
                disabled={loading || !watchlist.length}
                data-testid="rsi-screen-button"
              >
                Watchlist RSI taraması
              </button>
            </div>
            {analysis && (
              <>
                <pre>{JSON.stringify(analysis, null, 2)}</pre>
              </>
            )}
          </div>

            <div className="card">
            <h2>RSI tarama sonuçları</h2>
            {rsiWarnings.length > 0 && (
              <div
                className="muted"
                data-testid="rsi-warning-panel"
                style={{
                  border: "1px solid #f59e0b",
                  borderRadius: 8,
                  padding: "10px 12px",
                  marginBottom: 12,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
                  <strong>
                    {rsiWarnings.length} sembol taramaya dahil edilemedi
                    {rsiSkippedSymbolsLabel ? ` (${rsiSkippedSymbolsLabel})` : ""}
                  </strong>
                  <button
                    onClick={() => setShowRsiWarningDetails((prev) => !prev)}
                    style={{
                      border: "1px solid #d97706",
                      color: "#92400e",
                      backgroundColor: "white",
                      borderRadius: 6,
                      padding: "4px 8px",
                    }}
                    data-testid="rsi-warning-toggle-button"
                  >
                    {showRsiWarningDetails ? "Detayları gizle" : "Detayları göster"}
                  </button>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
                  {rsiWarningSummaryByCode.map((summary) => (
                    <span
                      key={`${summary.code}-${summary.count}`}
                      style={{
                        border: "1px solid transparent",
                        borderRadius: 999,
                        padding: "4px 10px",
                        ...warningToneStyle(summary.tone),
                        fontSize: 12,
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                      }}
                    >
                      <strong>{summary.icon}</strong>
                      {summary.label} ({summary.count})
                    </span>
                  ))}
                </div>
                {showRsiWarningDetails && (
                  <ul
                    style={{ listStyleType: "none", paddingLeft: 0, marginTop: 10, marginBottom: 0 }}
                  >
                    {rsiWarnings.map((warning, index) => {
                      const metadata = warningMetadataByCode(warning?.code)
                      return (
                        <li
                          key={`${warning?.symbol || "warning"}-${index}`}
                          data-testid="rsi-warning-detail-item"
                          style={{
                            border: "1px solid transparent",
                            borderRadius: 8,
                            padding: "6px 10px",
                            marginBottom: 6,
                            display: "inline-block",
                            ...warningToneStyle(metadata.tone),
                          }}
                        >
                          <strong>
                            {metadata.icon} {warning?.symbol || "Sembol"}
                          </strong>
                          {warning?.code && (
                            <span style={{ margin: "0 6px", opacity: 0.8 }}>
                              [{metadata.label}]
                            </span>
                          )}
                          {warning?.reason || "Sebep belirtilmedi"}
                        </li>
                      )
                    })}
                  </ul>
                )}
              </div>
            )}
            <table data-testid="rsi-results-table">
              <thead>
                <tr>
                  <th>Sembol</th>
                  <th>RSI</th>
                  <th>Fiyat</th>
                  <th>Veri Zamanı</th>
                  <th>Kaynak</th>
                  <th>Çekilme Zamanı</th>
                  <th>Yaşlanma</th>
                </tr>
              </thead>
              <tbody>
                {rsiRows.map((row) => (
                  <tr key={row.symbol}>
                    <td>{row.symbol}</td>
                    <td>{row.rsi?.toFixed(2)}</td>
                    <td>{row.price?.toFixed(2)}</td>
                    <td>{formatDateTime(row.as_of)}</td>
                    <td>{row.source || "-"}</td>
                    <td>{formatDateTime(row.fetched_at)}</td>
                    <td>{formatStaleness(row.staleness_seconds)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!rsiRows.length && <p className="muted">Henüz tarama yapılmadı.</p>}
          </div>

          <div className="card">
            <h2>Paper portföy</h2>
            <div className="row">
              <input
                placeholder="Sembol"
                value={portfolioSymbol}
                onChange={(e) => setPortfolioSymbol(e.target.value.toUpperCase())}
                data-testid="portfolio-symbol-input"
              />
              <input
                placeholder="Miktar"
                value={portfolioQuantity}
                onChange={(e) => setPortfolioQuantity(e.target.value)}
                type="number"
                min="0"
                step="0.01"
                data-testid="portfolio-qty-input"
              />
              <input
                placeholder="Ort. Maliyet"
                value={portfolioAverageCost}
                onChange={(e) => setPortfolioAverageCost(e.target.value)}
                type="number"
                min="0"
                step="0.01"
                data-testid="portfolio-cost-input"
              />
              <button
                onClick={addPortfolioHolding}
                disabled={loading || !portfolioSymbol || !portfolioQuantity || !portfolioAverageCost}
                data-testid="portfolio-add-button"
              >
                Ekle
              </button>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Miktar</th>
                  <th>Ort. Maliyet</th>
                  <th>Güncelleme</th>
                  <th>İşlem</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.map((holding) => (
                  <tr key={holding.symbol}>
                    <td>{holding.symbol}</td>
                    <td>{holding.quantity}</td>
                    <td>{holding.average_cost}</td>
                    <td>{holding.updated_at}</td>
                    <td>
                      <button
                        onClick={() => removePortfolioHolding(holding.symbol)}
                        disabled={loading}
                        data-testid={`portfolio-delete-${holding.symbol}`}
                      >
                        Sil
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!portfolio.length && <p className="muted">Henüz portföy kaydı yok.</p>}
          </div>
        </>
      )}
    </div>
  )
}
