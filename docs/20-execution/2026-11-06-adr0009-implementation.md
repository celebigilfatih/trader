# ADR-0009 implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` task by task; checkboxes track verification.

**Goal:** Accepted ADR-0009 ile OS-01 HTTP500 hatasını, mevcut `stoch` skalerini
gerçek %K'ya eşleyerek düzeltmek.

**Architecture:** Yalnız `analyze_symbol` sonuç projeksiyonunda `stoch` →
`stoch_k`; gerçek analyzer/ta hesapları ve diğer seçim yolları korunur.

**Tech Stack:** Mevcut Python/FastAPI/unittest/pandas/ta; bağımlılık değişmez.

**Spec:** [ADR-0009](../50-decisions/ADR-0009.md),
[tasarım/kabul ölçütleri](2026-11-06-stochastic-api-design.md).

## Global constraints

- Somut `stoch=%K` sorusuna “Sıradaki Adım ile devam edelim.” yanıtı bu A
  uygulamasına onay kaydedildi; önceki genel devam taleplerine uygulanmaz.
- Dış `stoch` alanı float/null kalır; K/D query/response alanı eklenmez.
- Legacy/core formül/config, `_safe_float`, auth/provider/metadata/diğer
  seçiciler değişmez. Genel `.get(...,None)` ile hata gizlenmez.
- Mevcut codex checkout/dalında devam edilir; kullanıcı değişiklikleri korunur.
- OS-01 dar dilimi kapanabilir; genel M1 ve finansal etkinlik kapanışı değildir.

## Review focus

- %K≠%D fixture'ında son K75 döner; D200/3 veya ilk/önceki K seçilmez.
- Gerçek0, warmup/flat NaN ve null ayrımı HTTP JSON seviyesinde korunur.
- Karma seçimde yalnız istenen anahtarlar, diğer scalar/null, fiyat/meta/correlation korunur.
- K/D query adları400, kimliksiz/geçersiz oturum401 kalır.
- Yalnız `stoch` için alias; core/legacy getter ve başka çoklu çıktılar değiştirilmez.

## Task 1: Stochastic HTTP regresyonu ve dar eşleme

**Files:** Yeni `backend/tests/test_stochastic_api.py`,
`backend/app/routers.py`; ADR/index, API konvansiyonları, execution, kapanış,
backlog, milestone, mimari/kalite ve Boot/CHANGELOG güncellemesi.

**Interfaces:** Mevcut `/api/v1/analysis/{symbol}?indicators=stoch` ve
`data.indicators.stoch: float | null`; yeni public fonksiyon/alan yok.

- [x] Gerçek HTTP router/analyzer/auth + fixture provider ile7 test yaz:
  K75≠D200/3, K0,13-bar warmup-null,20-bar flat-null, mixed
  `stoch,williams_r,cci,macd` → `75,-25,null,null` ve metadata;
  K/D query400, eksik/geçersiz oturum401. Dış socket engellenir; loopback serbest.
- [x] `.venv/Scripts/python.exe -B -m unittest backend.tests.test_stochastic_api -v`
  komutunu Windows socketpair için sandbox dışında çalıştır.
  Expected: başarı bekleyen5 test HTTP500 nedeniyle assertion failure; error yok.
- [x] Sonuç döngüsünde yalnız stoch seçicisinin seri adı stoch_k olsun.
  Aynı komut Expected:7/7 OK. Ardından mevcut API/momentum hedefi de OK.
- [x] Alias kaldırma/D/önceki bar/alan adı/null-zero mutantlarını assertion
  failure ile yakala, restore green; production kaynağı diskte mutate edilmez.
- [x] Tam backend dış socket engelli; compile/CDSK yapı/link/ADR/diff kontrolü.
  Expected: bütün testler OK; frontend/canlı smoke tekrar yokluğu açık raporlanır.
- [x] Bağımsız son inceleme ve belgeler/Boot handoff'u tamamla.

**Teslim:** Önceki açık commit/push yetkisiyle aynı dal; SHA/remote/clean tree
kapanışta doğrulanır. Test sayısı M1 kabul eşiği değildir.

## Evidence

- 7 yeni HTTP testi gerçek router/analyzer/ta/auth ile çalışır; yalnız provider
  fixture ve test state'i kontrollüdür. Dış socket engellenir, asyncio loopback serbest.
- RED:7 testte5 assertion failure/0 error, HTTP500≠200. Dar2 satırlık API
  alias'ından sonra GREEN7/7 (0.092 s); diğer seçiciler/dependency/legacy aynı.
- Mevcut API+momentum+yeni HTTP hedefi38/38 (0.337 s); tam backend
  **373 test /27.902 s /OK**, dış socket engelli. Windows TestClient/AppTest
  gereksinimiyle sandbox dışında; frontend/canlı/uzak CI tekrar koşulmadı.
- 10 geçerli transient mutant alias kaldırma, D, bütün seçicilere alias,
  response alanı, önceki/ilk bar, zero/null, alan atlama/ek D'yi bozdu;
  tamamı assertion failure,0 error. Restore7/7 (0.069 s). İlk lazy-router lookup
  ve mock descriptor restore hataları harness hatasıdır, başarı sayılmadı.
  Mutasyon yalnız endpoint'in bellek içi code nesnesindeydi, try/finally ile
  geri yüklendi; üretim dosyası diskte mutate edilmedi.
- 64 Python compile/39 zorunlu CDSK dosyası/11 Boot/9 Accepted ADR ve yerel
  bağlantılar/diff kontrolü geçti. Upstream validator ortamda yok; yapı kontrolü
  onun yerine geçti sayılmaz. Production kapsamı yalnız routers.py alias'ıdır.
- OS-01 dar dilimi kapandı; genel M1 ve diğer çoklu çıktıların kapsamı açık.
  Sonraki iş VWEMA/Supertrend ve aktif okuyucu bağlantıları.
- Bağımsız son inceleme eyleme dönük bulgu bulmadı; reviewer7 HTTP testini
  **0.088 s /OK** yeniden çalıştırdı. Alias/son bar,zero/null,mixed/metadata,
  cleanup ve Accepted/kapsam durumu doğrulandı. RED/mutant/tam373 koşuları
  controller kanıtıdır; reviewer bunları tekrar koşmadı. Sohbet alıntısı somut
  soru/yanıt bağlamıyla incelendi; tüm sohbetin ayrıca tekrar incelendiği iddia edilmez.
