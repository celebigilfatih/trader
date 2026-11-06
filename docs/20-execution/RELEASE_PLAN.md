# Release Plan

## Güncel aday — M1 güvenilirlik tabanı

- **Version:** Henüz sürüm etiketi yok (`TBD`).
- **Target:** M1 çevrimdışı güvenilirlik kanıtlarını tamamlamak; kapalı beta
  M2 ve üretim M3 ayrı kalır. Yayın tarihi `TBD`.
- **Included outcomes:** Mevcut hesaplama, RSS/provenance, OTT ve ML dar
  düzeltmeleri sürümlendi; genel M1 kapanmadı. Kapsam/eksikler
  [M1 kapanış kontrolündedir](M1_CLOSEOUT.md).
- **Quality gates:** Mevcut M1 kabul ölçütleri, ilgili P0 testleri, veri niteliği
  ve hata görünürlüğü; sözdizimi/build etkisi, diff/belge/handoff ve ADR uyumu.
  CDSK upstream validator bu ortamda yoktur; yapı kontrolü onun yerine geçmiş
  sayılmaz. Yerel test sonucu uzak CI veya üretim onayı değildir.
- **Rollout:** Bu kayıt deployment yapmaz. Aynı çalışma dalına commit/push,
  yayın/merge veya gerçek kullanıcıya açılma anlamına gelmez.
- **Rollback:** Her uygulama diliminin sınırı ilgili execution/ADR kaydındadır.
  ADR-0007 öncesi sürümsüz model artifact'ı yeniden eğitim ister; uyumluluk ve
  kullanıcı verisi korunmadan toplu revert/restore yapılmaz. Üretim rollback ve
  backup/restore kanıtı M3'te açık kalır.

## Tarihsel M0 kapsamı

İlk plan CDSK dosyaları ve README CDSK bölümünün eklenmesiydi; o belgeleme
değişikliğinin geri alınabilirliği mevcut uygulamanın release rollback kanıtı
değildir. API/web ve sonraki güvenilirlik değişiklikleri artık repository'dedir.
