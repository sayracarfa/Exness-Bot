# Exness MT5 Algoritmik Scalping Bot

## Mimari

```
src/
├── universe.py        Terminaldeki tüm sembolleri bulur ve sınıflandırır (us_stock / metal / forex / other)
├── sessions.py        NYSE seans saatleri, tatiller, kapanıştan önce hisse pozisyonlarını kapatma
├── check_connection.py  Demo hesap ön kontrol scripti
├── portfolio_risk.py  Volatilite hedefleme + korelasyonlu portföy VaR
├── backtest.py        Tick bazlı backtest + walk-forward doğrulama (MT5'ten veri çekme dahil)
├── config.py          Pydantic ayarları (hesap, semboller, risk limitleri, eşikler)
├── models.py          Ortak veri modelleri (Side, SymbolSpec, PositionInfo)
├── mt5_client.py      Bloklayan MT5 API'sini tek thread'e serileştiren async wrapper + gecikme ölçümü
├── mt5_simulator.py   MT5 API'sini taklit eden simülatör (OU fiyat + 5 kademe DOM) — Linux/CI için
├── data_ingestion.py  Tick + DOM toplayıcı, mikro-fiyat  P = (Va·Pb + Vb·Pa)/(Va+Vb)
├── nlp_engine.py      RSS/takvim çekme, FinBERT skoru, S(t) = S0·e^(−λ(t−t0)), haber karartması
├── strategy.py        OU z-skoru, VPIN toksisite filtresi, duyarlılık katmanı → sinyal
├── risk_controller.py Kelly boyutlandırma, VaR, max drawdown, günlük zarar, devre kesici
├── order_manager.py   Emir iletimi, requote yeniden deneme, SL/TP temizleme, acil kapatma, paper mod
├── api.py             uvicorn ile /health ve /metrics
└── main.py            Ana asyncio döngüsü
```

Veri akışı: `DataIngestor → asyncio.Queue → SignalEngine → RiskController → OrderManager`.
`NLPEngine` arka planda sembol başına sönümlenen duyarlılık skorunu günceller; `RiskController`
her MT5 çağrısının gecikmesini izler ve eşik aşılırsa tüm pozisyonları kapatır.

## Kurulum

```bash
pip install -r requirements.txt        # Windows + MT5 terminali (canlı)
pip install -r requirements-dev.txt    # testler için
```

`MetaTrader5` paketi yalnızca **Windows**'ta çalışır. Diğer platformlarda `--simulate` kullanın.
`transformers`/`torch` kurulu değilse sözlük tabanlı yedek duyarlılık modeli otomatik devreye girer.

## Çalıştırma

```bash
python -m src.main --simulate --duration 60                 # simülatörle tam akış
python -m src.main --config config.example.json             # gerçek terminal, paper (dry-run)
python -m src.main --live --i-understand-the-risk           # GERÇEK emir
python -m pytest                                            # testler
curl localhost:8080/metrics
```

Varsayılan mod **dry-run**'dır: canlı fiyat akışı kullanılır, emirler yerel olarak simüle edilir.
Önce demo (Trial) hesapta çalıştırın.

## Enstrüman evreni (ABD hisseleri + metaller)

Varsayılan `universe.mode = "auto"`: bot açılışta terminaldeki tüm sembolleri okur ve
`asset_classes` (varsayılan `us_stock`, `metal`) sınıflarını seçer; `symbols` listesi her zaman eklenir.
Sınıflandırmayı ve seçilen listeyi görmek için:

```powershell
python -m src.universe          # özet + universe.csv
```

Hiç hisse seçilmezse `universe.csv` içindeki `path` sütununa bakıp `universe.stock_path_regex` değerini ayarlayın.

- Veri, tek tek sorgu yerine `copy_ticks_from` ile sembol başına imleçle toplu okunur; arada tick kaçmaz.
  Açılışta son `warmup_seconds` geçmişi modelleri ısıtır.
- Hisse haberleri ticker, şirket adı ve takma adlarla eşleşir (Apple/iPhone, Google/Alphabet...).
  Kısa ya da kelime gibi okunan tickerlar (`V`, `KO`, `ON`) yalnızca `$V` veya `(KO)` biçiminde eşleşir.
  Genel piyasa haberleri ("Wall Street", "S&P 500") tüm hisselere `market_beta_weight` ile yansır.
- Hisselerde seans dışında ve kapanışa 30 dk kala yeni işlem açılmaz; açık pozisyonlar kapanıştan 15 dk önce kapatılır.
  Tatil ve erken kapanış listesi (`sessions.holidays` / `early_closes`) her yıl güncellenmelidir.
- Risk: sınıf başına pozisyon limiti, ABD hisselerinde net yön limiti (hepsi aynı yöne gidemez)
  ve teminat kontrolü (`max_margin_usage`).

## Karar mantığına yapılan güvenlik eklemeleri

- **ADF (Dickey-Fuller) testi:** AR(1) tahmini rastgele seride bile b < 1 verdiği için eski filtre
  rastgele serilerin ~%84'ünü "ortalamaya dönüyor" sanıyordu. Artık t-istatistiği `adf_critical`
  (varsayılan −2,87, %5) altında değilse işlem yok (`no_stationarity`). Rastgele seride geçme oranı ~%5.
- **Zaman stopu:** Pozisyon `time_stop_half_lives` (varsayılan 3) yarı ömür içinde dönmezse kapatılır.
- **İşlem bazlı Kelly:** Kazanma olasılığı OU sürecinin "SL'den ve zaman stopundan önce TP'ye ulaşma"
  olasılığından Monte Carlo ile hesaplanır (en fazla 0,8 kabul edilir, `win_prob_haircut` kadar düşülür);
  ödeme oranı spread dahil gerçek TP/SL mesafesinden gelir. Beklenen değer ≤ 0 ise işlem açılmaz
  (`negative_ev`). Yeterli işlem birikince gerçekleşen Kelly, model Kelly'sini üstten sınırlar.

## Portföy riski (volatilite hedefleme + korelasyon)

Fiyatlar sabit süreli barlara (`portfolio.bar_seconds`, varsayılan 10 sn) bölünür.

- **Volatilite hedefleme:** Kısa (≈5 dk) ve uzun (≈1 saat) EWMA oynaklık karşılaştırılır.
  Pozisyon çarpanı `clamp(σ_uzun / σ_kısa, 0,25, 1)`: piyasa normalden oynaksa pozisyon küçülür,
  asla büyümez. `σ_kısa / σ_uzun > 2` ise yeni işlem açılmaz (oynaklık sıçraması). Oynaklık henüz
  bilinmiyorsa pozisyon yarıya indirilir.
- **Portföy VaR:** Açık pozisyonların hesap para birimindeki tutarları ve semboller arası kovaryans
  matrisiyle `VaR = z · √H · √(eᵀΣe)` hesaplanır (%99, 5 dk ufuk). Yeni işlem için VaR'ı
  `max_portfolio_var_pct` (varsayılan sermayenin %1'i) altında tutan en büyük lot karesel denklemle
  bulunur: aynı yönde korelasyonlu işlemler küçülür, riski azaltan ters yönlü işlemlere daha fazla izin verilir.
  Limit zaten aşılmışsa yalnızca toplam riski düşüren işlemler açılabilir.

Sıralama: Kelly → volatilite çarpanı → portföy VaR sınırı → teminat sınırı → sınıf limitleri.

## Backtest ve walk-forward

```powershell
python -m src.backtest fetch --symbol EURUSDm --days 10          # VPS'te: MT5'ten tick verisi -> data/
python -m src.backtest run --data data/EURUSDm.csv --commission 0 --slippage-ticks 1
python -m src.backtest walkforward --data data/EURUSDm.csv --folds 4
python -m src.backtest run --synthetic ou                        # verisiz deneme (sentetik)
```

Önemli olan **OUT-OF-SAMPLE** sonucudur: kâr faktörü > 1,2, t_stat > 2 ve yeterli işlem sayısı (≥ 100)
görmeden gerçek parayla çalıştırmayın. `predicted_win_prob` ile gerçekleşen `win_rate` arasındaki fark
modelin kalibrasyonunu gösterir; tahmin sistematik olarak yüksekse `win_prob_haircut` artırılmalı.
Hisseler için `--asset-class us_stock`, Raw/Zero hesaplarda komisyon için `--commission` kullanın.

## Hesap bilgilerini dosyadan vermek

Ortam değişkeni yerine bot klasöründe bir `config.json` oluşturabilirsiniz; varsa otomatik okunur
(Notepad'in UTF-8 BOM'u sorun değildir). Örnek:

```json
{
  "mt5": {
    "login": 12345678,
    "password": "islem-sifreniz",
    "server": "Exness-MT5Trial9"
  },
  "symbols": ["EURUSDm"]
}
```

Bu dosya şifre içerir: kimseyle paylaşmayın, buluta yüklemeyin.

## Pairs trading (eşbütünleşme + OU)

`src/pairs.py`: iki enstrümanın log-fiyat farkı `s = log Y − β·log X − α` üzerinde ortalamaya dönme.
Hedge oranı β ve Engle-Granger eşbütünleşme testi (%5 kritik değer −3,34) her `refit_bars` barda bir
yenilenir; spread OU ile modellenir, |z| ≥ z_giriş'te iki bacaklı pozisyon açılır. Her iki bacağın spread'i,
kayma ve komisyon dahildir; sinyaller sonraki barın açılış fiyatlarıyla gerçekleşir. Pozisyon, açıldığı
modelle yönetilir; para bazlı zarar kes ve bacak başına nominal tavan (sermayenin 3 katı) vardır.

```powershell
python -m src.backtest fetch --symbol XAGUSDm --days 10
python -m src.pairs walkforward --y data/XAUUSDm.csv --x data/XAGUSDm.csv --folds 4
python -m src.pairs run --synthetic coint      # verisiz doğrulama
```

Önceden belirlenmiş geçme şartı (değiştirilmez): out-of-sample ≥ 100 işlem, kâr faktörü > 1,2, t > 2.
Pairs şu an yalnızca backtest modülüdür; şartı geçen bir çift olursa canlı bota bağlanacaktır.

## Günlük pairs taraması (önceden kayıtlı protokol)

Sonuçlar görülmeden sabitlenen test: 16 ekonomik çift, günlük bar (10 yıla kadar), spread = gerçek tick ortancasının 2 katı,
1 tick kayma, Exness swap oranları (üç kat swap günü brokerdan). İlk %70 eleme (walk-forward: ≥8 işlem, PF>1,2, t>1),
son %30 kilitli test yalnızca bir kez; seçilen çiftlerin birleşik sonucu ≥30 işlem, PF>1,2, t>2 ise PASS.

```powershell
python -m src.pair_scan fetch-all --out data_d1      # MT5 açık ve giriş yapılmış olmalı
python -m src.pair_scan run --data data_d1           # tablo + pair_scan_report.json
python -m src.pair_scan run --synthetic coint        # kendi kendini doğrulama (PASS beklenir)
```

Not: test, lot yuvarlamasının sonucu bozmaması için büyük nominal sermayeyle (1.000.000) çalışır; PF ve t ölçekten
bağımsızdır. Küçük hesapta uygulanabilirlik ayrı bir sorudur. Protokol sabitleri `src/pair_scan.py` başındadır:
sonuçları gördükten sonra değiştirmeyin.

## Demo hesapta test

```powershell
$env:MT5_LOGIN="12345678"; $env:MT5_PASSWORD="..."; $env:MT5_SERVER="Exness-MT5Trial7"
python -m src.check_connection                     # salt-okuma kontroller
python -m src.check_connection --test-order        # 1 min-lot aç/kapa (yalnızca demo)
python -m src.main --config config.example.json --duration 600            # dry-run
python -m src.main --config config.example.json --live --i-understand-the-risk   # demo'da gerçek emir
```

`require_demo=true` (varsayılan) olduğu sürece bot gerçek (real) hesaba bağlanmayı reddeder.

## Önemli notlar

- **Gecikme:** MT5 Python API'sinde push callback yoktur; veriler yoklama (polling) ile alınır ve
  `order_send` terminal → broker sunucusu gidiş-dönüşü içerir. 50 ms devre kesici eşiğinin anlamlı
  olması için botu Exness sunucularına yakın bir Windows VPS'te çalıştırın. Tek bir anlık sıçramada
  her şeyi kapatmamak için varsayılan olarak **3 ardışık** ihlal gerekir (`latency_breach_count`;
  anında tetik için 1 yapın).
- **DOM:** Exness FX sembollerinde derinlik verisi sınırlı olabilir; yoksa mikro-fiyat orta fiyata düşer.
- **VPIN hacmi:** MT5 FX kotasyonları genelde işlem hacmi taşımaz; her tick 1 birim hacim sayılır.
- **Hard trip:** Maksimum drawdown ve günlük zarar limitleri devre kesiciyi kalıcı olarak açar;
  `RiskController.reset()` ile manuel sıfırlama gerekir. Gecikme/hata tetikleri cooldown sonrası kendiliğinden kapanır.
- Simülatör sonuçları bir backtest değildir; strateji kârlılığı hakkında bilgi vermez.
