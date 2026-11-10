# Claude (@AnthropicAI) ile eğitim amaçlı kodladığım Exness algoritmik scalping botunu test ettim. Bot demo hesapta sorunsuz çalışıyor. Asıl soru şuydu: bu matematik para kazandırıyor mu?

Kısa cevap: Hayır. Ama nasıl öğrendiğimiz önemli..!

# Kuralı baştan koyduk: Hiçbir sonucu görmeden geçme kriterini sabitledik.

✅ En az 100 işlem
✅ Kâr faktörü > 1,2
✅ t-istatistiği > 2

# Parametreler eğitim verisinde seçildi, hiç görülmemiş veride sınandı (walk-forward).

Tick verisi, altın (XAUUSD), 8 gün:
30 işlem, −132 $.
Zararın 65 $'ı spread ve kayma, yani işlem başına 2,15 $ maliyet.
Eğitimde +44 $ kazandıran ayar, görmediği veride −47 $ kaybettirdi. Klasik aşırı uydurma.

Tersini açsak kazanır mıydık? diye de ölçtük.
Ters sistem: +3 $, t = 0,06. Yani başa baş.
Çünkü maliyet iki yönde de ödeniyor. Kaybettiren yön değil, spread'di.

# Sonra pairs trading denedik:

• 1 dakikalık, 6 ay, altın/gümüş ve altın/EURUSD: geçmedi
• Günlük, 10 yıl, 11 çift, swap dahil: hiçbiri elemeyi bile geçemedi

# Çıkan dersler:

• Perakende maliyetle kısa vadeli hareketler masrafı karşılamıyor
• Eğitimdeki kâr çoğu zaman tesadüf
• Kriteri sonuçtan önce koymazsan kendini kandırırsın

Sırada trend takibi var, aynı disiplinle. Yatırım tavsiyesi değildir.

---

# Aşağıda finansal piyasalarda en çok kullanılan matematik ve fizik kökenli formülleri listeledim. Sırası ile kullanacağım.

# Fiyat hareketini modelleyenler

1. Geometrik Brown hareketi (fizikteki Brown hareketinden): dS = μ·S·dt + σ·S·dW. Fiyatın rastgele ama oranlı değiştiğini varsayar; çoğu modelin temeli.

2. Black–Scholes denklemi (fizikteki ısı denkleminin bir biçimi): ∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S − rV = 0. Opsiyon fiyatlama için.

3. Ornstein–Uhlenbeck süreci: dX = θ(μ − X)dt + σ·dW. Ortalamaya dönüş. Botta var; tick ve pairs testlerinde avantaj çıkmadı.

4. + ADF / Engle–Granger testi: Δx*t = γ·x*{t−1} + ε, γ < 0 ise ortalamaya dönüş var demektir. Botta var.-------

5. Hurst üssü: E[R/S] ∝ n^H. H > 0,5 ise piyasa trendli, H < 0,5 ise ortalamaya dönüyor. Hangi stratejinin uygun olduğunu seçmek için kullanılır.

6. Lévy / kuvvet yasası kuyrukları (ekonofizik): P(|r| > x) ∝ x^(−α), α ≈ 3. Büyük hareketlerin normal dağılımın öngördüğünden çok daha sık olduğunu gösterir; risk hesabı için önemli.

7. Fokker–Planck denklemi (istatistiksel fizik): Getiri dağılımının zamanla nasıl değiştiğini tarif eder.

# Sinyal üretenler

8. Zaman serisi momentumu: sinyal = işaret(son 12 ayın getirisi), pozisyon = hedef_vol / σ_t. Sıradaki adayımız.

9. Carry (faiz farkı): Beklenen getiri ≈ iki para biriminin faiz farkı, yani pozitif swap. Düşük faizli parayla yüksek faizli parayı almak.

10. Kalman filtresi (mühendislik ve fizikte hedef takibi): Pairs’teki hedge oranını sabit varsaymak yerine sürekli günceller.

11. Fourier / dalgacık (wavelet) analizi: Fiyatı frekanslarına ayırıp döngü arar. Piyasada genelde zayıf kalır.

12. Shannon entropisi: H = −Σ p·log p. Piyasanın o an ne kadar öngörülebilir olduğunu ölçer.

# Piyasa mikroyapısı

13. - Microprice: (ask·V_bid + bid·V_ask) / (V_bid + V_ask). Botta var.-------

14. - VPIN: Bilgili emir akışının toksisitesini ölçer. Botta var.-------

15. Kyle lambdası: Δp = λ · emir akışı dengesizliği. Fiyat etkisini ölçer.

16. Hawkes süreci (deprem bilimindeki artçı sarsıntı modelinden): λ(t) = μ + Σ α·e^(−β(t − t_i)). Emirlerin birbirini tetiklemesini modeller.

17. Avellaneda–Stoikov: r = s − q·γ·σ²·(T − t). Piyasa yapıcı fiyatlama. Kurumsal maliyet gerektirir, perakendede pek işlemez.

# Risk ve pozisyon büyüklüğü

18. - Kelly kriteri: f* = p − q/b, sürekli hâli f* = μ/σ². Botta var.-------

19. GARCH(1,1): σ²*t = ω + α·ε²*{t−1} + β·σ²\_{t−1}. Oynaklık tahmini; pozisyon büyüklüğünü ayarlamak için çok faydalı.

20. Markowitz ortalama-varyans: max wᵀμ − (λ/2)·wᵀΣw. Portföy ağırlıklarını belirler.

21. - VaR / CVaR: Belirli bir olasılıkla en fazla ne kaybedeceğini ölçer. Botta var.------

22. Sharpe oranı ve deflated Sharpe (Bailey & López de Prado): SR = μ/σ. Deflated hâli, kaç strateji denendiyse o kadar yüksek bir eşik koyar. Çok formül deneyeceksek bu en önemlisi.


# Şimdiye kadarki tablo (önceden kayıtlı testler):

# Testler ve SonuçLar

+ Ornstein–Uhlenbeck (tick, 1 dk ve günlük pairs)	FAIL------
+ Test 1: Trend takibi	FAIL (maliyet öncesi bile avantaj yok)-----
+ Test 4: Carry	Exness’te pozitif swap olmadığı için çalıştırılmayacak------
