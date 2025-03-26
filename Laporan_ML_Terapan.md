# Laporan Submission Machine Learning Terapan - Aswin Setiawan

## Domain Proyek

**Latar Belakang:** Berinvestasi dalam saham adalah salah satu instrumen keuangan yang banyak diminati oleh masyarakat untuk meningkatkan keuntungan jangka panjang. Bagi para investor, kemampuan untuk memprediksi pergerakan harga saham sangat penting dalam pengambilan keputusan. Saham BRI (BBRI) merupakan salah satu saham yang populer di Indonesia dan menarik untuk dianalisis. Pergerakan harga saham dipengaruhi oleh faktor ekonomi, kondisi pasar, serta faktor teknikal dan fundamental perusahaan. Oleh karena itu, menggunakan model machine learning untuk memprediksi harga saham dapat membantu investor membuat keputusan yang lebih baik.

**Mengapa dan Bagaimana Masalah Harus Diselesaikan:** Dengan menggunakan teknik analisis data dan model machine learning, kita dapat memprediksi tren harga saham di masa mendatang. Prediksi ini dapat membantu investor dalam mengoptimalkan strategi investasi dan mengurangi risiko yang mungkin terjadi.

**Referensi**:
- [PT. Bank Rakyat Indonesia (Persero) Tbk. (BBRI.JK) Yahoo Finance](https://finance.yahoo.com/quote/BBRI.JK/) 

- [Forecasting: What It Is, How It’s Used in Business and Investing](https://www.investopedia.com/terms/f/forecasting.asp)
  

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana memprediksi harga saham BBRI dalam jangka waktu tertentu melalui penggunaan data historis saham?
- Bagaimana model prediksi dapat secara akurat menggambarkan pola tren harga?
  
### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun model prediksi harga saham BBRI menggunakan metode machine learning.
- Membandingkan performa beberapa model dalam memprediksi harga saham.
  
### Solution statements
- Memanfaatkan model Prophet yang dikembangkan oleh Facebook untuk memproyeksikan harga saham berdasarkan data historis time series. 
- Menggunakan model ARIMA sebagai acuan dasar untuk membandingkan hasil dari Prophet dan memverifikasi ketepatan prediksi.

## Data Understanding
Sumber Data: Dataset diambil dari sumber historis harga saham BBRI yang mencakup informasi seperti harga pembukaan, harga penutupan, harga tertinggi, harga terendah, dan volume perdagangan.

|Date|Close|Close|High&|Low|Open|Volume|
|---|---|---|---|---|---|---|
|2022-04-11 00:00:00+00:00|382\.0|382\.0|416\.0|372\.0|400\.0|9410897000|
|2022-04-12 00:00:00+00:00|370\.0|370\.0|442\.0|360\.0|422\.0|3887331000|
|2022-04-13 00:00:00+00:00|374\.0|374\.0|380\.0|360\.0|370\.0|3262811400|
|2022-04-14 00:00:00+00:00|376\.0|376\.0|382\.0|374\.0|374\.0|3675981900|
|2022-04-18 00:00:00+00:00|378\.0|378\.0|380\.0|370\.0|376\.0|2660312700|
|2022-04-19 00:00:00+00:00|358\.0|358\.0|380\.0|358\.0|378\.0|2252971800|
|2022-04-20 00:00:00+00:00|338\.0|338\.0|364\.0|338\.0|358\.0|5804281200|
|2022-04-21 00:00:00+00:00|340\.0|340\.0|340\.0|336\.0|340\.0|1670584600|
|2022-04-22 00:00:00+00:00|340\.0|340\.0|340\.0|330\.0|340\.0|6075753400|
|2022-04-25 00:00:00+00:00|328\.0|328\.0|338\.0|318\.0|318\.0|1960587900|
|2022-04-26 00:00:00+00:00|310\.0|310\.0|348\.0|308\.0|348\.0|1843997100|
|2022-04-27 00:00:00+00:00|290\.0|290\.0|310\.0|290\.0|300\.0|2077577300|
|2022-04-28 00:00:00+00:00|272\.0|272\.0|290\.0|270\.0|270\.0|2280309800|
|2022-05-09 00:00:00+00:00|254\.0|254\.0|270\.0|254\.0|254\.0|621345600|
|2022-05-10 00:00:00+00:00|238\.0|238\.0|254\.0|238\.0|254\.0|201334400|
|2022-05-11 00:00:00+00:00|222\.0|222\.0|238\.0|222\.0|224\.0|2942404500|
|2022-05-12 00:00:00+00:00|208\.0|208\.0|220\.0|208\.0|212\.0|312982800|


### Variabel dalam dataset:
- **Date:** Tanggal perdagangan saham.
- **Open:** Harga pembukaan saham.
- **High:** Harga tertinggi saham pada hari tersebut.
- **Low:** Harga terendah saham pada hari tersebut.
- **Close:** Harga penutupan saham.
- **Volume:** Jumlah transaksi saham pada hari tersebut.

## Data Preparation

- **Handling Missing Data:** Dataset ini tidak memiliki nilai yang hilang, sehingga tidak memerlukan penanganan khusus.

**Alasan Data Preparation:**

- Tahapan seperti resampling dan scaling tidak diterapkan karena data hasil scraping sudah terstruktur dengan baik, hanya perlu penyesuaian pada nama kolom atau header.
  
## Modeling

### Model 1: Prophet
Prophet digunakan untuk menangkap pola musiman dan tren jangka panjang dalam data saham. Prophet merupakan model prediksi time series yang efisien dalam mengidentifikasi tren jangka panjang serta pola musiman. Dalam proyek ini, Prophet dipilih karena kemampuannya yang baik dalam mengelola data dengan nilai yang hilang serta kemampuannya untuk memodelkan komponen tren, musiman, dan perubahan.

Parameter Prophet:
- Daily seasonality: True
- Changepoint prior scale: Default (0.05)

### Model 2: ARIMA
ARIMA berfungsi sebagai model dasar untuk mengevaluasi kinerja prediksi. Model ini dipilih karena kemudahannya dalam memodelkan data time series non-stationary.

Parameter ARIMA:
- p: 5
- d: 1
- q: 0
  
**Kelebihan dan Kekurangan:**
Prophet sangat efektif dalam mengelola data time series yang memiliki pola musiman yang kuat, meskipun memerlukan jumlah data yang lebih besar untuk mencapai akurasi yang optimal. Di sisi lain, ARIMA menunjukkan kinerja yang baik pada data dengan pola musiman jangka pendek, tetapi kurang mampu menangani perubahan musiman yang kompleks, seperti yang terlihat pada saham BBRI.

## Evaluation

### Metrik Evaluasi:
- Mean Absolute Error (MAE): Metode ini digunakan untuk mengukur rata-rata kesalahan dalam prediksi dibandingkan dengan harga yang sebenarnya.
- Root Mean Square Error (RMSE): Metode ini berfungsi untuk memberikan penekanan pada kesalahan prediksi yang signifikan dan memberikan gambaran mengenai seberapa jauh prediksi dari nilai yang aktual.

## Visualisasi Prediksi

### Prophet
![Prophet](https://github.com/dream2hvn/SUB-Machine-Learning-Terapan/blob/main/Prophet%20Forecast.png)

### Arima
![Arima](https://github.com/dream2hvn/SUB-Machine-Learning-Terapan/blob/main/ARIMA%20Forecast.png)


### Hasil Evaluasi:

| Model  | MAE  | RMSE  |
|--------|------|-------|
| Prophet | 88.54 | 115.05  |
| ARIMA | 58.33 | 147.65  |

Setelah melakukan evaluasi performa kedua model menggunakan metrik MAE dan RMSE, didapatkan hasil yaitu Model ARIMA menghasilkan MAE sebesar 58.33 dan RMSE sebesar 147.65. Model Prophet menghasilkan MAE sebesar 88.54 dan RMSE sebesar 115.05. Dari hasil ini, ARIMA lebih unggul dalam hal akurasi prediksi berdasarkan nilai MAE yang lebih rendah, namun Prophet lebih unggul berdasarkan nilai RMSE yang lebih rendah pada data saham BBRI.


**Kesimpulan:** Meskipun Prophet dirancang untuk menangkap tren jangka panjang, pada dataset ini ARIMA memberikan hasil yang lebih akurat dalam memprediksi harga saham BBRI jika ditinjau dari MAE. Namun, Prophet menunjukkan performa yang lebih baik dalam menghindari error yang sangat besar jika ditinjau dari RMSE. Oleh karena itu, pemilihan model terbaik bergantung pada prioritas Anda, apakah ingin meminimalkan rata-rata kesalahan (MAE) atau meminimalkan error besar (RMSE). Jika ingin meminimalkan rata-rata kesalahan, ARIMA adalah pilihan yang lebih baik. Jika ingin meminimalisir error besar, Prophet adalah pilihan yang lebih baik.

