# Demand Forecasting

* Bu projede ilk amacım, 2013-2018 yılları arasındaki satış verisini kullanarak LightGBM algoritması ile 2018 yılının ilk üç ayı için talep tahmini yapmaktır.
* İkinci amacım ise günlük olan bu veri setini haftalığa indirgeyerek 2017 yılı için LightGBM algoritması ve zaman serilerini (Smoothing yöntemleri, ARIMA, SARIMA)
kullanarak bir yıllık talep tahmini yapmaktır.

![alt text](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/10/82241time_series.jpg)

## Veri Setine Genel Bakış
* Bir mağaza zincirinin 5 yıllık verilerinde, 10 farklı mağaza ve 50 farklı ürünün bilgileri yer almaktadır.
* Veri seti 01-01-2013 ile 31-12-2017 arasındaki dönemi kapsamaktadır.

## İş Problemi
* Bir mağaza zincirinin 10 farklı mağazası ve 50 farklı ürünü için 3 aylık bir talep tahmini modeli oluşturulmak istenmektedir.
* Daha sonra ise veri seti haftalığa indirgenip 2017 yılı için bir talep tahmin modeli oluşturulmak istenmektedir.

## Değişkenler
* date – Satış verilerinin tarihi Tatil efekti veya mağaza kapanışı yoktur.
* store – Mağaza ID’si Her bir mağaza için eşsiz numara.
* item – Ürün ID’si Her bir ürün için eşsiz numara.
* sales – Satılan ürün sayıları, Belirli bir tarihte belirli bir mağazadan satılan ürünlerin sayısı

Kaggle: https://www.kaggle.com/haticeebraralc/demand-forecasting
