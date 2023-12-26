import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# CSV dosyasını oku
df = pd.read_csv('C:\\Users\\mrdnc\\OneDrive\\Desktop\\difFolder\\kultur-sanat_news_dataset.csv')



# 'Content' sütununu ayırıp listele
transactions = df['Content'].apply(lambda x: x.split()).tolist()

#İstenmeyen Bağlaçlar
trash_words = [
    "ama", "ancak", "artik", "asla", "az", "bazen", "bazi", "bazisi", "belki", "bile",
    "bir", "biraz", "bircoğu", "birçok", "biri", "birkac", "birkez", "birsey", "birseyi",
    "biz", "bize", "bizden", "bizim", "boyle", "boylece", "bu", "buna", "bunda", "bundan",
    "bunu", "bunun", "burada", "butun", "çoğu", "çoğuna", "çoğunu", "cok", "cunku", "da",
    "daha", "dahi", "de", "defa", "degil", "diğer", "diye", "eger", "en", "gibi", "hem",
    "hep", "hepsi", "her", "hic", "icin", "ile", "ise", "kez", "ki", "kim", "kimden", "kime",
    "kimi", "kimse", "madem", "mi", "mu", "nasil", "ne", "neden", "nerde", "nerede", "nereye",
    "niçin", "niye", "o", "on", "ona", "ondan", "onlar", "onlara", "onlardan", "onlarin", "onu",
    "onun", "otuz", "oysa", "sanki", "sekiz", "seksen", "sen", "senden", "seni", "senin", "siz",
    "sizden", "size", "sizi", "sizin", "tum", "var", "ve", "veya", "ya", "yani", "yetmis",
    "yirmi", "yuz", "zaten", "cunku"
]
#Bağlaçları veri setinden çıkardığımız yer
transactions = [[word for word in transaction if word not in trash_words] for transaction in transactions]
# TransactionEncoder kullanarak veriyi uygun formata dönüştürür
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transformed = pd.DataFrame(te_ary, columns=te.columns_)

# FP-Growth algoritmasıyla sık kalıpları çıkardığımız yer
frequent_itemsets = fpgrowth(df_transformed, min_support=0.2, use_colnames=True)
sorted_frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=True)


#Çıktı
print(frequent_itemsets)
#Çıktıyı yeni oluşturduğu csv dosyasına yazdırdığı kod
sorted_frequent_itemsets.to_csv('C:\\Users\\mrdnc\\OneDrive\\Desktop\\difFolder\\output.csv', index=False)