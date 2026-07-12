import re
from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded if possible, but handle offline cases gracefully
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

class ReviewPreprocessor:
    def __init__(self):
        # Initialize Sastrawi stemmer
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Load Indonesian stopwords
        try:
            self.stop_words = set(stopwords.words('indonesian'))
        except Exception:
            # Fallback if NLTK stopwords fail to load
            self.stop_words = {
                'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua', 
                'ia', 'seperti', 'jika', 'sehingga', 'kembali', 'dan', 'tidak', 'ini', 'karena', 
                'kepada', 'oleh', 'saat', 'harus', 'sementara', 'setelah', 'belum', 'kami', 'mereka', 
                'sudah', 'adalah', 'dari', 'kalau', 'atau', 'dengan', 'ada', 'telah', 'bagi', 'dalam',
                'akan', 'itu', 'adalah', 'tentang', 'bahwa', 'oleh', 'serta', 'dapat', 'tersebut'
            }
            
        # Add custom marketplace-specific stopwords that don't carry sentiment
        custom_stopwords = {
            'tokopedia', 'shopee', 'lazada', 'bukalapak', 'seller', 'admin', 'paket', 'kurir', 
            'pengiriman', 'kirim', 'ekspedisi', 'barang', 'produk', 'beli', 'pesan', 'order', 
            'belanja', 'toko', 'aplikasi', 'situs', 'web', 'tgl', 'tanggal', 'rupiah', 'rp'
        }
        self.stop_words.update(custom_stopwords)
        
        # Comprehensive Indonesian Slang/Abbreviation Dictionary
        self.slang_dict = {
            'yg': 'yang',
            'ga': 'tidak',
            'gak': 'tidak',
            'g': 'tidak',
            'nda': 'tidak',
            'tdk': 'tidak',
            'bgt': 'banget',
            'dgn': 'dengan',
            'utk': 'untuk',
            'kalo': 'kalau',
            'tp': 'tapi',
            'sdh': 'sudah',
            'uda': 'sudah',
            'udah': 'sudah',
            'bs': 'bisa',
            'brg': 'barang',
            'dpt': 'dapat',
            'sy': 'saya',
            'aq': 'saya',
            'km': 'kamu',
            'lu': 'kamu',
            'loe': 'kamu',
            'mksih': 'terima kasih',
            'makasih': 'terima kasih',
            'thx': 'terima kasih',
            'thanks': 'terima kasih',
            'recomended': 'rekomendasi',
            'recsel': 'rekomendasi',
            'aj': 'saja',
            'aja': 'saja',
            'pake': 'pakai',
            'pakek': 'pakai',
            'krn': 'karena',
            'bener': 'benar',
            'mantul': 'bagus',
            'mantap': 'bagus',
            'cepet': 'cepat',
            'cpt': 'cepat',
            'peking': 'kemasan',
            'packing': 'kemasan',
            'ongkir': 'ongkos kirim',
            'ori': 'asli',
            'original': 'asli',
            'jos': 'bagus',
            'sip': 'bagus',
            'oke': 'baik',
            'ok': 'baik',
            'suka': 'suka',
            'joss': 'bagus',
            'dikit': 'sedikit',
            'mura': 'murah',
            'murah': 'murah',
            'mahal': 'mahal',
            'bagus': 'bagus',
            'jelek': 'jelek'
        }
        
        # Setup cached stemmer to boost performance during preprocessing
        self._setup_cached_stemmer()

    def _setup_cached_stemmer(self):
        stem_func = self.stemmer.stem
        
        @lru_cache(maxsize=30000)
        def cached_stem(word):
            return stem_func(word)
            
        self.cached_stem = cached_stem

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # 1. Case Folding
        text = text.lower()
        
        # 2. Clean URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # 3. Clean HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # 4. Clean email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # 5. Clean numbers, punctuation, non-alphabetic chars, and emojis
        # Keep only alphabetic characters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # 5.b Collapse repeated characters (3+ repetitions -> 2 repetitions)
        # Prevents Sastrawi stemmer from hanging due to regex catastrophic backtracking on long slang repetitions
        # (e.g. "baguuuuuuuuusssssss" -> "baguuss", which stems correctly and fast)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 6. Remove excess whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess(self, text, stem=False):
        """
        Preprocess text: clean, tokenize, normalize slang, remove stopwords, and optionally stem.
        Default is stem=False because training on the full 40k dataset without stemming is 100x faster,
        prevents CPU bottlenecks, and preserves word forms which improves sentiment classification.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Clean text first
        cleaned = self.clean_text(text)
        
        # Tokenize (split by space)
        tokens = cleaned.split()
        
        processed_tokens = []
        for token in tokens:
            # 1. Slang Normalization (token-based lookup to avoid substring replacement bugs)
            token_normalized = self.slang_dict.get(token, token)
            
            # 2. Stopwords Removal (Skip if in stop_words)
            if token_normalized in self.stop_words:
                continue
                
            # 3. Optional Stemming (cached for speed)
            if stem:
                processed_token = self.cached_stem(token_normalized)
            else:
                processed_token = token_normalized
                
            if processed_token and len(processed_token) > 1:
                processed_tokens.append(processed_token)
                
        # Reconstruct string
        return " ".join(processed_tokens)
