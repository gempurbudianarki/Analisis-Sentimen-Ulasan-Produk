import os
import joblib
from preprocessing import ReviewPreprocessor

class SentimentPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'model_sentimen.joblib')
        vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.joblib')
        
        # Load artifacts with auto-retrain fallback on mismatch
        try:
            # Check if model files exist
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                raise FileNotFoundError("Model files missing.")
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            print(f"[PREDICTOR] Gagal memuat model ({e}). Menjalankan retrain otomatis...")
            try:
                # Import and run training script dynamically
                import sys
                sys.path.insert(0, os.path.join(base_dir, 'src'))
                from train_model import main as train_main
                train_main()
                
                # Reload after training
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                print("[PREDICTOR] Retrain otomatis berhasil dan model dimuat.")
            except Exception as retrain_error:
                raise RuntimeError(
                    f"Gagal melatih ulang model secara otomatis: {retrain_error}"
                ) from e
        
        # Initialize preprocessor
        self.preprocessor = ReviewPreprocessor()
        
    def predict(self, text):
        """
        Predicts the sentiment of a given Indonesian text.
        Returns:
            dict: A dictionary containing:
                - 'sentiment': 'Positif' | 'Netral' | 'Negatif'
                - 'clean_text': The preprocessed text
                - 'confidence': The confidence score (probability) of the predicted class
                - 'probabilities': A dictionary of probabilities for all classes
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'Netral',
                'clean_text': '',
                'confidence': 1.0,
                'probabilities': {'Positif': 0.33, 'Netral': 0.34, 'Negatif': 0.33}
            }
            
        # 1. Preprocess (stem=False to match the training preprocessing pipeline)
        clean_text = self.preprocessor.preprocess(text, stem=False)
        
        # If text is empty after preprocessing
        if not clean_text:
            return {
                'sentiment': 'Netral',
                'clean_text': '',
                'confidence': 0.5,
                'probabilities': {'Positif': 0.33, 'Netral': 0.34, 'Negatif': 0.33}
            }
            
        # 2. Vectorize
        vectorized = self.vectorizer.transform([clean_text])
        
        # 3. Predict class
        prediction = self.model.predict(vectorized)[0]
        
        # 4. Predict probabilities
        probabilities = self.model.predict_proba(vectorized)[0]
        class_labels = self.model.classes_
        
        prob_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
        confidence = prob_dict[prediction]
        
        return {
            'sentiment': prediction,
            'clean_text': clean_text,
            'confidence': confidence,
            'probabilities': prob_dict
        }
