import re
from underthesea import pos_tag
import pandas as pd
import joblib
from gensim.models.phrases import Phrases, Phraser
from pyvi import ViTokenizer
import os
import argparse
import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


class Preprocess:
    @staticmethod
    def train_phrase_model(sentence_list):
        # Bước 1: Tách từ
        tokenized_sentences = [ViTokenizer.tokenize(sentence.lower()).split() for sentence in sentence_list]
        # Bước 2: Huấn luyện bigram
        bigram = Phrases(tokenized_sentences, min_count=3, threshold=5)
        bigram_phraser = Phraser(bigram)
        # Bước 3: Huấn luyện trigram (trên dữ liệu đã có bigram)
        trigram = Phrases(bigram[tokenized_sentences], min_count=3, threshold=5)
        trigram_phraser = Phraser(trigram)
        # Bước 4: Huấn luyện quadgram_phraser (trên dữ liệu đã có trigram)
        quadgram = Phrases(trigram[tokenized_sentences], min_count=3, threshold=5)
        quadgram_phraser = Phraser(quadgram)
        # Bước 5: Lưu mô hình
        joblib.dump(bigram_phraser, r'D:\ML2AI\ML2AI_api\apps\comment_emotion\training\bigram.model')
        joblib.dump(trigram_phraser, r'D:\ML2AI\ML2AI_api\apps\comment_emotion\training\trigram.model')
        joblib.dump(quadgram_phraser, r'D:\ML2AI\ML2AI_api\apps\comment_emotion\training\quadgram.model')
        return True

    @staticmethod
    def extract_phrases(text, bigram_phraser, trigram_phraser, quadgram_phraser):
        # Tiền xử lý câu đầu vào
        tokenized_text = ViTokenizer.tokenize(text).split()
        # Áp dụng mô hình tìm cụm từ
        bigram_text = bigram_phraser[tokenized_text]
        trigram_text = trigram_phraser[bigram_text]
        quadgram_text = quadgram_phraser[trigram_text]
        return quadgram_text

    @staticmethod
    def remove_punctuation(text):
        text = re.sub(r"[,.\"\'():;]", "", text)  # Bỏ dấu câu không quan trọng
        return text

    @staticmethod
    def run_preprocess(path):
        df = pd.read_csv(rf'{path}')
        print(df["sentiment"].value_counts())

        print('remove_punctuation()...')
        df["comment_clean"] = df["comment"].apply(Preprocess.remove_punctuation)
        print(f"✅ remove_punctuation() done!")

        print('train_phrase_model()...')
        Preprocess.train_phrase_model(df["comment_clean"])
        print(f"✅ train_phrase_model() done!")

        print('extract_phrases()...')
        bigram_phraser = joblib.load(r"D:\ML2AI\ML2AI_api\apps\comment_emotion\training\bigram.model")
        trigram_phraser = joblib.load(r"D:\ML2AI\ML2AI_api\apps\comment_emotion\training\trigram.model")
        quadgram_phraser = joblib.load(r"D:\ML2AI\ML2AI_api\apps\comment_emotion\training\quadgram.model")
        df["comment_clean"] = df["comment_clean"].apply(
            lambda x: Preprocess.extract_phrases(x, bigram_phraser, trigram_phraser, quadgram_phraser)
        )
        print(f"✅ extract_phrases() done!")

        label_mapping = {"Tích cực": 0, "Tiêu cực": 1, "Bình thường": 2,"Hỗn hợp": 3}
        df['sentiment'] = df['sentiment'].map(label_mapping)

        df = df.dropna()
        print(df.head(1))
        return df


class TrainModel:
    @staticmethod
    def vectorize_sentence(phrases, word2vec_model_loaded):
        word_vectors = [word2vec_model_loaded.wv[word] for word in phrases if word in word2vec_model_loaded.wv]
        if not word_vectors:  # Nếu câu không có từ nào trong Word2Vec
            return np.zeros(word2vec_model_loaded.vector_size)
        return np.mean(word_vectors, axis=0)  # Tính trung bình vector của các từ

    @staticmethod
    def train():
        print("🔄 Preprocessing data...")
        train_path = r'D:\ML2AI\ML2AI_api\apps\comment_emotion\dataset\train.csv'
        test_path = r'D:\ML2AI\ML2AI_api\apps\comment_emotion\dataset\test.csv'
        df_train = Preprocess.run_preprocess(train_path)
        df_test = Preprocess.run_preprocess(test_path)

        X_train, y_train = df_train['comment_clean'].tolist(), df_train['sentiment'].tolist()
        X_test, y_test = df_test['comment_clean'].tolist(), df_test['sentiment'].tolist()

        print("🔄 Training Word2Vec...")
        # Train Word2Vec trên dữ liệu huấn luyện
        word2vec_model_loaded = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=2, workers=os.cpu_count())

        # Áp dụng vector hóa cho dữ liệu train/test
        X_train_vec = np.array([TrainModel.vectorize_sentence(phrases, word2vec_model_loaded) for phrases in X_train])
        X_test_vec = np.array([TrainModel.vectorize_sentence(phrases, word2vec_model_loaded) for phrases in X_test])

        print("🔄 Training SVM model...")
        # Huấn luyện mô hình SVM
        svm_model = SVC(kernel='rbf', C=1.0, verbose=1)
        svm_model.fit(X_train_vec, y_train)

        print("🔄 Testing SVM model...")
        # Dự đoán
        y_pred_svm = svm_model.predict(X_test_vec)

        # Đánh giá
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        print(f"Accuracy: {accuracy_svm:.2f}")
        print("Report:\n", classification_report(y_test, y_pred_svm))

        # Lưu SVM
        joblib.dump(svm_model, r"D:\ML2AI\ML2AI_api\apps\comment_emotion\training\svm_model.model")
        # Lưu word2vec_model_loaded
        word2vec_model_loaded.save(r"D:\ML2AI\ML2AI_api\apps\comment_emotion\training\word2vec_sentiment.model")

        print("✅ Training complete! Saved model!")
        return True


class Predict:
    @staticmethod
    def load_model():
        svm_model_loaded = joblib.load(r"D:\ML2AI\ML2AI_api\apps\comment_emotion\training\svm_model.model")
        vectorizer_loaded = joblib.load(r"D:\ML2AI\ML2AI_api\apps\comment_emotion\training\word2vec_sentiment.model")
        return svm_model_loaded, vectorizer_loaded

    @staticmethod
    def predict(sentence, output_type="convert"):
        """
        sentence: 'San pham nay rat tot'
        output_type: 'raw' | 'convert'
        """
        svm_model_loaded, word2vec_model_loaded = Predict.load_model()
        bigram_phraser = joblib.load(r"D:\ML2AI\ML2AI_api\apps\comment_emotion\training\bigram.model")
        trigram_phraser = joblib.load(r"D:\ML2AI\ML2AI_api\apps\comment_emotion\training\trigram.model")
        quadgram_phraser = joblib.load(r"D:\ML2AI\ML2AI_api\apps\comment_emotion\training\quadgram.model")
        phrases = Preprocess.extract_phrases(sentence, bigram_phraser, trigram_phraser, quadgram_phraser)
        sentence_vec = TrainModel.vectorize_sentence(phrases, word2vec_model_loaded).reshape(1, -1)
        prediction = svm_model_loaded.predict(sentence_vec)

        # Đảo ngược label_mapping để chuyển số -> chuỗi
        label_mapping = {"Tích cực": 0, "Tiêu cực": 1, "Bình thường": 2, "Hỗn hợp": 3}
        inverse_label_mapping = {v: k for k, v in label_mapping.items()}

        # Chuyển kết quả số về nhãn văn bản
        return inverse_label_mapping[prediction[0]] if output_type == "convert" else prediction[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False, help="is train")
    parser.add_argument("--predict", type=bool, default=False, help="is predict")
    parser.add_argument("--sentence", type=str, default='', help="sentence predict")
    args = parser.parse_args()

    if args.train:
        TrainModel.train()
    elif args.predict and args.sentence:
        print(Predict.predict(args.sentence))
    else:
        print('Error args format!')
