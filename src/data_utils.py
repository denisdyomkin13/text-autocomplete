import pandas as pd
import numpy as np
import re
from typing import List, Tuple
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os
import csv

# Скачиваем необходимые ресурсы NLTK для английского
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

class TextPreprocessor:
    """Класс для предобработки текстовых данных"""
    
    def __init__(self, language: str = "english"):
        self.language = language
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        
    def clean_tweet_text(self, text: str) -> str:
        """
        Специальная очистка для твитов
        """
        if not isinstance(text, str):
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Удаление упоминаний (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Удаление хэштегов (но сохраняем текст хэштега)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Удаление HTML-сущностей
        text = re.sub(r'&amp;|&lt;|&gt;|&quot;', ' ', text)
        
        # Удаление нежелательных символов, оставляем буквы, цифры и основные знаки препинания
        text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", ' ', text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Токенизация текста с помощью NLTK
        """
        return word_tokenize(text, language=self.language)
    
    def build_vocabulary(self, tokens_list: List[List[str]], 
                        max_vocab_size: int = 10000,
                        min_freq: int = 2) -> Tuple[dict, dict]:
        """
        Построение словаря на основе токенизированных текстов
        """
        from collections import Counter
        
        # Считаем частоту слов
        word_counts = Counter()
        for tokens in tokens_list:
            word_counts.update(tokens)
        
        # Отбираем наиболее частые слова
        common_words = [word for word, count in word_counts.most_common(max_vocab_size) 
                       if count >= min_freq]
        
        # Добавляем специальные токены
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        vocab = special_tokens + common_words
        
        # Создаем словари
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.vocab = vocab
        
        return self.word2idx, self.idx2word
    
    def load_tweets_from_csv(self, data_path: str) -> pd.DataFrame:
        """
        Загрузка твитов из CSV файла с обработкой ошибок форматирования
        """
        tweets = []
        
        try:
            # Пробуем разные способы чтения CSV
            try:
                # Способ 1: Чтение как обычный текст (построчно)
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Пропускаем пустые строки
                            tweets.append(line)
                
                # Если строк слишком мало, пробуем другой метод
                if len(tweets) < 10:
                    raise ValueError("Слишком мало строк, пробуем другой метод")
                    
            except:
                # Способ 2: Используем csv.reader с разными параметрами
                with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f, quotechar='"', delimiter=',', 
                                      quoting=csv.QUOTE_ALL, skipinitialspace=True)
                    
                    for row in reader:
                        if row:  # Пропускаем пустые строки
                            # Объединяем все ячейки в одну строку
                            tweet_text = ' '.join(str(cell) for cell in row)
                            tweets.append(tweet_text)
                
        except Exception as e:
            print(f"Ошибка при чтении CSV: {e}")
            
            # Способ 3: Простое построчное чтение
            with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
                tweets = [line.strip() for line in f if line.strip()]
        
        # Создаем DataFrame
        df = pd.DataFrame({'text': tweets})
        
        print(f"Загружено {len(df)} твитов")
        
        # Показываем примеры
        if len(df) > 0:
            print("\nПримеры загруженных твитов:")
            for i in range(min(3, len(df))):
                print(f"  {i+1}. {df.iloc[i]['text'][:80]}...")
        
        return df
    
    def prepare_twitter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка твиттер-датасета
        """
        # Очистка текстов
        print("Очистка текстов твитов...")
        df['cleaned_text'] = df['text'].apply(self.clean_tweet_text)
        
        # Удаляем пустые тексты
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Токенизация
        print("Токенизация текстов...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_text)
        
        # Фильтрация слишком коротких текстов (минимум 3 токена)
        df = df[df['tokens'].apply(len) >= 3]
        
        # Ограничение длины последовательности
        max_len = 40
        df['tokens'] = df['tokens'].apply(lambda x: x[:max_len])
        
        return df
    
    def load_and_preprocess_data(self, 
                                data_path: str,
                                sample_size: int = None,
                                test_size: float = 0.2,
                                val_size: float = 0.1) -> dict:
        """
        Загрузка и предобработка данных
        """
        print(f"\nЗагрузка данных из {data_path}...")
        
        # Загрузка данных
        df = self.load_tweets_from_csv(data_path)
        
        if df.empty:
            raise ValueError(f"Не удалось загрузить данные из {data_path}")
        
        # Если указан sample_size, берем часть данных
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            print(f"Используем {sample_size} случайных примеров")
        
        # Подготовка твиттер-датасета
        df = self.prepare_twitter_dataset(df)
        
        # Построение словаря
        print("Построение словаря...")
        self.build_vocabulary(df['tokens'].tolist(), max_vocab_size=8000, min_freq=2)
        
        # Разделение на train/test
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # Дополнительное разделение train на train/val
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42)
        
        print(f"\nРазмеры выборок:")
        print(f"  Обучающая: {len(train_df)}")
        print(f"  Валидационная: {len(val_df)}")
        print(f"  Тестовая: {len(test_df)}")
        print(f"  Размер словаря: {len(self.vocab)}")
        
        # Сохранение обработанных данных
        data_dir = os.path.dirname(data_path)
        
        df.to_csv(os.path.join(data_dir, 'dataset_processed.csv'), index=False)
        train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
        
        # Сохранение словаря
        vocab_df = pd.DataFrame({
            'word': list(self.word2idx.keys()),
            'index': list(self.word2idx.values())
        })
        vocab_df.to_csv(os.path.join(data_dir, 'vocab.csv'), index=False)
        
        # Примеры очищенных текстов
        print("\nПримеры очищенных текстов:")
        for i in range(min(3, len(train_df))):
            original = train_df.iloc[i]['text'][:60]
            cleaned = train_df.iloc[i]['cleaned_text'][:60]
            print(f"  {i+1}. Исходный: {original}...")
            print(f"     Очищенный: {cleaned}...")
            print(f"     Токены: {train_df.iloc[i]['tokens'][:10]}...")
            print()
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': len(self.vocab)
        }


def quick_preprocess_sample():
    """
    Быстрая предобработка небольшой выборки для тестирования
    """
    preprocessor = TextPreprocessor(language="english")
    
    # Путь к вашему датасету
    data_path = "data/raw_dataset.csv"
    
    if not os.path.exists(data_path):
        print(f"Файл {data_path} не найден.")
        print("Создаем образец данных...")
        
        # Создаем тестовые данные
        sample_tweets = [
            "@tommcfly Awesome movie!!",
            "Yay. They're coming back tomorrow!!",
            "Just had a fun photo shoot with my girls",
            "I love this movie so much",
            "Going to the beach today",
            "Can't wait for the weekend",
            "Happy birthday to me",
            "Just finished my homework",
            "I need to buy groceries",
            "Feeling so happy right now"
        ]
        
        os.makedirs('data', exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            for tweet in sample_tweets:
                f.write(tweet + '\n')
        
        print(f"Создан файл с {len(sample_tweets)} примерами")
    
    try:
        print("\nЗапуск быстрой предобработки...")
        data = preprocessor.load_and_preprocess_data(
            data_path=data_path,
            sample_size=1000,  # Маленькая выборка для теста
            test_size=0.2,
            val_size=0.1
        )
        print(f"\n✅ Предобработка завершена успешно!")
        print(f"   Размер словаря: {data['vocab_size']}")
        return data
    except Exception as e:
        print(f"\n❌ Ошибка при предобработке: {e}")
        return None


if __name__ == "__main__":
    # Быстрый тест
    quick_preprocess_sample()