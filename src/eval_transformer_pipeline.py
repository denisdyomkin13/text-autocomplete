import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import evaluate
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TransformerEvaluator:
    """Класс для оценки предобученной модели Transformer"""
    
    def __init__(self, model_name: str = "distilgpt2", device: int = -1):
        """
        Args:
            model_name: название модели из HuggingFace
            device: -1 для CPU, 0 для GPU
        """
        self.device = device
        self.model_name = model_name
        
        # Загрузка токенизатора и модели
        print(f"Загрузка модели {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Добавляем pad_token если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Создание pipeline для генерации текста
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device=device,
            framework="pt"
        )
        
        # Загрузка метрики ROUGE
        self.rouge = evaluate.load('rouge')
        
        print(f"Модель {model_name} загружена успешно")
    
    def preprocess_text(self, text: str) -> str:
        """Предобработка текста"""
        # Удаляем лишние пробелы
        text = ' '.join(text.split())
        return text
    
    def split_text_for_evaluation(self, text: str, split_ratio: float = 0.75) -> tuple:
        """
        Разделение текста на промпт и продолжение
        
        Args:
            text: исходный текст
            split_ratio: доля текста для промпта
            
        Returns:
            (prompt, continuation)
        """
        # Токенизация текста
        tokens = text.split()
        
        # Определяем точку разделения
        split_point = int(len(tokens) * split_ratio)
        
        # Разделяем
        prompt_tokens = tokens[:split_point]
        continuation_tokens = tokens[split_point:]
        
        prompt = ' '.join(prompt_tokens)
        continuation = ' '.join(continuation_tokens)
        
        return prompt, continuation
    
    def generate_continuation(self, 
                             prompt: str,
                             max_new_tokens: int = 20,
                             temperature: float = 0.9,
                             top_k: int = 50,
                             top_p: float = 0.95,
                             do_sample: bool = True) -> str:
        """
        Генерация продолжения текста
        
        Args:
            prompt: начальный текст
            max_new_tokens: максимальное количество новых токенов
            temperature: температура для сэмплирования
            top_k: top-k фильтрация
            top_p: nucleus sampling
            do_sample: использовать ли сэмплирование
            
        Returns:
            Сгенерированное продолжение
        """
        # Генерация
        result = self.generator(
            prompt,
            max_length=len(prompt.split()) + max_new_tokens,
            num_return_sequences=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Извлекаем сгенерированный текст
        generated_text = result[0]['generated_text']
        
        # Удаляем промпт из результата
        continuation = generated_text[len(prompt):].strip()
        
        return continuation
    
    def evaluate_on_dataset(self, 
                           data_path: str,
                           num_samples: int = 100,
                           text_column: str = 'text',
                           split_ratio: float = 0.75) -> Dict[str, float]:
        """
        Оценка модели на датасете
        
        Args:
            data_path: путь к файлу с данными
            num_samples: количество примеров для оценки
            text_column: название колонки с текстом
            split_ratio: доля текста для промпта
            
        Returns:
            Словарь с метриками ROUGE
        """
        # Загрузка данных
        df = pd.read_csv(data_path)
        
        # Ограничиваем количество примеров
        if num_samples < len(df):
            df = df.sample(num_samples, random_state=42)
        
        predictions = []
        references = []
        
        print(f"Оценка на {len(df)} примерах...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Оценка"):
            text = str(row[text_column])
            text = self.preprocess_text(text)
            
            # Пропускаем слишком короткие тексты
            if len(text.split()) < 10:
                continue
            
            # Разделяем текст на промпт и продолжение
            prompt, reference = self.split_text_for_evaluation(text, split_ratio)
            
            # Генерируем продолжение
            try:
                prediction = self.generate_continuation(
                    prompt,
                    max_new_tokens=20,
                    temperature=0.9,
                    top_k=50,
                    do_sample=True
                )
                
                predictions.append(prediction)
                references.append(reference)
                
            except Exception as e:
                print(f"Ошибка при генерации для примера {idx}: {e}")
                continue
        
        # Вычисляем метрики ROUGE
        if len(predictions) > 0:
            results = self.rouge.compute(
                predictions=predictions,
                references=references,
                rouge_types=['rouge1', 'rouge2', 'rougeL'],
                use_aggregator=True
            )
        else:
            results = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        return results, predictions, references
    
    def generate_examples(self, 
                         data_path: str,
                         num_examples: int = 5,
                         text_column: str = 'text',
                         split_ratio: float = 0.75) -> List[Dict[str, str]]:
        """
        Генерация примеров для визуальной оценки
        
        Args:
            data_path: путь к файлу с данными
            num_examples: количество примеров
            text_column: название колонки с текстом
            split_ratio: доля текста для промпта
            
        Returns:
            Список примеров
        """
        df = pd.read_csv(data_path)
        df = df.sample(num_examples, random_state=42)
        
        examples = []
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            text = self.preprocess_text(text)
            
            # Пропускаем слишком короткие тексты
            if len(text.split()) < 10:
                continue
            
            # Разделяем текст
            prompt, reference = self.split_text_for_evaluation(text, split_ratio)
            
            # Генерируем продолжение
            try:
                prediction = self.generate_continuation(
                    prompt,
                    max_new_tokens=20,
                    temperature=0.9,
                    top_k=50,
                    do_sample=True
                )
                
                examples.append({
                    'prompt': prompt,
                    'generated': prediction,
                    'reference': reference,
                    'model': self.model_name
                })
                
            except Exception as e:
                print(f"Ошибка при генерации для примера {idx}: {e}")
                continue
            
            if len(examples) >= num_examples:
                break
        
        return examples


def compare_models(lstm_results: Dict, transformer_results: Dict) -> Dict:
    """
    Сравнение результатов двух моделей
    
    Args:
        lstm_results: результаты LSTM модели
        transformer_results: результаты Transformer модели
        
    Returns:
        Сравнительная таблица
    """
    comparison = {
        'Model': ['LSTM', 'Transformer (distilgpt2)'],
        'ROUGE-1': [
            lstm_results.get('rouge1', 0),
            transformer_results.get('rouge1', 0)
        ],
        'ROUGE-2': [
            lstm_results.get('rouge2', 0),
            transformer_results.get('rouge2', 0)
        ],
        'ROUGE-L': [
            lstm_results.get('rougeL', 0),
            transformer_results.get('rougeL', 0)
        ]
    }
    
    return comparison


if __name__ == "__main__":
    # Инициализация оценщика
    device = 0 if torch.cuda.is_available() else -1
    evaluator = TransformerEvaluator(model_name="distilgpt2", device=device)
    
    # Оценка на валидационном наборе
    print("\nОценка на валидационном наборе:")
    val_results, predictions, references = evaluator.evaluate_on_dataset(
        data_path='data/val.csv',
        num_samples=100,
        text_column='cleaned_text',
        split_ratio=0.75
    )
    
    print(f"\nРезультаты Transformer модели (distilgpt2):")
    print(f"ROUGE-1: {val_results['rouge1']:.4f}")
    print(f"ROUGE-2: {val_results['rouge2']:.4f}")
    print(f"ROUGE-L: {val_results['rougeL']:.4f}")
    
    # Генерация примеров
    print("\nГенерация примеров...")
    examples = evaluator.generate_examples(
        data_path='data/val.csv',
        num_examples=5,
        text_column='cleaned_text',
        split_ratio=0.75
    )
    
    print("\nПримеры предсказаний Transformer модели:")
    for i, example in enumerate(examples, 1):
        print(f"\nПример {i}:")
        print(f"  Промпт: {example['prompt']}")
        print(f"  Предсказание: {example['generated']}")
        print(f"  Эталон: {example['reference']}")
    
    # Сохранение результатов
    results_df = pd.DataFrame([val_results])
    results_df.to_csv('transformer_results.csv', index=False)
    
    print("\nРезультаты сохранены в transformer_results.csv")