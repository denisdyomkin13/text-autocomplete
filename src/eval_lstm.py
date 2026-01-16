import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List
import evaluate
from tqdm import tqdm

from lstm_model import LSTMAutocomplete, TextGenerator

def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Вычисление метрик ROUGE
    
    Args:
        predictions: список предсказанных текстов
        references: список эталонных текстов
        
    Returns:
        Словарь с метриками ROUGE
    """
    rouge = evaluate.load('rouge')
    
    results = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_aggregator=True
    )
    
    return results

def evaluate_rouge(model: LSTMAutocomplete, 
                  dataloader: DataLoader,
                  device: torch.device,
                  idx2word: Dict[int, str] = None,
                  num_samples: int = 100) -> Dict[str, float]:
    """
    Оценка модели с помощью метрик ROUGE
    
    Args:
        model: модель LSTM
        dataloader: DataLoader с данными
        device: устройство для вычислений
        idx2word: словарь для преобразования индексов в слова
        num_samples: количество примеров для оценки
        
    Returns:
        Словарь с метриками ROUGE
    """
    model.eval()
    
    predictions = []
    references = []
    
    with torch.no_grad():
        # Берем только часть данных для ускорения
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= num_samples:
                break
            
            # Перемещаем данные на устройство
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Генерация продолжения (берем 3/4 текста как вход)
            batch_size, seq_len = input_seq.shape
            
            # Определяем точку разделения (3/4 текста)
            split_point = int(seq_len * 0.75)
            
            for i in range(batch_size):
                # Входная последовательность (первые 3/4)
                input_part = input_seq[i:i+1, :split_point]
                
                # Генерируем продолжение
                generated = model.generate(
                    input_part,
                    max_length=seq_len - split_point,
                    temperature=0.9,
                    top_k=50,
                    do_sample=True
                )[0]
                
                # Получаем сгенерированный текст (только продолжение)
                if idx2word:
                    gen_text = ids_to_text(generated[split_point:], idx2word)
                    ref_text = ids_to_text(target_seq[i].cpu().tolist(), idx2word)
                else:
                    # Без преобразования в слова
                    gen_text = ' '.join(map(str, generated[split_point:]))
                    ref_text = ' '.join(map(str, target_seq[i].cpu().tolist()))
                
                predictions.append(gen_text)
                references.append(ref_text)
    
    # Вычисляем метрики ROUGE
    if len(predictions) > 0:
        rouge_scores = calculate_rouge(predictions, references)
    else:
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    return rouge_scores

def ids_to_text(ids: List[int], idx2word: Dict[int, str]) -> str:
    """
    Преобразование последовательности индексов в текст
    
    Args:
        ids: список индексов
        idx2word: словарь для преобразования
        
    Returns:
        Текст
    """
    tokens = []
    for idx in ids:
        if idx == 3:  # EOS token
            break
        if idx not in [0, 2]:  # Игнорируем PAD и SOS
            tokens.append(idx2word.get(idx, '<UNK>'))
    return ' '.join(tokens)

def generate_examples(model: LSTMAutocomplete,
                     dataloader: DataLoader,
                     device: torch.device,
                     idx2word: Dict[int, str],
                     num_examples: int = 5) -> List[Dict[str, str]]:
    """
    Генерация примеров автодополнения
    
    Args:
        model: модель LSTM
        dataloader: DataLoader с данными
        device: устройство для вычислений
        idx2word: словарь для преобразования
        num_examples: количество примеров
        
    Returns:
        Список примеров с исходным текстом и предсказанием
    """
    model.eval()
    examples = []
    
    with torch.no_grad():
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            if len(examples) >= num_examples:
                break
            
            input_seq = input_seq.to(device)
            batch_size = input_seq.size(0)
            
            for i in range(min(batch_size, num_examples - len(examples))):
                # Берем первые 3/4 текста как промпт
                seq_len = input_seq.size(1)
                split_point = int(seq_len * 0.75)
                
                prompt_ids = input_seq[i:i+1, :split_point]
                
                # Генерируем продолжение
                generated_ids = model.generate(
                    prompt_ids,
                    max_length=seq_len - split_point,
                    temperature=0.9,
                    top_k=50,
                    do_sample=True
                )[0]
                
                # Преобразуем в текст
                prompt_text = ids_to_text(prompt_ids[0].cpu().tolist(), idx2word)
                generated_text = ids_to_text(generated_ids[split_point:], idx2word)
                reference_text = ids_to_text(target_seq[i].cpu().tolist(), idx2word)
                
                examples.append({
                    'prompt': prompt_text,
                    'generated': generated_text,
                    'reference': reference_text
                })
    
    return examples

def evaluate_model_on_test(model_path: str,
                          test_loader: DataLoader,
                          device: torch.device,
                          idx2word: Dict[int, str],
                          vocab_size: int,
                          model_config: Dict = None) -> Dict[str, float]:
    """
    Полная оценка модели на тестовом наборе
    
    Args:
        model_path: путь к сохраненной модели
        test_loader: DataLoader тестового набора
        device: устройство для вычислений
        idx2word: словарь для преобразования
        vocab_size: размер словаря
        model_config: конфигурация модели
        
    Returns:
        Словарь с метриками
    """
    # Загрузка модели
    if model_config is None:
        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.3
        }
    
    model = LSTMAutocomplete(**model_config)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Модель загружена из {model_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    # Оценка ROUGE
    rouge_scores = evaluate_rouge(model, test_loader, device, idx2word, num_samples=200)
    
    # Генерация примеров
    examples = generate_examples(model, test_loader, device, idx2word, num_examples=5)
    
    return {
        'rouge_scores': rouge_scores,
        'examples': examples
    }


if __name__ == "__main__":
    # Пример использования
    import pandas as pd
    from next_token_dataset import NextTokenDataset
    
    # Загрузка словаря
    vocab_df = pd.read_csv('data/vocab.csv')
    word2idx = dict(zip(vocab_df['word'], vocab_df['index']))
    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(word2idx)
    
    # Создание тестового DataLoader
    test_dataset = NextTokenDataset(
        data_path='data/test.csv',
        word2idx=word2idx,
        max_length=50
    )
    test_loader = test_dataset.create_dataloader(batch_size=32, shuffle=False)
    
    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Оценка модели
    results = evaluate_model_on_test(
        model_path='models/lstm_best_model.pth',
        test_loader=test_loader,
        device=device,
        idx2word=idx2word,
        vocab_size=vocab_size
    )
    
    print("\nРезультаты на тестовом наборе:")
    print(f"ROUGE-1: {results['rouge_scores']['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge_scores']['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rouge_scores']['rougeL']:.4f}")
    
    print("\nПримеры предсказаний:")
    for i, example in enumerate(results['examples'], 1):
        print(f"\nПример {i}:")
        print(f"  Промпт: {example['prompt']}")
        print(f"  Предсказание: {example['generated']}")
        print(f"  Эталон: {example['reference']}")