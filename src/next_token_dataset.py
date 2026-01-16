import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple

class NextTokenDataset(Dataset):
    """
    Dataset для задачи предсказания следующего токена
    Каждый пример: (входная последовательность, целевая последовательность)
    где целевая последовательность сдвинута на один токен
    """
    
    def __init__(self, 
                 data_path: str,
                 word2idx: dict,
                 max_length: int = 50,
                 sos_token: str = '<SOS>',
                 eos_token: str = '<EOS>',
                 pad_token: str = '<PAD>',
                 unk_token: str = '<UNK>'):
        """
        Args:
            data_path: путь к файлу с данными
            word2idx: словарь для преобразования слов в индексы
            max_length: максимальная длина последовательности
        """
        self.data = pd.read_csv(data_path)
        self.word2idx = word2idx
        self.max_length = max_length
        self.sos_idx = word2idx.get(sos_token, 2)
        self.eos_idx = word2idx.get(eos_token, 3)
        self.pad_idx = word2idx.get(pad_token, 0)
        self.unk_idx = word2idx.get(unk_token, 1)
        
        # Предобработка данных
        self.sequences = self._prepare_sequences()
    
    def _prepare_sequences(self) -> list:
        """Подготовка последовательностей"""
        sequences = []
        
        for _, row in self.data.iterrows():
            # Получаем токены из текста
            if 'tokens' in row:
                # Если токены уже предобработаны
                if isinstance(row['tokens'], str):
                    tokens = eval(row['tokens'])  # Осторожно! Лучше хранить как JSON
                else:
                    tokens = row['tokens']
            elif 'cleaned_text' in row:
                # Токенизируем на лету
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(row['cleaned_text'])
            else:
                # Используем исходный текст
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(str(row['text']).lower())
            
            # Ограничиваем длину
            tokens = tokens[:self.max_length-2]  # -2 для SOS и EOS
            
            # Преобразуем в индексы
            seq = [self.sos_idx]  # Начало последовательности
            for token in tokens:
                seq.append(self.word2idx.get(token, self.unk_idx))
            seq.append(self.eos_idx)  # Конец последовательности
            
            # Добавляем padding
            if len(seq) < self.max_length:
                seq += [self.pad_idx] * (self.max_length - len(seq))
            else:
                seq = seq[:self.max_length]
                seq[-1] = self.eos_idx  # Гарантируем EOS в конце
            
            sequences.append(seq)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает:
            input_seq: последовательность индексов
            target_seq: та же последовательность, сдвинутая на 1 токен
        """
        seq = self.sequences[idx]
        
        # Входная последовательность (все кроме последнего токена)
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        
        # Целевая последовательность (все кроме первого токена)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        
        return input_seq, target_seq
    
    def create_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Создание DataLoader"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )


def create_dataloaders(data_dir: str, 
                      word2idx: dict,
                      batch_size: int = 32,
                      max_length: int = 50) -> dict:
    """
    Создание DataLoader'ов для train, val, test
    
    Returns:
        Словарь с DataLoader'ами
    """
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        data_path = f"{data_dir}/{split}.csv"
        
        try:
            dataset = NextTokenDataset(
                data_path=data_path,
                word2idx=word2idx,
                max_length=max_length
            )
            datasets[split] = dataset
            
            shuffle = (split == 'train')
            dataloaders[split] = dataset.create_dataloader(
                batch_size=batch_size,
                shuffle=shuffle
            )
            
            print(f"{split}: {len(dataset)} примеров")
            
        except FileNotFoundError:
            print(f"Файл {data_path} не найден")
    
    return dataloaders