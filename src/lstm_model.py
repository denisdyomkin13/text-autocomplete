import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import random

class LSTMAutocomplete(nn.Module):
    """
    Модель LSTM для автодополнения текста
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        
        # Embedding слой
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_idx
        )
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Dropout слой
        self.dropout = nn.Dropout(dropout)
        
        # Полносвязный слой для предсказания следующего токена
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        # Инициализация embedding слоя
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        if self.pad_idx is not None:
            self.embedding.weight.data[self.pad_idx].zero_()
        
        # Инициализация LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Инициализация полносвязного слоя
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, 
                input_ids: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Прямой проход для обучения
        
        Args:
            input_ids: входные токены [batch_size, seq_len]
            hidden: скрытое состояние LSTM
            
        Returns:
            logits: предсказания для следующего токена [batch_size, seq_len, vocab_size]
            hidden: обновленное скрытое состояние
        """
        batch_size = input_ids.size(0)
        
        # Embedding
        embedded = self.dropout(self.embedding(input_ids))
        
        # LSTM
        lstm_output, hidden = self.lstm(embedded, hidden)
        
        # Применяем dropout
        lstm_output = self.dropout(lstm_output)
        
        # Предсказание следующего токена
        logits = self.fc(lstm_output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """Инициализация скрытого состояния"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def generate(self,
                 input_ids: torch.Tensor,
                 max_length: int = 20,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 do_sample: bool = True,
                 eos_token_id: int = 3) -> List[List[int]]:
        """
        Генерация продолжения текста
        
        Args:
            input_ids: начальная последовательность [batch_size, seq_len]
            max_length: максимальная длина генерируемого текста
            temperature: температура для сэмплирования
            top_k: количество топ-k токенов для рассмотрения
            top_p: вероятность для nucleus sampling
            do_sample: использовать ли сэмплирование
            eos_token_id: ID токена конца последовательности
            
        Returns:
            Сгенерированные последовательности
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Инициализация скрытого состояния
        hidden = self.init_hidden(batch_size, device)
        
        # Копируем входную последовательность
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Получаем предсказания для последнего токена
                logits, hidden = self.forward(generated, hidden)
                
                # Берем последний токен
                next_token_logits = logits[:, -1, :] / temperature
                
                # Применяем top-k фильтрацию
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Применяем nucleus (top-p) фильтрацию
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Удаляем токены с cumulative probability выше threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Сдвигаем индексы на 1, чтобы оставить первый токен выше threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Получаем следующий токен
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Добавляем следующий токен к сгенерированной последовательности
                generated = torch.cat([generated, next_token], dim=1)
                
                # Проверяем, не достигли ли конца последовательности
                if (next_token == eos_token_id).all():
                    break
        
        return generated.cpu().tolist()


class TextGenerator:
    """Вспомогательный класс для генерации текста"""
    
    def __init__(self, model: LSTMAutocomplete, idx2word: dict):
        self.model = model
        self.idx2word = idx2word
        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
    
    def ids_to_text(self, ids: List[int]) -> str:
        """Преобразование индексов в текст"""
        tokens = []
        for idx in ids:
            if idx == self.eos_idx:
                break
            if idx not in [self.pad_idx, self.sos_idx]:
                tokens.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(tokens)
    
    def generate_text(self, 
                     prompt: str,
                     max_length: int = 20,
                     temperature: float = 1.0,
                     top_k: int = 50,
                     top_p: float = 0.95,
                     do_sample: bool = True) -> str:
        """
        Генерация текста по промпту
        
        Args:
            prompt: начальный текст
            max_length: максимальная длина генерируемого текста
            temperature: температура для сэмплирования
            top_k: top-k фильтрация
            top_p: nucleus sampling
            do_sample: использовать ли сэмплирование
            
        Returns:
            Сгенерированный текст
        """
        # Токенизация промпта
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(prompt.lower())
        
        # Преобразование в индексы
        ids = [self.sos_idx]
        for token in tokens:
            # Ищем токен в словаре
            for idx, word in self.idx2word.items():
                if word == token:
                    ids.append(idx)
                    break
            else:
                ids.append(self.unk_idx)
        
        # Добавляем padding до минимальной длины
        while len(ids) < 10:  # Минимальная длина для LSTM
            ids.append(self.pad_idx)
        
        # Подготавливаем тензор
        input_tensor = torch.tensor([ids], device=next(self.model.parameters()).device)
        
        # Генерация
        generated_ids = self.model.generate(
            input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.eos_idx
        )[0]
        
        # Преобразование в текст
        return self.ids_to_text(generated_ids)