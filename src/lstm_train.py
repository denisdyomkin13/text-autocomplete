import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

from lstm_model import LSTMAutocomplete
from eval_lstm import evaluate_rouge

class LSTMTrainer:
    """Класс для обучения LSTM модели"""
    
    def __init__(self, 
                 model: LSTMAutocomplete,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 clip_grad: float = 1.0):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.clip_grad = clip_grad
        
        # Функция потерь (игнорируем padding токены)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Оптимизатор
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Планировщик скорости обучения
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rouge1': [],
            'val_rouge1': [],
            'train_rouge2': [],
            'val_rouge2': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> float:
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        total_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Обучение")
        for batch_idx, (input_seq, target_seq) in enumerate(pbar):
            # Перемещаем данные на устройство
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Прямой проход
            logits, _ = self.model(input_seq)
            
            # Вычисление потерь
            # Reshape для cross-entropy: [batch_size * seq_len, vocab_size]
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_seq.reshape(-1)
            )
            
            # Обратный проход
            loss.backward()
            
            # Градиентный clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
            # Шаг оптимизации
            self.optimizer.step()
            
            # Обновляем статистику
            total_loss += loss.item()
            total_batches += 1
            
            # Обновляем прогресс-бар
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return total_loss / total_batches
    
    def validate(self) -> float:
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Валидация")
            for batch_idx, (input_seq, target_seq) in enumerate(pbar):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Прямой проход
                logits, _ = self.model(input_seq)
                
                # Вычисление потерь
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_seq.reshape(-1)
                )
                
                total_loss += loss.item()
                total_batches += 1
                
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        return total_loss / total_batches
    
    def train(self, 
              num_epochs: int = 10,
              save_dir: str = 'models',
              save_every: int = 5,
              early_stopping_patience: int = 5):
        """
        Полный цикл обучения
        
        Args:
            num_epochs: количество эпох
            save_dir: директория для сохранения моделей
            save_every: сохранять модель каждые N эпох
            early_stopping_patience: patience для ранней остановки
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Начало обучения на устройстве: {self.device}")
        print(f"Количество параметров: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nЭпоха {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Обучение
            train_loss = self.train_epoch()
            
            # Валидация
            val_loss = self.validate()
            
            # Вычисление метрик ROUGE
            train_rouge = evaluate_rouge(self.model, self.train_loader, self.device)
            val_rouge = evaluate_rouge(self.model, self.val_loader, self.device)
            
            # Обновление планировщика
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  Learning rate уменьшился с {old_lr} до {new_lr}")
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rouge1'].append(train_rouge['rouge1'])
            self.history['val_rouge1'].append(val_rouge['rouge1'])
            self.history['train_rouge2'].append(train_rouge['rouge2'])
            self.history['val_rouge2'].append(val_rouge['rouge2'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Вывод статистики
            print(f"\nСтатистика эпохи {epoch}:")
            print(f"  Потери (train/val): {train_loss:.4f} / {val_loss:.4f}")
            print(f"  ROUGE-1 (train/val): {train_rouge['rouge1']:.4f} / {val_rouge['rouge1']:.4f}")
            print(f"  ROUGE-2 (train/val): {train_rouge['rouge2']:.4f} / {val_rouge['rouge2']:.4f}")
            print(f"  Скорость обучения: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Сохранение лучшей модели
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Сохраняем лучшую модель
                model_path = os.path.join(save_dir, 'lstm_best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history
                }, model_path)
                print(f"  ✓ Сохранена лучшая модель (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Сохранение модели каждые save_every эпох
            if epoch % save_every == 0:
                model_path = os.path.join(save_dir, f'lstm_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history
                }, model_path)
                print(f"  ✓ Сохранена модель эпохи {epoch}")
            
            # Ранняя остановка
            if patience_counter >= early_stopping_patience:
                print(f"\nРанняя остановка на эпохе {epoch}")
                break
        
        # Сохранение финальной модели
        model_path = os.path.join(save_dir, 'lstm_final_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }, model_path)
        
        # Сохранение истории обучения
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nОбучение завершено!")
        print(f"Лучшая val_loss: {best_val_loss:.4f}")
        
        return self.history


def train_lstm_model(config: dict):
    """
    Основная функция для обучения LSTM модели
    
    Args:
        config: конфигурация обучения
    """
    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Загрузка данных
    from next_token_dataset import create_dataloaders
    import pandas as pd
    
    # Загрузка словаря
    vocab_df = pd.read_csv('data/vocab.csv')
    word2idx = dict(zip(vocab_df['word'], vocab_df['index']))
    vocab_size = len(word2idx)
    
    # Создание DataLoader'ов
    dataloaders = create_dataloaders(
        data_dir='data',
        word2idx=word2idx,
        batch_size=config.get('batch_size', 32),
        max_length=config.get('max_length', 50)
    )
    
    # Создание модели
    model = LSTMAutocomplete(
        vocab_size=vocab_size,
        embedding_dim=config.get('embedding_dim', 256),
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        pad_idx=0
    )
    
    # Создание тренера
    trainer = LSTMTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=device,
        learning_rate=config.get('learning_rate', 0.001),
        clip_grad=config.get('clip_grad', 1.0)
    )
    
    # Обучение
    history = trainer.train(
        num_epochs=config.get('num_epochs', 10),
        save_dir='models',
        save_every=config.get('save_every', 5),
        early_stopping_patience=config.get('early_stopping_patience', 5)
    )
    
    return model, history


if __name__ == "__main__":
    # Конфигурация по умолчанию
    config = {
        'batch_size': 64,
        'max_length': 50,
        'embedding_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'clip_grad': 1.0,
        'num_epochs': 15,
        'save_every': 5,
        'early_stopping_patience': 5
    }
    
    # Запуск обучения
    model, history = train_lstm_model(config)