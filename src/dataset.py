import os
import glob
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T

class ICBHIDataset(Dataset):
    def __init__(self, data_dir, is_train=True, max_pad_len=862, max_samples_per_class=None):
        """
        ICBHI Respiratory Sound Dataset Loader
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.max_pad_len = max_pad_len
        
        # Load diagnosis mapping
        # ICBHI format: Patient_ID, Diagnosis
        diagnosis_path = os.path.join(data_dir, 'patient_diagnosis.csv')
        if not os.path.exists(diagnosis_path):
            print(f"Warning: {diagnosis_path} not found. Ensure dataset is fully downloaded.")
            self.patient_diagnosis = {}
        else:
            df = pd.read_csv(diagnosis_path, names=['Patient_ID', 'Diagnosis'])
            self.patient_diagnosis = dict(zip(df['Patient_ID'], df['Diagnosis']))
            
        # Class mapping to integers
        self.class_mapping = {
            'Healthy': 0, 'Normal': 0,
            'Asthma': 1,
            'Pneumonia': 2,
            'COPD': 3,
            'Bronchiectasis': 4, 'Bronchitis': 4,
            'URTI': 5, 'LRTI': 5, 'COVID-19': 5 # Grouping respiratory infections
        }
        
        # Get all wav files
        self.wav_files = glob.glob(os.path.join(data_dir, '*.wav'))
        
        # Filter files that have a valid diagnosis
        self.valid_files = []
        self.labels = []
        class_counts = {}
        for f in self.wav_files:
            patient_id = int(os.path.basename(f).split('_')[0])
            diagnosis = self.patient_diagnosis.get(patient_id, 'Normal')
            class_idx = self.class_mapping.get(diagnosis, 0)
            
            # Apply per-class cap if specified
            if max_samples_per_class is not None:
                current = class_counts.get(class_idx, 0)
                if current >= max_samples_per_class:
                    continue
                class_counts[class_idx] = current + 1
            
            self.valid_files.append(f)
            self.labels.append(class_idx)
            
        # SpecAugment transforms
        self.time_masking = T.TimeMasking(time_mask_param=30)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)

    def __len__(self):
        return len(self.valid_files)

    def extract_features(self, file_path):
        """Extract log-mel spectrogram features"""
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=22050, duration=10.0) # Limit to 10s max
            
            # Extract Mel Spectrogram
            melspectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmin=50, fmax=4000
            )
            log_mel = librosa.power_to_db(melspectrogram, ref=np.max)
            
            # Padding / Truncation
            if log_mel.shape[1] > self.max_pad_len:
                log_mel = log_mel[:, :self.max_pad_len]
            else:
                pad_width = self.max_pad_len - log_mel.shape[1]
                log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)), mode='constant')
                
            # Normalize
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
            
            # Convert to tensor and add channel dim [1, 128, max_pad_len]
            tensor_feature = torch.FloatTensor(log_mel).unsqueeze(0)
            
            # Apply SpecAugment during training
            if self.is_train:
                tensor_feature = self.time_masking(tensor_feature)
                tensor_feature = self.freq_masking(tensor_feature)
                
            return tensor_feature
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            # Return zero tensor as fallback
            return torch.zeros((1, 128, self.max_pad_len))

    def __getitem__(self, idx):
        file_path = self.valid_files[idx]
        label = self.labels[idx]
        
        features = self.extract_features(file_path)
        
        # Convert grayscale (1 channel) to RGB (3 channels) for ResNet
        features = features.repeat(3, 1, 1)
        
        return features, torch.tensor(label, dtype=torch.long)


class CoswaraDataset(Dataset):
    def __init__(self, data_dir, is_train=True, max_pad_len=862, max_normal_samples=200):
        """
        Coswara Dataset Loader for COVID-19 detection
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.max_pad_len = max_pad_len
        
        csv_path = os.path.join(data_dir, 'combined_data.csv')
        df = pd.read_csv(csv_path)
        
        covid_statuses = ['positive_mild', 'positive_moderate', 'positive_asymp']
        normal_statuses = ['healthy']
        
        self.valid_files = []
        self.labels = []
        
        extracted_dir = os.path.join(data_dir, 'Extracted_data')
        
        normal_count = 0
        for idx, row in df.iterrows():
            uid = row['id']
            status = row['covid_status']
            date = str(row['record_date']).replace('-', '')
            
            if status in covid_statuses:
                label = 5 # COVID-19
            elif status in normal_statuses:
                # Cap normal samples to avoid overwhelming the model
                if normal_count >= max_normal_samples:
                    continue
                label = 0 # Normal
                normal_count += 1
            else:
                continue
                
            # Use deep breathing audio which closely matches ICBHI general audio
            wav_path = os.path.join(extracted_dir, date, str(uid), 'breathing-deep.wav')
            if os.path.exists(wav_path):
                self.valid_files.append(wav_path)
                self.labels.append(label)
                
        self.time_masking = T.TimeMasking(time_mask_param=30)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)

    def __len__(self):
        return len(self.valid_files)

    def extract_features(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=22050, duration=10.0)
            melspectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmin=50, fmax=4000
            )
            log_mel = librosa.power_to_db(melspectrogram, ref=np.max)
            
            if log_mel.shape[1] > self.max_pad_len:
                log_mel = log_mel[:, :self.max_pad_len]
            else:
                pad_width = self.max_pad_len - log_mel.shape[1]
                log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)), mode='constant')
                
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
            tensor_feature = torch.FloatTensor(log_mel).unsqueeze(0)
            
            if self.is_train:
                tensor_feature = self.time_masking(tensor_feature)
                tensor_feature = self.freq_masking(tensor_feature)
                
            return tensor_feature
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return torch.zeros((1, 128, self.max_pad_len))

    def __getitem__(self, idx):
        file_path = self.valid_files[idx]
        label = self.labels[idx]
        features = self.extract_features(file_path)
        features = features.repeat(3, 1, 1)
        return features, torch.tensor(label, dtype=torch.long)

def get_class_weights(dataset):
    """Calculate weights for imbalanced classes"""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        labels = []
        for d in dataset.datasets:
            labels.extend(d.dataset.labels if hasattr(d, 'dataset') else d.labels)
    else:
        labels = dataset.dataset.labels if hasattr(dataset, 'dataset') else dataset.labels
        
    class_counts = np.bincount(labels, minlength=6)
    
    # Avoid division by zero for unrepresented classes
    class_counts[class_counts == 0] = 1 
    
    total = len(labels)
    # Inverse frequency weighting
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)
