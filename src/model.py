import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import glob
import scipy.io
import gc
import json

# ==================== CONFIGURATION OPTIMISÉE ====================
INPUT_SIZE = 1024
N_FEATURES = 1
CONV1_FILTERS = 8
CONV2_FILTERS = 16
CONV3_FILTERS = 32
LSTM_HIDDEN = 32
OUTPUT_SIZE = 4
TIME_STEPS = 32
FC1_SIZE = 64
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
Q_BITS = 8
FIXED_SCALE = 8
FIXED_SCALE_VAL = 1 << FIXED_SCALE
DROPOUT_RATE = 0.3
MODEL_WEIGHTS_DIR = "firmware/model"
ARRAYS_DIR = os.path.join(MODEL_WEIGHTS_DIR, "arrays")
DATA_DIR = "CWRU_Vibration_Data"

# Classes de défauts CWRU
FAULT_CLASSES = {
    'Normal': 0,
    'Ball_Fault': 1,
    'Inner_Race_Fault': 2,
    'Outer_Race_Fault': 3
}

CLASS_NAMES = list(FAULT_CLASSES.keys())

# ==================== MODÈLE OPTIMISÉ POUR VITESSE ====================

class FastLIFNeuron(nn.Module):
    """LIF neuron optimisé pour la vitesse"""
    def __init__(self, threshold=0.5, decay=0.9, time_steps=32):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.time_steps = time_steps
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        membrane = torch.zeros(batch_size, features, device=x.device)
        spikes = torch.zeros(batch_size, seq_len, features, device=x.device)
        
        for t in range(seq_len):
            membrane = self.decay * membrane + x[:, t, :]
            spike = (membrane > self.threshold).float()
            spikes[:, t, :] = spike
            membrane = membrane * (1 - spike)
            
        return spikes

class FastHybridModel(nn.Module):
    """Modèle hybride optimisé pour l'entraînement rapide"""
    def __init__(self):
        super().__init__()
        
        # CNN simplifié mais efficace - STRIDES AUGMENTÉS
        self.conv1 = nn.Conv1d(N_FEATURES, CONV1_FILTERS, kernel_size=7, stride=4, padding=3)
        self.conv2 = nn.Conv1d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=5, stride=4, padding=2)
        self.conv3 = nn.Conv1d(CONV2_FILTERS, CONV3_FILTERS, kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm1d(CONV1_FILTERS)
        self.bn2 = nn.BatchNorm1d(CONV2_FILTERS)
        self.bn3 = nn.BatchNorm1d(CONV3_FILTERS)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        # Calcul des dimensions (1024 → 256 → 64 → 32)
        self.conv_out_length = 32
        self.conv_out_features = CONV3_FILTERS
        self.snn_input_size = (self.conv_out_length * self.conv_out_features) // TIME_STEPS
        
        # SNN/LSTM
        self.lif = FastLIFNeuron(threshold=0.5, decay=0.9, time_steps=TIME_STEPS)
        self.lstm = nn.LSTM(self.snn_input_size, LSTM_HIDDEN, batch_first=True, num_layers=1)
        
        # Classification rapide
        self.fc1 = nn.Linear(LSTM_HIDDEN + self.conv_out_features, FC1_SIZE)
        self.fc2 = nn.Linear(FC1_SIZE, FC1_SIZE // 2)
        self.fc3 = nn.Linear(FC1_SIZE // 2, OUTPUT_SIZE)
        
        self.dropout_layer = nn.Dropout(DROPOUT_RATE * 0.5)
        
        self.precision_mode = 0
        
        print(f"Model dimensions:")
        print(f"  Conv output: {self.conv_out_length} x {self.conv_out_features}")
        print(f"  SNN input size: {self.snn_input_size}")
        print(f"  FC1 input: {LSTM_HIDDEN + self.conv_out_features}")
    
    def forward(self, x, domain="VIBRATION"):
        batch_size = x.size(0)
        
        # ========== CNN RAPIDE ==========
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        spatial_features = x.mean(dim=2)
        
        # ========== PRÉPARATION SNN ==========
        x = x.permute(0, 2, 1)
        
        total_features = self.conv_out_length * self.conv_out_features
        usable_features = (total_features // TIME_STEPS) * TIME_STEPS
        
        x_flat = x.contiguous().view(batch_size, -1)
        x_flat = x_flat[:, :usable_features]
        snn_input = x_flat.view(batch_size, TIME_STEPS, -1)
        
        # ========== SNN/LSTM ==========
        spikes = self.lif(snn_input)
        lstm_out, _ = self.lstm(spikes)
        temporal_features = lstm_out[:, -1, :]
        
        # ========== FUSION ET CLASSIFICATION ==========
        combined = torch.cat([temporal_features, spatial_features], dim=1)
        
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout_layer(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        output = self.fc3(x)
        
        return output, {
            'spatial_features': spatial_features,
            'temporal_features': temporal_features,
            'precision_mode': self.precision_mode
        }
    
    def update_temperature(self, temp):
        if temp > 70:
            self.precision_mode = 2
        elif temp > 50:
            self.precision_mode = 1
        else:
            self.precision_mode = 0

# ==================== DATASET OPTIMISÉ ====================

class OptimizedCWRUDataset(Dataset):
    """Dataset optimisé avec moins de chevauchement"""
    def __init__(self, data_dir=DATA_DIR, segment_length=INPUT_SIZE, 
                 overlap=0.3, augment=True, max_samples_per_class=200):
        self.segment_length = segment_length
        self.augment = augment
        self.data = []
        self.labels = []
        
        print("Loading optimized dataset...")
        self._load_data_fast(data_dir, overlap, max_samples_per_class)
        print(f"Total samples: {len(self.data)}")
    
    def _load_data_fast(self, data_dir, overlap, max_samples_per_class):
        """Chargement rapide avec limitation"""
        mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
        
        if not mat_files:
            print(f"No .mat files found in {data_dir}")
            return
        
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        step = int(self.segment_length * (1 - overlap))
        
        for mat_file in mat_files:
            if sum(class_counts.values()) >= max_samples_per_class * 4:
                break
                
            filename = os.path.basename(mat_file)
            label = self._get_label_from_filename(filename)
            
            if label is None or class_counts[label] >= max_samples_per_class:
                continue
            
            # Charger données
            try:
                data = scipy.io.loadmat(mat_file)
                for key in data.keys():
                    if 'DE_time' in key and not key.startswith('__'):
                        vibration = data[key].flatten().astype(np.float32)
                        
                        # Normalisation simple
                        if len(vibration) > 0:
                            mean = np.mean(vibration)
                            std = np.std(vibration)
                            if std > 1e-8:
                                vibration = (vibration - mean) / std
                        
                        # Segmenter
                        for i in range(0, len(vibration) - self.segment_length + 1, step):
                            segment = vibration[i:i + self.segment_length]
                            
                            if len(segment) == self.segment_length:
                                self.data.append(segment)
                                self.labels.append(label)
                                class_counts[label] += 1
                                
                                if class_counts[label] >= max_samples_per_class:
                                    break
                        
                        break
            except Exception as e:
                continue
        
        # Convertir en arrays numpy pour plus de rapidité
        if self.data:
            self.data = np.array(self.data, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"Class distribution: {dict(zip(CLASS_NAMES, [class_counts[i] for i in range(4)]))}")
    
    def _get_label_from_filename(self, filename):
        """Identifier la classe rapidement"""
        filename = filename.lower()
        
        # Normal
        if any(x in filename for x in ['97', '98', '99', '156', '157', '158']):
            return 0
        
        # Ball Fault  
        if any(x in filename for x in ['105', '106', '107', '108', '109', '111', '112']):
            return 1
        
        # Inner Race Fault
        if any(x in filename for x in ['122', '124', '130', '131', '132', '133']):
            return 2
        
        # Outer Race Fault
        if any(x in filename for x in ['144', '145', '146', '147', '148', '149']):
            return 3
        
        return None
    
    def _augment_signal_fast(self, signal):
        """Augmentation rapide"""
        signal = signal.copy()
        
        # Scaling
        signal *= np.random.uniform(0.9, 1.1)
        
        # Bruit léger
        signal += np.random.normal(0, 0.01, len(signal))
        
        # Time shift occasionnel
        if np.random.random() < 0.3:
            shift = np.random.randint(-5, 5)
            if shift > 0:
                signal = np.concatenate([signal[shift:], signal[:shift]])
            elif shift < 0:
                shift = abs(shift)
                signal = np.concatenate([signal[-shift:], signal[:-shift]])
        
        return signal
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        signal = self.data[idx]
        
        if self.augment:
            signal = self._augment_signal_fast(signal)
        
        # Normalisation finale
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        if signal_std > 1e-8:
            signal = (signal - signal_mean) / signal_std
        
        signal_tensor = torch.FloatTensor(signal).unsqueeze(-1)
        
        return signal_tensor, self.labels[idx]

# ==================== ENTRAÎNEMENT RAPIDE ====================

def get_labels_from_subset(subset):
    """Récupérer les labels d'un Subset"""
    if hasattr(subset.dataset, 'labels'):
        indices = subset.indices
        return subset.dataset.labels[indices]
    else:
        labels = []
        for i in range(len(subset)):
            _, label = subset[i]
            labels.append(label)
        return np.array(labels)

def train_fast(model, train_loader, val_loader, epochs=EPOCHS):
    """Entraînement optimisé pour la vitesse"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Training on: {device}")
    
    # Calculer les poids des classes depuis le train loader
    train_labels = get_labels_from_subset(train_loader.dataset)
    class_counts = np.bincount(train_labels, minlength=OUTPUT_SIZE)
    total_samples = len(train_labels)
    class_weights = total_samples / (OUTPUT_SIZE * (class_counts + 1))
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class weights: {class_weights}")
    
    # Optimiseur
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler simple
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LEARNING_RATE/50)
    
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': [], 'train_acc': []}
    
    print("\n" + "="*60)
    print("FAST TRAINING STARTING")
    print(f"Epochs: {epochs}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
    print("="*60)
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calcul d'accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            # Affichage de progression
            if (batch_idx + 1) % 10 == 0:
                batch_acc = 100. * (pred == target).sum().item() / target.size(0)
                print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} - Acc: {batch_acc:.1f}%")
        
        # Mettre à jour le scheduler
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)
        
        train_acc = 100. * correct / total if total > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        # Sauvegarder si amélioration
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'optimizer_state_dict': optimizer.state_dict()
            }, 'best_fast_model_cwru.pth')
            print(f"  💾 Model saved! Val Acc: {val_acc:.2f}%")
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['train_acc'].append(train_acc)
        
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%, Time: {epoch_time:.1f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping conditionnelle
        if epoch >= 5 and val_acc < best_acc - 5.0:
            print(f"  ⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # Charger le meilleur modèle
    if os.path.exists('best_fast_model_cwru.pth'):
        checkpoint = torch.load('best_fast_model_cwru.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✅ Best model loaded from epoch {checkpoint['epoch']+1}")
        print(f"   Training Accuracy: {checkpoint['train_acc']:.2f}%")
        print(f"   Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model, history

# ==================== FONCTIONS UTILITAIRES RAPIDES ====================

def evaluate_fast(model, loader):
    """Évaluation rapide"""
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = 100. * np.mean(all_preds == all_targets)
    
    print("\n" + "="*60)
    print("FAST EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {acc:.2f}%")
    print(f"Total samples: {len(all_targets)}")
    
    print("\nClass-wise Performance:")
    for i, class_name in enumerate(CLASS_NAMES):
        mask = (all_targets == i)
        if np.any(mask):
            class_acc = 100. * np.mean(all_preds[mask] == i)
            class_total = np.sum(mask)
            print(f"  {class_name:20s}: {class_acc:6.2f}% ({class_total} samples)")
    
    print("\n" + classification_report(all_targets, all_preds, 
                                      target_names=CLASS_NAMES, digits=3))
    
    return acc, all_preds, all_targets

# ==================== EXPORT COMPLET POUR RISC-V ====================

def quantize_tensor(tensor, bits=Q_BITS):
    """Quantifier un tenseur avec la précision spécifiée"""
    max_val = max(tensor.max().item(), -tensor.min().item())
    scale = max_val / (2**(bits-1)-1) if max_val > 0 else 1.0
    quantized = torch.clamp(torch.round(tensor/scale), -2**(bits-1), 2**(bits-1)-1).int()
    return quantized, scale

def float_to_fixed(f, scale=FIXED_SCALE_VAL):
    """Convertir un float en représentation fixed-point"""
    return int(round(f * scale))

def export_weight_arrays_complete(model, output_dir=ARRAYS_DIR):
    """Export complet des poids pour RISC-V"""
    os.makedirs(output_dir, exist_ok=True)
    
    layers = [
        ('conv1.weight', 'conv1_weight'),
        ('conv1.bias', 'conv1_bias'),
        ('conv2.weight', 'conv2_weight'),
        ('conv2.bias', 'conv2_bias'),
        ('conv3.weight', 'conv3_weight'),
        ('conv3.bias', 'conv3_bias'),
        ('lstm.weight_ih_l0', 'lstm_weight_ih'),
        ('lstm.weight_hh_l0', 'lstm_weight_hh'),
        ('lstm.bias_ih_l0', 'lstm_bias_ih'),
        ('lstm.bias_hh_l0', 'lstm_bias_hh'),
        ('fc1.weight', 'fc1_weight'),
        ('fc1.bias', 'fc1_bias'),
        ('fc2.weight', 'fc2_weight'),
        ('fc2.bias', 'fc2_bias'),
        ('fc3.weight', 'fc3_weight'),
        ('fc3.bias', 'fc3_bias')
    ]
    
    scales = {}
    total_params = 0
    
    print("\n" + "="*60)
    print("EXPORTING COMPLETE WEIGHTS FOR RISC-V")
    print("="*60)
    
    for param_name, var_name in layers:
        try:
            param = dict(model.named_parameters())[param_name]
            quantized, scale = quantize_tensor(param.data)
            
            param_count = quantized.numel()
            total_params += param_count
            
            # Écrire le fichier array
            with open(f"{output_dir}/{var_name}_array.txt", "w") as f:
                values = quantized.numpy().flatten().tolist()
                for i in range(0, len(values), 12):
                    f.write(", ".join(map(str, values[i:i+12])) + ",\n")
            
            scales[var_name] = float_to_fixed(scale)
            print(f"  ✓ {var_name:20s} [{param_count:5d} params] scale={scales[var_name]}")
            
        except KeyError:
            print(f"  ✗ {var_name:20s} not found (skipping)")
            continue
    
    print("-" * 60)
    print(f"  Total parameters exported: {total_params:,}")
    
    return scales

def generate_model_weights_header(scales, model):
    """Générer le fichier header C complet"""
    conv1_out_size = INPUT_SIZE // 4  # stride=4
    conv2_out_size = conv1_out_size // 4  # stride=4
    conv3_out_size = conv2_out_size // 2  # stride=2
    snn_input_size = model.snn_input_size
    
    header_content = f"""#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

// ==================== MODEL CONFIGURATION ====================
#define INPUT_SIZE {INPUT_SIZE}
#define N_FEATURES {N_FEATURES}
#define CONV1_FILTERS {CONV1_FILTERS}
#define CONV2_FILTERS {CONV2_FILTERS}
#define CONV3_FILTERS {CONV3_FILTERS}
#define LSTM_HIDDEN {LSTM_HIDDEN}
#define OUTPUT_SIZE {OUTPUT_SIZE}
#define TIME_STEPS {TIME_STEPS}
#define FC1_SIZE {FC1_SIZE}

// Calculated sizes
#define CONV1_OUT_SIZE {conv1_out_size}
#define CONV2_OUT_SIZE {conv2_out_size}
#define CONV3_OUT_SIZE {conv3_out_size}
#define SNN_INPUT_SIZE {snn_input_size}
#define TOTAL_CONV3_FEATURES (CONV3_OUT_SIZE * CONV3_FILTERS)
#define USABLE_FEATURES ((TOTAL_CONV3_FEATURES / TIME_STEPS) * TIME_STEPS)

// RV32X-SQ Extensions
#define RV32X_CUSTOM1  0x0B  // 4-bit MAC
#define RV32X_CUSTOM2  0x0C  // LIF neuron update
#define RV32X_CUSTOM3  0x0D  // Attention fusion

// Thermal management
#define TEMP_THRESHOLD_HIGH   70
#define TEMP_THRESHOLD_MEDIUM 50
#define PRECISION_HIGH        0  // 8-bit
#define PRECISION_MEDIUM      1  // 4-bit
#define PRECISION_LOW         2  // 2-bit

// Fault classes
#define CLASS_NORMAL 0
#define CLASS_BALL_FAULT 1
#define CLASS_INNER_RACE_FAULT 2
#define CLASS_OUTER_RACE_FAULT 3

// Fixed-point configuration
#define Q_BITS {Q_BITS}
#define FIXED_SCALE {FIXED_SCALE}
#define FIXED_SCALE_VAL {FIXED_SCALE_VAL}

// ==================== BUFFER STRUCTURE ====================
typedef struct {{
    // Input buffer (1 channel, 1024 samples)
    int8_t input_buf[INPUT_SIZE];
    
    // Spatial path (CNN)
    int32_t conv1_out[CONV1_FILTERS * CONV1_OUT_SIZE];
    int32_t conv2_out[CONV2_FILTERS * CONV2_OUT_SIZE];
    int32_t conv3_out[CONV3_FILTERS * CONV3_OUT_SIZE];
    
    // Temporal path (SNN/LSTM)
    int8_t snn_input[SNN_INPUT_SIZE * TIME_STEPS];
    int8_t spike_train[TIME_STEPS * LSTM_HIDDEN];
    int8_t lstm_state[LSTM_HIDDEN];
    
    // Fusion buffer
    int32_t fc1_out[FC1_SIZE];
    int32_t fc2_out[FC1_SIZE / 2];
    
    // Output buffer
    int32_t output[OUTPUT_SIZE];
    
    // Thermal management
    uint8_t precision_mode;
    int16_t temperature;
    
    // Benchmarking
    uint32_t total_cycles;
    uint32_t inference_count;
}} VibrationModelBuffers;

// ==================== WEIGHT DECLARATIONS ====================
"""

    # Ajouter les déclarations de poids
    weight_sizes = {
        'conv1_weight': CONV1_FILTERS * N_FEATURES * 7,
        'conv1_bias': CONV1_FILTERS,
        'conv2_weight': CONV2_FILTERS * CONV1_FILTERS * 5,
        'conv2_bias': CONV2_FILTERS,
        'conv3_weight': CONV3_FILTERS * CONV2_FILTERS * 3,
        'conv3_bias': CONV3_FILTERS,
        'lstm_weight_ih': 4 * LSTM_HIDDEN * snn_input_size,
        'lstm_weight_hh': 4 * LSTM_HIDDEN * LSTM_HIDDEN,
        'lstm_bias_ih': 4 * LSTM_HIDDEN,
        'lstm_bias_hh': 4 * LSTM_HIDDEN,
        'fc1_weight': FC1_SIZE * (LSTM_HIDDEN + CONV3_FILTERS),
        'fc1_bias': FC1_SIZE,
        'fc2_weight': (FC1_SIZE // 2) * FC1_SIZE,
        'fc2_bias': FC1_SIZE // 2,
        'fc3_weight': OUTPUT_SIZE * (FC1_SIZE // 2),
        'fc3_bias': OUTPUT_SIZE
    }
    
    for var_name, size in weight_sizes.items():
        if var_name in scales:
            header_content += f"extern const int8_t {var_name}[{size}];\n"
            header_content += f"extern const int32_t {var_name}_scale;\n\n"
    
    header_content += """// ==================== FUNCTION PROTOTYPES ====================
void model_init(VibrationModelBuffers* buffers);
void model_predict(VibrationModelBuffers* buffers, const int8_t* input);
void thermal_management(VibrationModelBuffers* buffers);

// RV32X-SQ Custom Instructions
int32_t custom1_mac(int8_t a, int8_t b, int32_t acc);
int8_t custom2_lif(int8_t input, int8_t membrane_potential, int8_t threshold);
int8_t custom3_fusion(int8_t snn_out, int8_t cnn_out, int8_t attention_weight);

// Utility functions
const char* get_fault_name(uint8_t fault_class);
uint8_t detect_fault(const int8_t* vibration_signal, uint32_t length);

#endif // MODEL_WEIGHTS_H
"""

    with open(f"{MODEL_WEIGHTS_DIR}/model_weights.h", "w") as f:
        f.write(header_content)
    
    print(f"✅ Generated {MODEL_WEIGHTS_DIR}/model_weights.h")
    
    return weight_sizes

def generate_model_weights_source(scales, weight_sizes):
    """Générer le fichier source C complet"""
    source_content = f"""#include "model_weights.h"

// ==================== FAULT NAMES ====================
const char* get_fault_name(uint8_t fault_class) {{
    switch(fault_class) {{
        case CLASS_NORMAL:
            return "Normal";
        case CLASS_BALL_FAULT:
            return "Ball_Fault";
        case CLASS_INNER_RACE_FAULT:
            return "Inner_Race_Fault";
        case CLASS_OUTER_RACE_FAULT:
            return "Outer_Race_Fault";
        default:
            return "Unknown";
    }}
}}

// ==================== WEIGHT DEFINITIONS ====================
"""

    for var_name, size in weight_sizes.items():
        if var_name in scales:
            source_content += f"""const int8_t {var_name}[] = {{
    #include "arrays/{var_name}_array.txt"
}};
const int32_t {var_name}_scale = {scales[var_name]};\n\n"""
    
    with open(f"{MODEL_WEIGHTS_DIR}/model_weights.c", "w") as f:
        f.write(source_content)
    
    print(f"✅ Generated {MODEL_WEIGHTS_DIR}/model_weights.c")

def generate_test_data():
    """Générer des données de test pour le firmware"""
    # Créer un signal de vibration synthétique
    t = np.linspace(0, 1, INPUT_SIZE)
    
    # Signal normal
    normal_signal = 0.5 * np.sin(2 * np.pi * 30 * t) + 0.2 * np.sin(2 * np.pi * 60 * t)
    
    # Signal avec défaut de bille
    ball_fault_signal = normal_signal + 0.3 * np.sin(2 * np.pi * 120 * t)
    
    # Normaliser
    normal_signal = (normal_signal - np.mean(normal_signal)) / np.std(normal_signal)
    ball_fault_signal = (ball_fault_signal - np.mean(ball_fault_signal)) / np.std(ball_fault_signal)
    
    # Convertir en int8
    normal_int8 = np.clip(np.round(normal_signal * 64), -128, 127).astype(np.int8)
    ball_fault_int8 = np.clip(np.round(ball_fault_signal * 64), -128, 127).astype(np.int8)
    
    # Écrire le fichier de test
    os.makedirs("firmware", exist_ok=True)
    
    with open("firmware/vibration_test_data.h", "w") as f:
        f.write("#ifndef VIBRATION_TEST_DATA_H\n")
        f.write("#define VIBRATION_TEST_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        f.write("// Test vibration signals for firmware testing\n")
        f.write(f"const int8_t vibration_test_normal[{INPUT_SIZE}] = {{\n    ")
        for i in range(0, INPUT_SIZE, 12):
            f.write(", ".join(map(str, normal_int8[i:i+12])) + ",\n    ")
        f.write("\n};\n\n")
        
        f.write(f"const int8_t vibration_test_ball_fault[{INPUT_SIZE}] = {{\n    ")
        for i in range(0, INPUT_SIZE, 12):
            f.write(", ".join(map(str, ball_fault_int8[i:i+12])) + ",\n    ")
        f.write("\n};\n\n")
        
        f.write("#endif // VIBRATION_TEST_DATA_H\n")
    
    print("✅ Generated firmware/vibration_test_data.h")

def generate_model_info(model, accuracy, total_params):
    """Générer un fichier JSON avec les infos du modèle"""
    model_info = {
        "name": "FastHybridVibrationModel",
        "input_size": INPUT_SIZE,
        "input_channels": N_FEATURES,
        "output_classes": OUTPUT_SIZE,
        "architecture": {
            "conv1_filters": CONV1_FILTERS,
            "conv2_filters": CONV2_FILTERS,
            "conv3_filters": CONV3_FILTERS,
            "lstm_hidden": LSTM_HIDDEN,
            "time_steps": TIME_STEPS,
            "fc1_size": FC1_SIZE
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "validation_accuracy": accuracy
        },
        "quantization": {
            "bits": Q_BITS,
            "fixed_scale": FIXED_SCALE
        },
        "statistics": {
            "total_parameters": int(total_params),
            "model_size_kb": float(total_params * Q_BITS / (8 * 1024))
        },
        "fault_classes": FAULT_CLASSES,
        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{MODEL_WEIGHTS_DIR}/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Generated {MODEL_WEIGHTS_DIR}/model_info.json")

def quick_demo(model, test_loader):
    """Démonstration rapide"""
    device = next(model.parameters()).device
    model.eval()
    
    # Prendre quelques échantillons
    data, targets = next(iter(test_loader))
    data, targets = data[:3].to(device), targets[:3]
    
    with torch.no_grad():
        outputs, metadata = model(data)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
    
    print("\n" + "="*60)
    print("QUICK DEMONSTRATION")
    print("="*60)
    
    for i in range(min(3, len(data))):
        true_label = CLASS_NAMES[targets[i].item()]
        pred_label = CLASS_NAMES[preds[i].item()]
        confidence = probs[i, preds[i]].item() * 100
        
        print(f"\nSample {i+1}:")
        print(f"  True: {true_label}")
        print(f"  Pred: {pred_label} ({confidence:.1f}%)")
        
        # Afficher toutes les probabilités
        for j, class_name in enumerate(CLASS_NAMES):
            prob = probs[i, j].item() * 100
            if prob > 5.0:  # Afficher seulement les probabilités > 5%
                print(f"    {class_name:20s}: {prob:5.1f}%")

# ==================== FONCTION PRINCIPALE ====================

def main_fast():
    """Version principale optimisée pour la vitesse"""
    start_time = time.time()
    
    print("="*70)
    print("FAST VIBRATION ANALYSIS - CWRU DATASET")
    print("OPTIMIZED FOR SPEED AND EFFICIENCY")
    print("="*70)
    
    # 1. Charger les données rapidement
    print("\n1. LOADING OPTIMIZED DATASET...")
    try:
        dataset = OptimizedCWRUDataset(
            data_dir=DATA_DIR,
            segment_length=INPUT_SIZE,
            overlap=0.3,
            augment=True,
            max_samples_per_class=200
        )
        
        if len(dataset) == 0:
            print("❌ No data loaded!")
            return
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Split avec stratification
        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=0.2,
            random_state=42,
            stratify=dataset.labels
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        # DataLoaders optimisés
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Créer le modèle
    print("\n2. CREATING FAST MODEL...")
    model = FastHybridModel()
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size_kb = total_params * Q_BITS / (8 * 1024)
    
    print(f"✅ Parameters: {total_params:,}")
    print(f"✅ Model size: {model_size_kb:.2f} KB")
    print(f"✅ Architecture: CNN({CONV1_FILTERS}→{CONV2_FILTERS}→{CONV3_FILTERS}) + LSTM({LSTM_HIDDEN})")
    
    # 3. Entraînement rapide
    print(f"\n3. FAST TRAINING (Max {EPOCHS} epochs)...")
    print("-" * 60)
    
    model, history = train_fast(model, train_loader, val_loader, epochs=EPOCHS)
    
    # 4. Évaluation finale
    print("\n4. FINAL EVALUATION...")
    final_acc, all_preds, all_targets = evaluate_fast(model, val_loader)
    
    # 5. Démonstration
    print("\n5. QUICK DEMONSTRATION...")
    quick_demo(model, val_loader)
    
    # 6. Export complet pour RISC-V
    print("\n6. COMPLETE RISC-V EXPORT...")
    
    # Créer les répertoires
    os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(ARRAYS_DIR, exist_ok=True)
    
    # Exporter les poids
    scales = export_weight_arrays_complete(model)
    
    # Générer les fichiers C
    weight_sizes = generate_model_weights_header(scales, model)
    generate_model_weights_source(scales, weight_sizes)
    
    # Générer les données de test
    generate_test_data()
    
    # Générer les infos du modèle
    generate_model_info(model, final_acc, total_params)
    
    # 7. Résumé final
    total_time = time.time() - start_time
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"✅ Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"✅ Final Validation Accuracy: {final_acc:.2f}%")
    print(f"✅ Best Model Saved: 'best_fast_model_cwru.pth'")
    print(f"✅ Total Training Time: {total_time/60:.1f} minutes")
    print(f"✅ RISC-V Files Generated in 'firmware/' directory:")
    print(f"   - firmware/model/model_weights.h")
    print(f"   - firmware/model/model_weights.c")
    print(f"   - firmware/model/arrays/*.txt")
    print(f"   - firmware/vibration_test_data.h")
    print(f"   - firmware/model/model_info.json")
    
    # Afficher la courbe d'apprentissage
    print(f"\n📈 Learning Curve:")
    print(f"  Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Best Training Accuracy: {max(history['train_acc']):.2f}%")
    print(f"  Epochs completed: {len(history['train_loss'])}")
    
    print("="*70)
    
    # Informations sur le déploiement
    print("\n🚀 READY FOR RISC-V DEPLOYMENT!")
    print("To compile for RISC-V:")
    print("1. Copy 'firmware/' directory to your RISC-V project")
    print("2. Include 'model_weights.h' in your main.c")
    print("3. Implement the buffer management functions")
    print("4. Use model_predict() for inference")

# ==================== ENTRÉE DU PROGRAMME ====================

if __name__ == "__main__":
    # Configurations pour la performance
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
    
    # Nettoyer la mémoire
    gc.collect()
    
    # Exécuter
    main_fast()
