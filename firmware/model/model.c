#include "firmware/model/model_weights.h"
#include "ops/math_ops.h"
#include "utils/memutils.h"
#include "utils/cycle_count.h"

// Fonction utilitaire de saturation
static inline int8_t saturate_int32_to_int8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

// Initialisation du modèle CWRU
void model_init(VibrationModelBuffers* buffers) {
    if (buffers == NULL) return;
    
    // Initialize all buffers to zero
    memset(buffers->input_buf, 0, INPUT_SIZE);
    memset(buffers->conv1_out, 0, CONV1_FILTERS * CONV1_OUT_SIZE * sizeof(int32_t));
    memset(buffers->conv2_out, 0, CONV2_FILTERS * CONV2_OUT_SIZE * sizeof(int32_t));
    memset(buffers->conv3_out, 0, CONV3_FILTERS * CONV3_OUT_SIZE * sizeof(int32_t));
    memset(buffers->snn_input, 0, SNN_INPUT_SIZE * TIME_STEPS);
    memset(buffers->spike_train, 0, TIME_STEPS * LSTM_HIDDEN);
    memset(buffers->lstm_state, 0, LSTM_HIDDEN);
    memset(buffers->fc1_out, 0, FC1_SIZE * sizeof(int32_t));
    memset(buffers->fc2_out, 0, (FC1_SIZE / 2) * sizeof(int32_t));
    memset(buffers->output, 0, OUTPUT_SIZE * sizeof(int32_t));
    
    // Initialize thermal management
    buffers->precision_mode = PRECISION_HIGH;
    buffers->temperature = 25;
    buffers->total_cycles = 0;
    buffers->inference_count = 0;
}

// MAC 8-bit simple pour CWRU
static inline int32_t mac_8bit(const int8_t* a, const int8_t* b, int32_t size, int32_t acc) {
    for (int i = 0; i < size; i++) {
        acc += a[i] * b[i];
    }
    return acc;
}

// Convolution 1D optimisée pour CWRU
static void conv1d_cwru(const int8_t* input, const int8_t* weights, const int8_t* bias,
                       int32_t* output, int input_len, int filters, int kernel_size, 
                       int stride, int32_t weight_scale, int32_t bias_scale,
                       uint8_t precision_mode) {
    int output_len = input_len / stride;
    
    for (int f = 0; f < filters; f++) {
        for (int o = 0; o < output_len; o++) {
            int32_t acc = 0;
            int input_idx = o * stride;
            
            for (int k = 0; k < kernel_size; k++) {
                if (input_idx + k < input_len) {
                    // Use custom MAC for efficiency
                    if (precision_mode >= PRECISION_MEDIUM) {
                        acc = custom1_mac(input[input_idx + k], weights[f * kernel_size + k], acc);
                    } else {
                        acc += input[input_idx + k] * weights[f * kernel_size + k];
                    }
                }
            }
            
            // Apply scaling based on precision
            int32_t scale_factor = weight_scale;
            if (precision_mode == PRECISION_LOW) {
                scale_factor = scale_factor >> 1;  // Reduce precision
            } else if (precision_mode == PRECISION_MEDIUM) {
                scale_factor = (scale_factor * 3) >> 2;  // ~0.75 scale
            }
            
            acc = ((int64_t)acc * scale_factor) >> FIXED_SCALE;
            
            // Add bias
            if (bias != NULL) {
                acc += ((int32_t)bias[f] * bias_scale) >> FIXED_SCALE;
            }
            
            // ReLU activation
            if (acc < 0) acc = 0;
            
            // Saturation
            if (acc > (int32_t)127) acc = 127;
            if (acc < (int32_t)-128) acc = -128;
            
            output[f * output_len + o] = acc;
        }
    }
}

// LSTM simplifiée pour CWRU
static void lstm_layer_cwru(const int8_t* input, int8_t* hidden_state,
                           const int8_t* w_ih, const int8_t* w_hh,
                           const int8_t* b_ih, const int8_t* b_hh,
                           int input_size, int hidden_size,
                           uint8_t precision_mode) {
    // Temporary state for calculations
    int8_t tmp_state[hidden_size];
    
    for (int i = 0; i < hidden_size; i++) {
        int32_t acc_ih = 0;
        int32_t acc_hh = 0;
        
        // Input to hidden
        for (int j = 0; j < input_size; j++) {
            if (precision_mode >= PRECISION_MEDIUM) {
                acc_ih = custom1_mac(input[j], w_ih[i * input_size + j], acc_ih);
            } else {
                acc_ih += input[j] * w_ih[i * input_size + j];
            }
        }
        
        // Hidden to hidden
        for (int j = 0; j < hidden_size; j++) {
            if (precision_mode >= PRECISION_MEDIUM) {
                acc_hh = custom1_mac(hidden_state[j], w_hh[i * hidden_size + j], acc_hh);
            } else {
                acc_hh += hidden_state[j] * w_hh[i * hidden_size + j];
            }
        }
        
        // Combine and add bias
        int32_t sum = acc_ih + acc_hh;
        sum += (b_ih[i] + b_hh[i]) << 3;  // Scale bias
        
        // Apply LIF neuron dynamics for temporal processing
        if (precision_mode >= PRECISION_MEDIUM) {
            tmp_state[i] = custom2_lif(saturate_int32_to_int8(sum >> 5), hidden_state[i], 50);
        } else {
            // Simple activation with proper saturation
            tmp_state[i] = saturate_int32_to_int8(sum >> 5);
        }
    }
    
    // Update hidden state
    for (int i = 0; i < hidden_size; i++) {
        hidden_state[i] = tmp_state[i];
    }
}

// Fully connected layer for CWRU
static void fc_layer_cwru(const int8_t* input, const int8_t* weights, const int8_t* bias,
                         int32_t* output, int input_size, int output_size,
                         int32_t weight_scale, int32_t bias_scale,
                         uint8_t precision_mode) {
    for (int i = 0; i < output_size; i++) {
        int32_t acc = 0;
        
        for (int j = 0; j < input_size; j++) {
            if (precision_mode >= PRECISION_MEDIUM) {
                acc = custom1_mac(input[j], weights[i * input_size + j], acc);
            } else {
                acc += input[j] * weights[i * input_size + j];
            }
        }
        
        // Apply weight scaling
        int32_t scale_factor = weight_scale;
        if (precision_mode == PRECISION_LOW) {
            scale_factor = scale_factor >> 1;
        }
        
        acc = ((int64_t)acc * scale_factor) >> FIXED_SCALE;
        
        // Add bias
        if (bias != NULL) {
            acc += ((int32_t)bias[i] * bias_scale) >> FIXED_SCALE;
        }
        
        // ReLU activation
        if (acc < 0) acc = 0;
        
        // Saturation for next layer
        if (acc > (int32_t)127) acc = 127;
        if (acc < (int32_t)-128) acc = -128;
        
        output[i] = acc;
    }
}

// Gestion thermique pour CWRU
void thermal_management(VibrationModelBuffers* buffers) {
    if (buffers == NULL) return;
    
    // Simple temperature simulation (increment slowly)
    static uint32_t thermal_counter = 0;
    thermal_counter++;
    
    if (thermal_counter % 100 == 0) {
        buffers->temperature += 1;
        if (buffers->temperature > 85) buffers->temperature = 25;
    }
    
    // Adjust precision based on temperature
    if (buffers->temperature > TEMP_THRESHOLD_HIGH) {
        buffers->precision_mode = PRECISION_LOW;
    } else if (buffers->temperature > TEMP_THRESHOLD_MEDIUM) {
        buffers->precision_mode = PRECISION_MEDIUM;
    } else {
        buffers->precision_mode = PRECISION_HIGH;
    }
}

// Prédiction principale CWRU
void model_predict(VibrationModelBuffers* buffers, const int8_t* input) {
    if (buffers == NULL || input == NULL) return;
    
    #ifdef ENABLE_BENCHMARKING
    cycle_t start_cycles = get_cycle_count();
    #endif
    
    // 1. Apply thermal management
    thermal_management(buffers);
    
    // 2. Copy input
    for (int i = 0; i < INPUT_SIZE; i++) {
        buffers->input_buf[i] = input[i];
    }
    
    // 3. Convolution 1 (stride=4, kernel=7)
    conv1d_cwru(buffers->input_buf, conv1_weight, conv1_bias,
               buffers->conv1_out, INPUT_SIZE, CONV1_FILTERS, 7, 4,
               conv1_weight_scale, conv1_bias_scale, buffers->precision_mode);
    
    // 4. Convolution 2 (stride=4, kernel=5)
    conv1d_cwru((int8_t*)buffers->conv1_out, conv2_weight, conv2_bias,
               buffers->conv2_out, CONV1_OUT_SIZE, CONV2_FILTERS, 5, 4,
               conv2_weight_scale, conv2_bias_scale, buffers->precision_mode);
    
    // 5. Convolution 3 (stride=2, kernel=3)
    conv1d_cwru((int8_t*)buffers->conv2_out, conv3_weight, conv3_bias,
               buffers->conv3_out, CONV2_OUT_SIZE, CONV3_FILTERS, 3, 2,
               conv3_weight_scale, conv3_bias_scale, buffers->precision_mode);
    
    // 6. Préparer pour SNN/LSTM
    int snn_idx = 0;
    for (int t = 0; t < TIME_STEPS; t++) {
        for (int f = 0; f < SNN_INPUT_SIZE; f++) {
            if (snn_idx < TOTAL_CONV3_FEATURES) {
                buffers->snn_input[t * SNN_INPUT_SIZE + f] = 
                    (int8_t)(buffers->conv3_out[snn_idx] >> FIXED_SCALE);
                snn_idx++;
            }
        }
    }
    
    // 7. LSTM temporal processing
    for (int t = 0; t < TIME_STEPS; t++) {
        lstm_layer_cwru(&buffers->snn_input[t * SNN_INPUT_SIZE],
                       buffers->lstm_state,
                       lstm_weight_ih, lstm_weight_hh,
                       lstm_bias_ih, lstm_bias_hh,
                       SNN_INPUT_SIZE, LSTM_HIDDEN,
                       buffers->precision_mode);
        
        // Store spike train
        for (int i = 0; i < LSTM_HIDDEN; i++) {
            buffers->spike_train[t * LSTM_HIDDEN + i] = buffers->lstm_state[i];
        }
    }
    
    // 8. Extraire features spatiales (moyenne)
    int8_t spatial_features[CONV3_FILTERS] = {0};
    for (int f = 0; f < CONV3_FILTERS; f++) {
        int32_t sum = 0;
        for (int i = 0; i < CONV3_OUT_SIZE; i++) {
            sum += buffers->conv3_out[f * CONV3_OUT_SIZE + i];
        }
        spatial_features[f] = saturate_int32_to_int8((sum / CONV3_OUT_SIZE) >> FIXED_SCALE);
    }
    
    // 9. Fusion et classification
    // Combiner features LSTM et CNN
    int8_t combined_features[LSTM_HIDDEN + CONV3_FILTERS];
    for (int i = 0; i < LSTM_HIDDEN; i++) {
        combined_features[i] = buffers->lstm_state[i];
    }
    for (int i = 0; i < CONV3_FILTERS; i++) {
        combined_features[LSTM_HIDDEN + i] = spatial_features[i];
    }
    
    // FC1 layer
    fc_layer_cwru(combined_features, fc1_weight, fc1_bias,
                 buffers->fc1_out, LSTM_HIDDEN + CONV3_FILTERS, FC1_SIZE,
                 fc1_weight_scale, fc1_bias_scale, buffers->precision_mode);
    
    // FC2 layer
    fc_layer_cwru((int8_t*)buffers->fc1_out, fc2_weight, fc2_bias,
                 buffers->fc2_out, FC1_SIZE, FC1_SIZE / 2,
                 fc2_weight_scale, fc2_bias_scale, buffers->precision_mode);
    
    // FC3 (output layer)
    fc_layer_cwru((int8_t*)buffers->fc2_out, fc3_weight, fc3_bias,
                 buffers->output, FC1_SIZE / 2, OUTPUT_SIZE,
                 fc3_weight_scale, fc3_bias_scale, buffers->precision_mode);
    
    // Incrémenter le compteur
    buffers->inference_count++;
    
    #ifdef ENABLE_BENCHMARKING
    cycle_t end_cycles = get_cycle_count();
    uint32_t cycles = (uint32_t)(end_cycles - start_cycles);
    buffers->total_cycles = cycles;
    #endif
}

// Détecter le type de défaut
uint8_t detect_fault(const int8_t* vibration_signal, uint32_t length) {
    if (length < INPUT_SIZE) return 0xFF; // Erreur
    
    VibrationModelBuffers buffers;
    model_init(&buffers);
    
    // Utiliser les premiers INPUT_SIZE échantillons
    model_predict(&buffers, vibration_signal);
    
    // Trouver la classe avec le score maximum
    int32_t max_score = buffers.output[0];
    uint8_t fault_class = 0;
    
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (buffers.output[i] > max_score) {
            max_score = buffers.output[i];
            fault_class = i;
        }
    }
    
    return fault_class;
}
