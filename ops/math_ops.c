#include <stdint.h>
#include "firmware/model/model_weights.h"

// Fonction utilitaire de saturation
static inline int8_t saturate_int32_to_int8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

// Fonctions RV32X-SQ custom pour CWRU
int32_t custom1_mac(int8_t a, int8_t b, int32_t acc) {
    // Simulated 4-bit MAC operation
    int16_t product = (int16_t)a * (int16_t)b;
    return acc + product;
}

int8_t custom2_lif(int8_t input, int8_t membrane_potential, int8_t threshold) {
    // LIF neuron update for vibration analysis
    int16_t new_potential = membrane_potential + input;
    
    // Leak
    new_potential = (new_potential * 90) / 100;  // 10% decay
    
    // Spike if threshold exceeded
    if (new_potential >= threshold) {
        // Generate spike
        new_potential = 0;  // Reset after spike
        return 127;         // Spike value
    }
    
    // Clip to int8 range using saturation
    return saturate_int32_to_int8(new_potential);
}

int8_t custom3_fusion(int8_t snn_out, int8_t cnn_out, int8_t attention_weight) {
    // Attention-based fusion for CWRU
    int16_t snn_weighted = snn_out * attention_weight;
    int16_t cnn_weighted = cnn_out * (127 - attention_weight);
    
    int16_t fused = (snn_weighted + cnn_weighted) / 127;
    
    // Clip and return using saturation
    return saturate_int32_to_int8(fused);
}

// Convolution 1D pour CWRU (3 couches)
void quantized_conv1d_rv32x(const int8_t *input, const int8_t *kernel,
                           int32_t kernel_scale, int32_t *output,
                           int in_channels, int out_channels,
                           int length, int kernel_size, int stride,
                           uint8_t precision_mode) {
    const int out_length = length / stride;
    
    for (int oc = 0; oc < out_channels; oc++) {
        for (int i = 0; i < out_length; i++) {
            int32_t sum = 0;
            const int in_pos = i * stride;
            const int8_t *k_ptr = &kernel[oc * in_channels * kernel_size];
            
            for (int ic = 0; ic < in_channels; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int input_idx = (ic * length) + in_pos + k;
                    int kernel_idx = (ic * kernel_size) + k;
                    
                    // Use custom1_mac for 4-bit MAC when in medium/low precision mode
                    if (precision_mode >= PRECISION_MEDIUM) {
                        sum = custom1_mac(input[input_idx], k_ptr[kernel_idx], sum);
                    } else {
                        sum += input[input_idx] * k_ptr[kernel_idx];
                    }
                }
            }
            
            // Apply scaling based on precision mode
            int32_t scale_factor = kernel_scale;
            if (precision_mode == PRECISION_LOW) {
                scale_factor = scale_factor >> 1;
            }
            
            output[oc * out_length + i] = ((int64_t)sum * scale_factor) >> FIXED_SCALE;
        }
    }
}

// LSTM pour CWRU
void quantized_lstm_layer_rv32x(const int8_t *input, int8_t *hidden_state,
                               const int8_t *w_ih, const int8_t *w_hh, const int8_t *bias,
                               int32_t w_ih_scale, int32_t w_hh_scale, int32_t bias_scale,
                               int8_t *output, int seq_len, int hidden_size,
                               uint8_t precision_mode) {
    (void)w_hh_scale; // Not used in this implementation
    
    for (int t = 0; t < seq_len; t++) {
        const int8_t *current_input = &input[t * hidden_size];
        
        for (int i = 0; i < hidden_size; i++) {
            int32_t sum = 0;
            
            for (int j = 0; j < hidden_size; j++) {
                if (precision_mode >= PRECISION_MEDIUM) {
                    sum = custom1_mac(current_input[j], w_ih[i * hidden_size + j], sum);
                    sum = custom1_mac(hidden_state[j], w_hh[i * hidden_size + j], sum);
                } else {
                    sum += current_input[j] * w_ih[i * hidden_size + j];
                    sum += hidden_state[j] * w_hh[i * hidden_size + j];
                }
            }
            
            sum = ((int64_t)sum * w_ih_scale) >> FIXED_SCALE;
            sum += (bias[i] * bias_scale) >> FIXED_SCALE;
            
            // Update hidden state with LIF dynamics
            if (precision_mode >= PRECISION_MEDIUM) {
                hidden_state[i] = custom2_lif(saturate_int32_to_int8(sum >> FIXED_SCALE), 
                                             hidden_state[i], 50);
            } else {
                hidden_state[i] = saturate_int32_to_int8(sum >> FIXED_SCALE);
            }
        }
    }
    
    // Copy final state to output
    for (int i = 0; i < hidden_size; i++) {
        output[i] = hidden_state[i];
    }
}

// Fusion pour CWRU
void dynamic_fusion_rv32x(const int8_t *snn_features, const int8_t *qnn_features,
                         uint8_t attention_weight, int32_t *fused_output,
                         int fusion_size, uint8_t precision_mode) {
    for (int i = 0; i < fusion_size; i++) {
        int8_t snn_val = snn_features[i];
        int8_t qnn_val = qnn_features[i];
        
        if (precision_mode >= PRECISION_MEDIUM) {
            fused_output[i] = custom3_fusion(snn_val, qnn_val, attention_weight);
        } else {
            int16_t snn_contrib = (snn_val * attention_weight);
            int16_t qnn_contrib = (qnn_val * (127 - attention_weight));
            fused_output[i] = saturate_int32_to_int8((snn_contrib + qnn_contrib) >> 7);
        }
    }
}

// Fonctions utilitaires pour CWRU
int8_t clamp_int8(int32_t x) {
    return saturate_int32_to_int8(x);
}

// Activation functions optimized for CWRU
int32_t relu_int32(int32_t x) {
    return (x > 0) ? x : 0;
}

int8_t relu_int8(int8_t x) {
    return (x > 0) ? x : 0;
}
