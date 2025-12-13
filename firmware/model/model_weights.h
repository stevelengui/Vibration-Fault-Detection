#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

// ==================== MODEL CONFIGURATION ====================
#define INPUT_SIZE 1024
#define N_FEATURES 1
#define CONV1_FILTERS 8
#define CONV2_FILTERS 16
#define CONV3_FILTERS 32
#define LSTM_HIDDEN 32
#define OUTPUT_SIZE 4
#define TIME_STEPS 32
#define FC1_SIZE 64

// Calculated sizes
#define CONV1_OUT_SIZE 256
#define CONV2_OUT_SIZE 64
#define CONV3_OUT_SIZE 32
#define SNN_INPUT_SIZE 32
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
#define Q_BITS 8
#define FIXED_SCALE 8
#define FIXED_SCALE_VAL 256

// ==================== BUFFER STRUCTURE ====================
typedef struct {
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
} VibrationModelBuffers;

// ==================== WEIGHT DECLARATIONS ====================
extern const int8_t conv1_weight[56];
extern const int32_t conv1_weight_scale;

extern const int8_t conv1_bias[8];
extern const int32_t conv1_bias_scale;

extern const int8_t conv2_weight[640];
extern const int32_t conv2_weight_scale;

extern const int8_t conv2_bias[16];
extern const int32_t conv2_bias_scale;

extern const int8_t conv3_weight[1536];
extern const int32_t conv3_weight_scale;

extern const int8_t conv3_bias[32];
extern const int32_t conv3_bias_scale;

extern const int8_t lstm_weight_ih[4096];
extern const int32_t lstm_weight_ih_scale;

extern const int8_t lstm_weight_hh[4096];
extern const int32_t lstm_weight_hh_scale;

extern const int8_t lstm_bias_ih[128];
extern const int32_t lstm_bias_ih_scale;

extern const int8_t lstm_bias_hh[128];
extern const int32_t lstm_bias_hh_scale;

extern const int8_t fc1_weight[4096];
extern const int32_t fc1_weight_scale;

extern const int8_t fc1_bias[64];
extern const int32_t fc1_bias_scale;

extern const int8_t fc2_weight[2048];
extern const int32_t fc2_weight_scale;

extern const int8_t fc2_bias[32];
extern const int32_t fc2_bias_scale;

extern const int8_t fc3_weight[128];
extern const int32_t fc3_weight_scale;

extern const int8_t fc3_bias[4];
extern const int32_t fc3_bias_scale;

// ==================== FUNCTION PROTOTYPES ====================
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
