#include "model_weights.h"

// ==================== FAULT NAMES ====================
const char* get_fault_name(uint8_t fault_class) {
    switch(fault_class) {
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
    }
}

// ==================== WEIGHT DEFINITIONS ====================
const int8_t conv1_weight[] = {
    #include "arrays/conv1_weight_array.txt"
};
const int32_t conv1_weight_scale = 1;

const int8_t conv1_bias[] = {
    #include "arrays/conv1_bias_array.txt"
};
const int32_t conv1_bias_scale = 1;

const int8_t conv2_weight[] = {
    #include "arrays/conv2_weight_array.txt"
};
const int32_t conv2_weight_scale = 1;

const int8_t conv2_bias[] = {
    #include "arrays/conv2_bias_array.txt"
};
const int32_t conv2_bias_scale = 0;

const int8_t conv3_weight[] = {
    #include "arrays/conv3_weight_array.txt"
};
const int32_t conv3_weight_scale = 0;

const int8_t conv3_bias[] = {
    #include "arrays/conv3_bias_array.txt"
};
const int32_t conv3_bias_scale = 0;

const int8_t lstm_weight_ih[] = {
    #include "arrays/lstm_weight_ih_array.txt"
};
const int32_t lstm_weight_ih_scale = 1;

const int8_t lstm_weight_hh[] = {
    #include "arrays/lstm_weight_hh_array.txt"
};
const int32_t lstm_weight_hh_scale = 1;

const int8_t lstm_bias_ih[] = {
    #include "arrays/lstm_bias_ih_array.txt"
};
const int32_t lstm_bias_ih_scale = 1;

const int8_t lstm_bias_hh[] = {
    #include "arrays/lstm_bias_hh_array.txt"
};
const int32_t lstm_bias_hh_scale = 1;

const int8_t fc1_weight[] = {
    #include "arrays/fc1_weight_array.txt"
};
const int32_t fc1_weight_scale = 0;

const int8_t fc1_bias[] = {
    #include "arrays/fc1_bias_array.txt"
};
const int32_t fc1_bias_scale = 0;

const int8_t fc2_weight[] = {
    #include "arrays/fc2_weight_array.txt"
};
const int32_t fc2_weight_scale = 0;

const int8_t fc2_bias[] = {
    #include "arrays/fc2_bias_array.txt"
};
const int32_t fc2_bias_scale = 0;

const int8_t fc3_weight[] = {
    #include "arrays/fc3_weight_array.txt"
};
const int32_t fc3_weight_scale = 1;

const int8_t fc3_bias[] = {
    #include "arrays/fc3_bias_array.txt"
};
const int32_t fc3_bias_scale = 0;

