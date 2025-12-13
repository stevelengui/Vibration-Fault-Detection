#include "cycle_count.h"
#include <stdint.h>

// Variable globale pour la fréquence CPU
static uint32_t cpu_freq_hz = 32000000; // 32 MHz par défaut

void cycle_count_init(uint32_t frequency_hz) {
    cpu_freq_hz = frequency_hz;
}

cycle_t get_cycle_count(void) {
    cycle_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

uint32_t cycles_to_us(cycle_t cycles) {
    // Éviter __udivdi3 en utilisant des divisions 32-bit
    uint64_t cycles_64 = cycles;
    
    // Calculer cycles / (freq_hz / 1000000) = cycles * 1000000 / freq_hz
    // Mais faire d'abord cycles / 1000 pour éviter overflow
    uint32_t cycles_k = (uint32_t)(cycles_64 / 1000);
    uint32_t freq_khz = cpu_freq_hz / 1000;
    
    if (freq_khz == 0) freq_khz = 1;
    
    // cycles_k / freq_khz donne des millisecondes
    uint32_t ms = cycles_k / freq_khz;
    
    // Convertir en microsecondes
    return ms * 1000;
}

// Fonction supplémentaire pour éviter les erreurs de lien
uint32_t cycles_to_us_simple(uint32_t cycles) {
    // Version simplifiée pour les petits nombres - éviter division 64-bit
    #if defined(HIFIVE1) && CPU_FREQ_HZ == 32000000
        // HiFive1: 32 MHz, donc 1 us = 32 cycles
        return cycles / 32;
    #elif defined(K210) && CPU_FREQ_HZ == 400000000
        // K210: 400 MHz, donc 1 us = 400 cycles
        return cycles / 400;
    #else
        // Fallback: division par 1000 (approximation)
        return cycles / 1000;
    #endif
}
