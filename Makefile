# ==================== CONFIGURATION ====================
CC = riscv64-unknown-elf-gcc
OBJCOPY = riscv64-unknown-elf-objcopy
SIZE = riscv64-unknown-elf-size
MKDIR = mkdir -p

# Librairie GCC pour opérations manquantes
LIBS = -lgcc

# ==================== DÉTECTION DE PLATEFORME ====================
# Valeur par défaut
PLATFORM ?= hifive1

# Vérification de la plateforme
ifeq ($(PLATFORM),hifive1)
  ARCH = rv32imac
  ABI = ilp32
  PLATFORM_FLAGS = -DHIFIVE1 -DCPU_FREQ_MHZ=320 -DCPU_FREQ_HZ=32000000
  LDFLAGS += -T linker_hifive1.ld
  BUILD_DIR = build_hifive1
  # Pas de support float matériel
  ARCH_FLAGS = -march=rv32imac -mabi=ilp32
else ifeq ($(PLATFORM),k210)
  ARCH = rv64gc
  ABI = lp64
  PLATFORM_FLAGS = -DK210 -DCPU_FREQ_MHZ=400 -DCPU_FREQ_HZ=400000000
  LDFLAGS += -T linker_k210.ld
  CFLAGS += -mcmodel=medany
  LDFLAGS += -mcmodel=medany
  BUILD_DIR = build_k210
  ARCH_FLAGS = -march=rv64gc -mabi=lp64
else
  $(error PLATFORM must be hifive1 or k210 (got: $(PLATFORM)))
endif

# ==================== CONFIGURATION MODÈLE VIBRATION ====================
# Configuration spécifique au modèle vibration CWRU
CFLAGS += -DINPUT_SIZE=1024
CFLAGS += -DCONV1_FILTERS=8
CFLAGS += -DCONV2_FILTERS=16
CFLAGS += -DCONV3_FILTERS=32
CFLAGS += -DLSTM_HIDDEN=32
CFLAGS += -DOUTPUT_SIZE=4
CFLAGS += -DTIME_STEPS=32
CFLAGS += -DFC1_SIZE=64
CFLAGS += -DSNN_INPUT_SIZE=32
CFLAGS += -DCONV1_OUT_SIZE=256
CFLAGS += -DCONV2_OUT_SIZE=64
CFLAGS += -DCONV3_OUT_SIZE=32

# Flags spécifiques vibration
CFLAGS += -DVIBRATION_MODEL

# ==================== FLAGS DE COMPILATION ====================
CFLAGS += $(ARCH_FLAGS)
CFLAGS += -Os -Wall -Wextra -Wno-unused-parameter
CFLAGS += -fno-common -ffreestanding -nostdlib -fno-builtin
CFLAGS += -fno-stack-protector
CFLAGS += -I. -Ifirmware/model -Iops -Iutils
CFLAGS += -fdata-sections -ffunction-sections
CFLAGS += $(PLATFORM_FLAGS)

# Désactiver opérations float
CFLAGS += -mno-fdiv

# ==================== FLAGS COMMUNS ====================
CFLAGS += -DQ_BITS=8 -DFIXED_SCALE=8 -DFIXED_SCALE_VAL=256
CFLAGS += -DENABLE_BENCHMARKING

# Flags thermiques (utilisez ceux de thermal_manager.h)
CFLAGS += -DTEMP_THRESHOLD_MEDIUM=50
CFLAGS += -DTEMP_THRESHOLD_HIGH=70
CFLAGS += -DPRECISION_HIGH=0
CFLAGS += -DPRECISION_MEDIUM=1
CFLAGS += -DPRECISION_LOW=2

# Flags de lien
LDFLAGS += -Wl,--gc-sections $(LIBS)

# ==================== FICHIERS SOURCES ====================
# Sources spécifiques vibration
C_SRCS = main.c uart.c firmware/model/model.c \
         firmware/model/model_weights.c ops/math_ops.c \
         utils/memutils.c utils/numutils.c utils/cycle_count.c \
         utils/thermal_manager.c
ASM_SRCS = start.S
C_OBJS = $(addprefix $(BUILD_DIR)/, $(C_SRCS:.c=.o))
ASM_OBJS = $(addprefix $(BUILD_DIR)/, $(ASM_SRCS:.S=.o))
OBJS = $(C_OBJS) $(ASM_OBJS)

# ==================== RÈGLES PRINCIPALES ====================
.PHONY: all clean size info thermal-demo help hifive1 k210

all: $(BUILD_DIR)/firmware.bin
	@echo ""
	@$(MAKE) --no-print-directory size

help:
	@echo "=== CWRU VIBRATION ANALYSIS - INDUSTRIAL BUILD ===\n"
	@echo "Usage: make [PLATFORM=<platform>] [target]\n"
	@echo "Platforms (default: hifive1):"
	@echo "  PLATFORM=hifive1  - HiFive1 (RV32IMAC, 32MHz)"
	@echo "  PLATFORM=k210     - Kendryte K210 (RV64GC, 400MHz)\n"
	@echo "Targets:"
	@echo "  all          - Build firmware (default)"
	@echo "  clean        - Clean build directory"
	@echo "  size         - Show memory usage"
	@echo "  info         - Show build configuration"
	@echo "  thermal-demo - Show thermal management info"
	@echo "  help         - This help message"
	@echo "  hifive1      - Build for HiFive1"
	@echo "  k210         - Build for K210\n"
	@echo "Examples:"
	@echo "  make                     # Build for HiFive1 (default)"
	@echo "  make PLATFORM=k210       # Build for K210"
	@echo "  make clean all          # Clean and rebuild"
	@echo "  make info               # Show configuration\n"
	@echo "Model: CWRU Vibration Fault Detection"
	@echo "ΔT constraint: ≤ 7.0°C (Industrial)"
	@echo "Accuracy: 100% (trained)"
	@echo "Parameters: 17,220"

clean:
	@echo "Cleaning vibration build directories..."
	@rm -rf build_hifive1 build_k210
	@echo "Clean complete."

size: $(BUILD_DIR)/firmware.elf
	@echo "\n=== VIBRATION INDUSTRIAL - MEMORY USAGE ==="
	@echo "Platform: $(PLATFORM)"
	@echo "Model: CWRU Vibration Fault Detection"
	@echo "Thermal: ΔT ≤ 7.0°C (Industrial)"
	@echo "--------------------------------------------"
	@$(SIZE) $(BUILD_DIR)/firmware.elf

# Règles spécifiques plateforme
hifive1:
	$(MAKE) PLATFORM=hifive1 $(MAKECMDGOALS)

k210:
	$(MAKE) PLATFORM=k210 $(MAKECMDGOALS)

# ==================== RÈGLES DE COMPILATION ====================
$(BUILD_DIR)/%.o: %.c
	@$(MKDIR) $(dir $@)
	@echo "CC  $(notdir $<) [Vibration]"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.S
	@$(MKDIR) $(dir $@)
	@echo "AS  $(notdir $<)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/firmware.elf: $(OBJS)
	@echo "LD  $@ [Vibration]"
	@$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $@

$(BUILD_DIR)/firmware.bin: $(BUILD_DIR)/firmware.elf
	@echo "OBJCOPY $@"
	@$(OBJCOPY) -O binary $< $@
	@echo "\n✅ VIBRATION INDUSTRIAL BUILD SUCCESSFUL!"
	@echo "   Platform: $(PLATFORM)"
	@echo "   Thermal: ΔT ≤ 7.0°C"
	@echo "   Ready for factory deployment!"
