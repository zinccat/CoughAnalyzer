#!/bin/bash
# noise_to_yolo_pipeline.sh - Automated pipeline for adding noise to audio and generating YOLO data

# Set working directory to project root
cd "$(dirname "$0")"

#############################################
# CUSTOMIZABLE PARAMETERS
#############################################

# Noise type parameters
PINK="False"              # Add pink noise (True/False)
BROWN="False"            # Add brown noise (True/False)
BEEPING="True"           # Add beeping sounds (True/False)
TALKING="True"          # Add talking sounds (True/False)

# Noise level parameters
WHITE_LEVEL="0.03"       # Level of white noise (0.0-1.0)
PINK_LEVEL="0.01"        # Level of pink noise (0.0-1.0)
BROWN_LEVEL="0.01"       # Level of brown noise (0.0-1.0)

# Noise burst parameters
NOISE_BURST_NUM="10"     # Number of random noise bursts
NOISE_BURST_DURATION="0.5"  # Duration of each noise burst in seconds

# Output directory parameters (optional)
CUSTOM_OUTPUT_DIR=""     # Leave empty to use default naming

# MFCC generation parameters
MAX_DURATION="11.0"      # Maximum audio duration in seconds
N_FFT="2048"             # FFT window size for MFCC calculation

#############################################
# PIPELINE EXECUTION (no need to modify below)
#############################################

echo "=== Step 1: Adding noise to audio files ==="
# Build the command with all parameters
NOISE_CMD="python3 src/data/add_noise.py \
  --pink $PINK \
  --brown $BROWN \
  --white_level $WHITE_LEVEL \
  --pink_level $PINK_LEVEL \
  --brown_level $BROWN_LEVEL \
  --noise_burst_num $NOISE_BURST_NUM \
  --noise_burst_duration $NOISE_BURST_DURATION \
  --beeping $BEEPING \
  --talking $TALKING"

# Add output directory if specified
if [ ! -z "$CUSTOM_OUTPUT_DIR" ]; then
  NOISE_CMD="$NOISE_CMD --output_dir $CUSTOM_OUTPUT_DIR"
fi

# Execute the command
echo "Running: $NOISE_CMD"
eval $NOISE_CMD

# Step 2: Get the generated directory name
NOISE_DIR=$(ls -td data_w_noise_* | head -1)
if [ -z "$NOISE_DIR" ]; then
  echo "Error: No noise directory found. The add_noise.py script may have failed."
  exit 1
fi
echo "=== Step 2: Found noise directory: $NOISE_DIR ==="

# Step 3: Create a modified version of gen_dataset_mfcc.py
echo "=== Step 3: Creating modified MFCC script ==="
cp src/data/gen_dataset_mfcc.py src/data/gen_dataset_mfcc_noisy.py
sed -i '' "s|data_dir = \"../CoughSegmentation/Data\"|data_dir = \"$NOISE_DIR\"|g" src/data/gen_dataset_mfcc_noisy.py

# Also update the max_duration and n_fft parameters
sed -i '' "s|max_duration: float = 11.0|max_duration: float = $MAX_DURATION|g" src/data/gen_dataset_mfcc_noisy.py
sed -i '' "s|n_fft: int = 2048|n_fft: int = $N_FFT|g" src/data/gen_dataset_mfcc_noisy.py

# Step 4: Run the modified script to generate YOLO data
echo "=== Step 4: Generating YOLO data from noisy audio ==="
python3 src/data/gen_dataset_mfcc_noisy.py

# Step 5: Summarize results
echo "=== Pipeline Complete ==="
echo "Noise parameters:"
echo "  - Types: pink=$PINK, brown=$BROWN, beeping=$BEEPING, talking=$TALKING"
echo "  - Levels: white=$WHITE_LEVEL, pink=$PINK_LEVEL, brown=$BROWN_LEVEL"
echo "  - Bursts: count=$NOISE_BURST_NUM, duration=$NOISE_BURST_DURATION"
echo "MFCC parameters:"
echo "  - Max duration: $MAX_DURATION seconds"
echo "  - FFT window size: $N_FFT"
echo "Noisy data directory: $NOISE_DIR"
echo "YOLO dataset directory: data_mfcc_padding_11"
echo "Number of training images: $(ls data_mfcc_padding_11/images/train 2>/dev/null | wc -l)"
echo "Number of test images: $(ls data_mfcc_padding_11/images/test 2>/dev/null | wc -l)"
echo "Number of validation images: $(ls data_mfcc_padding_11/images/val 2>/dev/null | wc -l)"