import sys
from fast_audiomentations.functions.audio_io import load, save
from fast_audiomentations import Gain
import torch


def main(input_path, output_path, batch_size_repeat):
    # Load the audio file
    samples, sr = load(input_path)

    # Convert samples to tensor, reshape, and repeat
    samples = samples.repeat(batch_size_repeat, 1).contiguous()

    # Initialize the Gain augmentation
    gain = Gain(min_gain_in_db=-12, max_gain_in_db=12, p=1.0, dtype=torch.float32)

    # Apply the gain augmentation
    augmented_samples = gain(samples=samples, sample_rate=sr)

    # Save all augmented samples to separate files
    for i in range(len(augmented_samples)):
        save(augmented_samples[i], sr, f"{output_path}_{i}.wav")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python -m examples.gain <input_path> <output_path> <batch_size_repeat>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    batch_size_repeat = int(sys.argv[3])

    main(input_path, output_path, batch_size_repeat)
