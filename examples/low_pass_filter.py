import sys
from fast_audiomentations.functions.audio_io import load, save
from fast_audiomentations import LowPassFilter


def main(input_path, output_path, batch_size_repeat):
    # Load the audio file
    samples, sr = load(input_path)

    # Repeat the samples 'batch_size_repeat' times along the first dimension
    samples = samples.repeat(batch_size_repeat, 1).contiguous()

    # Initialize the LowPassFilter with specified cutoff frequencies and probability
    low_pass_filter = LowPassFilter(min_cutoff_freq=100, max_cutoff_freq=10000, p=1.0, num_taps=101)

    # Apply the low-pass filter to the samples
    augmented_samples = low_pass_filter(samples=samples, sample_rate=sr)

    # Save all augmented samples to separate files
    for i in range(len(augmented_samples)):
        save(augmented_samples[i], sr, f"{output_path}_{i}.wav")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python -m examples.low_pass_filter <input_path> <output_path> <batch_size_repeat>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    batch_size_repeat = int(sys.argv[3])

    main(input_path, output_path, batch_size_repeat)
