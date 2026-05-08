import sys

from fast_audiomentations import Clip
from fast_audiomentations.functions.audio_io import load, save


def main(input_path: str, output_path: str, batch_size_repeat: int) -> None:
    # Load the audio file
    samples, sr = load(input_path)

    # Convert samples to tensor, reshape, and repeat
    samples = samples.repeat(batch_size_repeat, 1).contiguous()

    # Initialize the Clip augmentation
    gain = Clip(min=-0.3, max=0.3, p=1.0)

    # Apply the clip augmentation
    augmented_samples = gain(samples=samples, sample_rate=sr)

    # Save all augmented samples to separate files
    for i in range(len(augmented_samples)):
        save(augmented_samples[i], sr, f"{output_path}_{i}.wav")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python -m examples.clip <input_path> <output_path> <batch_size_repeat>"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    batch_size_repeat = int(sys.argv[3])

    main(input_path, output_path, batch_size_repeat)
