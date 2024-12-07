import os
import torch
import torchaudio
from torch.utils.data import Dataset, random_split
from pathlib import Path
from torch.utils.data import DataLoader
import csv



class AudioDataset(Dataset):
    def __init__(self, src_dir, dst_dir, transform=None, input_type="wav"):
        """
        Class to extract source audio files and transform them for training

        Args:
            src_dir (str): Source directory containing audio files
            dst_dir (str): Directory to save the processed audio files.
            transform: Transform to apply to audio data.
            input_type: Spectrogram or repeated 1 second long wav file
        """
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.transform = transform
        self.input_type = input_type
        self.file_paths = []
        self.labels = []
        self.data = []

        # Extract audio file paths and labels from the source directory
        for folder in os.listdir(src_dir):
            subject_path = os.path.join(src_dir, folder)
            if not os.path.isdir(subject_path):
                continue

            for file_name in os.listdir(subject_path):
                if file_name.endswith('.wav'):
                    # Full path to the audio file
                    file_path = os.path.join(subject_path, file_name)
                    self.file_paths.append(file_path)

                    # Extract class label from file name
                    # Class label is the part before the first underscore (e.g., "0_01_0.wav")
                    label = int(file_name.split('_')[0])
                    # print(f'Labels type: {type(label)}')
                    self.labels.append(label)

                    # Make the destination directory match the source directory structure
                    dst_folder_dir = os.path.join(dst_dir, folder)
                    os.makedirs(dst_folder_dir, exist_ok=True)

                    # Path to the transformed file
                    transformed_path = os.path.join(dst_folder_dir, file_name)

                    # Load audio
                    if os.path.exists(transformed_path):
                        # Load the transformed audio if it exists
                        # print(f"Loading saved file: {transformed_path}")
                        waveform, sample_rate = torchaudio.load(transformed_path)
                        self.data.append(waveform)
                    else:
                        # Process the raw audio and save the transformed file
                        # print(f"Processing and saving file: {transformed_path}")
                        waveform, sample_rate = torchaudio.load(file_path)

                        # Change all files to 8000 Hz sample rate
                        if sample_rate != 8000:
                            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
                            waveform = resample_transform(waveform)

                        if self.input_type == "wav":
                            # Repeat waveform to ensure 5-second duration
                            duration = waveform.shape[1] / 8000  # Current duration in seconds
                            if duration < 5.0:
                                repeat_count = int(5.0 / duration) + 1
                                waveform = waveform.repeat(1, repeat_count)[:, :40000]

                        if self.transform:
                            waveform = self.transform(waveform)

                        # Save the transformed waveform
                        torchaudio.save(transformed_path, waveform, sample_rate)
                        self.data.append(waveform)

        # Create a mapping of labels to numeric values
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def process_load_data(src_dir, dst_dir, input_type="spectro"):
    """
    Processes audio files located in the source directory and saves them in the destination directory.
    Loads the dataset if the audio files have already been processed

    Args:
    src_dir (str): Source directory containing audio files
    dst_dir (str): Directory to save the processed audio files.
    input_type (str): If input_type = spectro, spectrograms will be loaded. Otherwise, the wav files will be loaded.
    """

    # src_dir = os.getcwd()+'\AudioMNIST\AudioMNIST\data'
    # dst_dir = os.getcwd()+'\ProcessedData'

    # If spectrograms are desired to be input into the neural network
    if input_type == "spectro":
        transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_mels=64),
            torchaudio.transforms.AmplitudeToDB()
        )

    else:
        transform = None

    dataset = AudioDataset(src_dir=src_dir, dst_dir=dst_dir, transform=transform, input_type=input_type)

    # Check label mapping
    print(f'Classes: {dataset.label_to_index.keys()}')

    return dataset, dataset.labels


def gen_loader(dataset, batch_size=16, train_split=0.8, val_split=0.1, test_split=0.1, length=None):
    """
    Returns the training, validation and test data loaders for input into the neural networks
    Args:
    dataset (AudioDataset): processed audio dataset
    train_split: Percentage of dataset to use for training (e.g. 0.8)
    val_split: Percentage of dataset to use for validation (e.g. 0.1)
    test_split: Percentage of dataset to use for testing (e.g. 0.1)
    max_len: Set a predefined length for the waveforms
    """
    # Split data into training, validation and test set
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = int(test_split * len(dataset))

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    collate_fn = lambda batch : collate(batch, length)

    # Create dataloaders for training, val and test set
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Check dataset size in each dataloader
    print(f'# of batches in training dataloader: {len(train_dataloader)}')
    print(f'# of batches in val dataloader: {len(val_dataloader)}')
    print(f'# of batches in test dataloader: {len(test_dataloader)}')

    return train_dataloader, val_dataloader, test_dataloader


# Create collate function for the DataLoader
def collate(batch, length = None):
    # Find the maximum length in the batch
    max_len = length or max([item[0].size(-1) for item in batch])

    # Pad each waveform in the batch to the max length
    padded_waveforms = []
    labels = []
    for waveform, label in batch:
        #print(waveform.size(-1))
        padding = max_len - waveform.size(-1)
        # Pad the waveform with zeros at the end
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding)).squeeze()
        padded_waveforms.append(padded_waveform)
        labels.append(label)

    # Stack waveforms to make a batch
    waveforms_batch = torch.stack(padded_waveforms)
    return waveforms_batch, torch.tensor(labels)



