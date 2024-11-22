import os
import torch
import torchaudio
from torch.utils.data import Dataset, random_split
from pathlib import Path
from torch.utils.data import DataLoader



class AudioDataset(Dataset):
    def __init__(self, src_dir, dst_dir, transform=None):
        """
        Class to extract source audio files and transform them for training

        Args:
            src_dir (str): Source directory containing audio files
            dst_dir (str): Directory to save the processed audio files.
            transform: Transform to apply to audio data.
        """
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []

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

                    # Extract class label from the filename
                    # Class label is the part before the first underscore (e.g., "0_01_0.wav")
                    label = file_name.split('_')[0]
                    self.labels.append(label)

                    # Make the destination directory match the source directory structure
                    dst_folder_dir = os.path.join(dst_dir, folder)
                    os.makedirs(dst_folder_dir, exist_ok=True)

        # Create a mapping of labels to numeric values
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        label_index = self.label_to_index[label]

        # Path to the transformed file
        folder_path = Path(file_path).parts[-2]
        file_name = Path(file_path).name
        transformed_path = os.path.join(self.dst_dir, folder_path, file_name)

        # Debugging print statements
        # print(f"Original File Path: {file_path}")
        # print(f"Transformed Path: {transformed_path}")

        # Ensure directory exists for saving
        os.makedirs(os.path.dirname(transformed_path), exist_ok=True)

        if os.path.exists(transformed_path):
            # Load the transformed audio if it exists
            # print(f"Loading saved file: {transformed_path}")
            waveform, sample_rate = torchaudio.load(transformed_path)
        else:
            # Process the raw audio and save the transformed file
            # print(f"Processing and saving file: {transformed_path}")
            waveform, sample_rate = torchaudio.load(file_path)

            # Change all files to 8000 Hz sample rate
            if sample_rate != 8000:
                resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
                waveform = resample_transform(waveform)

            if self.transform:
                waveform = self.transform(waveform)

            # Save the transformed waveform
            torchaudio.save(transformed_path, waveform[0], sample_rate)

        return waveform, label_index


def process_load_data(src_dir, dst_dir):
    """
    Processes audio files located in the source directory and saves them in the destination directory.
    Loads the dataset if the audio files have already been processed

    Args:
    src_dir (str): Source directory containing audio files
    dst_dir (str): Directory to save the processed audio files.
    """

    # src_dir = os.getcwd()+'\AudioMNIST\AudioMNIST\data'
    # dst_dir = os.getcwd()+'\ProcessedData'

    transform = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_mels=64),
        torchaudio.transforms.AmplitudeToDB()
    )

    dataset = AudioDataset(src_dir=src_dir, dst_dir=dst_dir, transform=transform)

    # Check label mapping
    print(f'Classes: {dataset.label_to_index.keys()}')

    return dataset, dataset.labels


def gen_loader(dataset, train_split=0.8, val_split=0.1, test_split=0.1):
    """
    Returns the training, validation and test data loaders for input into the neural networks
    Args:
    dataset (AudioDataset): processed audio dataset
    train_split: Percentage of dataset to use for training (e.g. 0.8)
    val_split: Percentage of dataset to use for validation (e.g. 0.1)
    test_split: Percentage of dataset to use for testing (e.g. 0.1)
    """
    # Split data into training, validation and test set
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = int(test_split * len(dataset))

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Create dataloaders for training, val and test set
    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Check dataset size in each dataloader
    print(f'# of batches in training dataloader: {len(train_dataloader)}')
    print(f'# of batches in val dataloader: {len(val_dataloader)}')
    print(f'# of batches in test dataloader: {len(test_dataloader)}')

    return train_dataloader, val_dataloader, test_dataloader


# Create collate function for the DataLoader
def collate_fn(batch):
    # Find the maximum length in the batch
    max_len = max([item[0].size(-1) for item in batch])
    
    # For testing ConvNet
    max_len = 40

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



