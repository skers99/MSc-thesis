from randman.randman import *
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from randman_utils import make_spiking_dataset
class RandmanDataset(Dataset):
    """Characterizes a PyTorch dataset for use with the PyTorch dataloader."""
    def __init__(self, data, labels):
        """Simple initialization of the given dataset."""
        self.data = data
        self.labels = labels

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.labels)

    def __getitem__(self, index):
        """Retrives a single sample from the given dataset."""
        # Load data and get label
        X = self.data[index]
        y = self.labels[index]

        return X, y

def convert_spike_times_to_raster(spike_times: np.ndarray, timestep: float = 1.0, max_time: Optional[float] = None, num_neurons: Optional[int] = None, dtype=None):
    """
    Convert spike times array to spike raster array. 
    For now, all neurons need to have same number of spike times.
    
    Args:
        spike_times: MoreArrays, spiketimes as array of shape (batch_dim x spikes/neuron X 2)
            with final dim: (times, neuron_id)
    """

    if dtype is None:
        dtype = np.int16
    # spike_times = spike_times.astype(np.uint16)
    if num_neurons is None:
        num_neurons = int(np.nanmax(spike_times[:,:,1]))+1
    if max_time is None:
        max_time = np.nanmax(spike_times[:,:,0])
    num_bins = int(max_time / timestep + 1)

    spike_raster = np.zeros((spike_times.shape[0], num_bins, num_neurons), dtype=np.float32)
    batch_id = np.arange(spike_times.shape[0]).repeat(spike_times.shape[1])
    spike_times_flat = (spike_times[:, :, 0].flatten() / timestep).astype(dtype)
    neuron_ids = spike_times[:, :, 1].flatten().astype(dtype)
    np.add.at(spike_raster, (batch_id, spike_times_flat, neuron_ids), 1)
    return spike_raster


def make_spike_raster_dataset(nb_classes=10, nb_units=100, nb_steps=100, dim_manifold=2, nb_samples=1000, alpha=2.0, shuffle=True, seed=None):
# def make_spike_raster_dataset(nb_classes, nb_units, nb_steps, dim_manifold=2, nb_spikes=1, nb_samples=1000, alpha=2.0, shuffle=True):
# def make_spike_raster_dataset():
    spike_times,labels = make_spiking_dataset(nb_classes=nb_classes, nb_units=nb_units, nb_steps=nb_steps, dim_manifold=dim_manifold, seed=seed,nb_samples=nb_samples,shuffle=shuffle,alpha=alpha)
    spike_raster = convert_spike_times_to_raster(spike_times)
    return spike_raster, labels

def get_data_loaders(nb_classes, nb_units, nb_steps, nb_samples, batchsize):
    data, labels = make_spike_raster_dataset(nb_classes=nb_classes, nb_units=nb_units, nb_steps=nb_steps, nb_samples=nb_samples)

    NUM_SAMPLES_TOTAL = (nb_classes*nb_samples)
    NUM_SAMPLES_TRAIN = int(NUM_SAMPLES_TOTAL*0.8)

    data_train, labels_train = data[:NUM_SAMPLES_TRAIN], labels[:NUM_SAMPLES_TRAIN]
    data_test,  labels_test  = data[NUM_SAMPLES_TRAIN:], labels[NUM_SAMPLES_TRAIN:]

    dataset_train = RandmanDataset(data_train, labels_train)
    dataset_test = RandmanDataset(data_test, labels_test)

    dataloader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=True, drop_last=True)
    return dataloader_train, dataloader_test