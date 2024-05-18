import os
import pandas
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.nn import ConstantPad2d
from scipy.stats import skew as Skewness, kurtosis as Kurtosis
from math import floor,ceil


def collate_by_center_padding(maxrows):
    def collate(batch):
        tensors = [d[0] for d in batch]
        targets = [d[1] for d in batch]
        labels = [d[2] for d in batch]
        for i in range(len(tensors)):
            tensors[i] = ConstantPad2d(((0,0,floor((maxrows-tensors[i].shape[0])/2),ceil((maxrows-tensors[i].shape[0])/2))),0.0)(tensors[i])
        tensors = torch.stack(tensors,dim=0)
        targets = torch.stack(targets,dim=0)
        return tensors, targets, labels
    return collate


def collate_by_end_padding(maxrows):
    def collate(batch):
        tensors = [d[0] for d in batch]
        targets = [d[1] for d in batch]
        labels = [d[2] for d in batch]
        for i in range(len(tensors)):
            tensors[i] = ConstantPad2d((0,0,0,maxrows-tensors[i].shape[0]),0.0)(tensors[i])
        tensors = torch.stack(tensors,dim=0)
        targets = torch.stack(targets,dim=0)
        return tensors, targets, labels
    return collate


class AirSignsFeaturesDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        loader_device,
        num_columns,
        num_windows,
        window_size,
        num_descriptors,
        minimum=True,
        maximum=True,
        mean=True,
        median=True,
        skewness=True,
        kurtosis=True,
    ):
        super().__init__()
        self.loader_device = loader_device
        self.num_columns = num_columns
        self.num_windows = num_windows
        self.window_size = window_size
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in os.listdir(os.path.join(dataset_path,foldername)):
                    self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.median = median
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.num_descriptors = sum(int(desc) for desc in [self.minimum,self.maximum,self.mean,self.median,self.skewness,self.kurtosis])
        if self.num_descriptors != num_descriptors:
            raise Exception("Invalid Number of Descriptors!")
        self.list_of_all = []
        self.full_length = len(self.filelist)

        for file_index in range(self.full_length):
            filepath = self.filelist[file_index]
            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            tensor = torch.from_numpy(
                np.genfromtxt(filepath,delimiter=',')
            ).to(self.loader_device).type(torch.float32)

            tensor = tensor[:,:self.num_columns]

            start_seq = [
                round((i * (tensor.shape[0] - self.window_size)) / (self.num_windows - 1))
                for i in range(self.num_windows)
            ]
            out = [
                tensor[start_seq[idx] : start_seq[idx] + self.window_size, :]
                for idx in range(self.num_windows)
            ]
            out.append(tensor)

            out = self.descriptors(out)

            self.list_of_all.append([out, self.target(label), label])

    def descriptors(self, tensor_list):
        descriptors_tensor = torch.zeros([self.num_windows+1,self.num_descriptors,self.num_columns]).to(self.loader_device)
        v = torch.zeros([1])
        for i,tensor in enumerate(tensor_list):
            descriptors_ = torch.zeros([self.num_descriptors])
            for k in range(self.num_columns):
                if tensor[tensor[:,k]!=-1,k].shape[0]>0:
                    j = 0
                    if self.minimum:
                        descriptors_[j],_ = torch.min(tensor[tensor[:,k]!=-1,k],0)
                        j += 1
                    if self.maximum:
                        descriptors_[j],_ = torch.max(tensor[tensor[:,k]!=-1,k],0)
                        j += 1
                    if self.mean:
                        descriptors_[j] = torch.mean(tensor[tensor[:,k]!=-1,k],dim=0)
                        j += 1
                    if self.median:
                        descriptors_[j],_ = torch.median(tensor[tensor[:,k]!=-1,k],dim=0)
                        j += 1
                    if self.skewness:
                        v[0] = Skewness(tensor[tensor[:,k]!=-1,k].cpu(),nan_policy='raise')
                        if not torch.isnan(v[0]): descriptors_[j] = v[0]
                        j += 1
                    if self.kurtosis:
                        v[0] = Kurtosis(tensor[tensor[:,k]!=-1,k].cpu(),nan_policy='raise')
                        if not torch.isnan(v[0]): descriptors_[j] = v[0]
                        j += 1
                    if self.variance:
                        v[0] = torch.var(tensor[tensor[:,k]!=-1,k],dim=0)
                        if not torch.isnan(v[0]): descriptors_[j] = v[0]
                        j += 1
                    if self.std:
                        v[0] = torch.std(tensor[tensor[:,k]!=-1,k],dim=0)
                        if not torch.isnan(v[0]): descriptors_[j] = v[0]
                        j += 1
                    descriptors_tensor[i,:,k] = descriptors_
                else:
                    descriptors_tensor[i,:,k] = descriptors_
        return descriptors_tensor

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def AirSignsFeaturesLoader(
    dataset_path,
    loader_device,
    num_columns,
    num_windows,
    window_size,
    batch_size,
    shuffle=True,
    num_workers=0,
    num_descriptors=5,
):
    dataset = AirSignsFeaturesDataset(dataset_path, loader_device, num_columns, num_windows, window_size, num_descriptors)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True
    )


class AirSignsUnpaddedDataset(Dataset):

    def __init__(self, dataset_path, loader_device, num_columns):
        super().__init__()
        self.loader_device = loader_device
        self.num_columns = num_columns
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in sorted(os.listdir(os.path.join(dataset_path,foldername))):
                self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.list_of_all = []
        self.full_length = len(self.filelist)

        for file_index in range(self.full_length):
            filepath = self.filelist[file_index]
            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            tensor = torch.from_numpy(
                pandas.read_csv(filepath, header=None,sep=",").to_numpy()
            ).to(self.loader_device).type(torch.float32)

            tensor = tensor[:,:self.num_columns]

            self.list_of_all.append([tensor, self.target(label), label])

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def AirSignsPaddedLoader(
    dataset_path,
    loader_device,
    num_columns,
    max_rows,
    batch_size,
    shuffle=True,
    num_workers=0,
    padding='center', # or end padded
):
    dataset = AirSignsUnpaddedDataset(dataset_path, loader_device, num_columns)
    padder = collate_by_center_padding(max_rows) if padding == 'center' else collate_by_end_padding(max_rows)
    padder = collate_by_end_padding(max_rows)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True,collate_fn=padder
    )


class AirSignsInterpolatedDataset(Dataset):
    def __init__(self, dataset_path, loader_device, num_columns, test_mode=False):
        super().__init__()
        self.loader_device = loader_device
        self.num_columns = num_columns
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        
        # Define cutoff for testing
        test_class_count = 10
        if test_mode:
            self.folderlist = self.folderlist[-test_class_count:]  # Last 10 classes for testing
        else:
            self.folderlist = self.folderlist[:-test_class_count]  # All but last 10 classes for training

        self.data_pairs = []

        # Iterate over each folder and construct data pairs with the next class
        num_folders = len(self.folderlist)
        for i, foldername in enumerate(self.folderlist):
            current_folder_path = os.path.join(dataset_path, foldername)
            next_folder_index = (i + 1) % num_folders
            next_folder_name = self.folderlist[next_folder_index]
            next_folder_path = os.path.join(dataset_path, next_folder_name)

            current_files = sorted(os.listdir(current_folder_path))
            next_files = sorted(os.listdir(next_folder_path))

            for current_file, next_file in zip(current_files, next_files):
                current_file_path = os.path.join(current_folder_path, current_file)
                next_file_path = os.path.join(next_folder_path, next_file)

                current_data = torch.from_numpy(
                    pandas.read_csv(current_file_path, header=None, sep=",").to_numpy()
                ).to(self.loader_device).type(torch.float32)[:,:self.num_columns]

                next_data = torch.from_numpy(
                    pandas.read_csv(next_file_path, header=None, sep=",").to_numpy()
                ).to(self.loader_device).type(torch.float32)[:,:self.num_columns]

                self.data_pairs.append((current_data, next_data, i, next_folder_index))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        return self.data_pairs[index]

def AirSignsRawLoader(
    dataset_path,
    loader_device,
    num_columns,
    batch_size,
    shuffle=True,
    num_workers=0,
    test_mode=False
):
    dataset = AirSignsInterpolatedDataset(dataset_path, loader_device, num_columns, test_mode=test_mode)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True
    )


class AirSignsImageSet(Dataset):

    def __init__(
        self,
        dataset_path,
        loader_device,
        image_size,
    ):
        super().__init__()
        self.loader_device = loader_device
        self.image_size = image_size
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in os.listdir(os.path.join(dataset_path,foldername)):
                    self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x.to(self.loader_device)
                ),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1)
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.list_of_all = []
        for index in range(len(self.filelist)):
            filepath = self.filelist[index]

            x = Image.open(filepath).convert('L')
            x = self.transform_image(x).type(torch.float32)

            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            self.list_of_all.append([x, self.target(label), label])

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def AirSignsImageLoader(
    dataset_path,
    loader_device,
    image_size,
    batch_size,
    shuffle=True,
    num_workers=0
):
    dataset = AirSignsImageSet(dataset_path,loader_device,image_size)
    return DataLoader(
         dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True
    )


class SVCFeaturesDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        loader_device,
        num_columns,
        num_windows,
        window_size,
        num_descriptors,
        minimum=True,
        maximum=True,
        mean=True,
        median=True,
        skewness=True,
        kurtosis=True,
    ):
        super().__init__()
        self.loader_device = loader_device
        self.num_columns = num_columns
        self.num_windows = num_windows
        self.window_size = window_size
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in os.listdir(os.path.join(dataset_path,foldername)):
                    self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.median = median
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.num_descriptors = sum(int(desc) for desc in [self.minimum,self.maximum,self.mean,self.median,self.skewness,self.kurtosis])
        if self.num_descriptors != num_descriptors:
            raise Exception("Invalid Number of Descriptors!")
        self.list_of_all = []
        self.full_length = len(self.filelist)

        for file_index in range(self.full_length):
            filepath = self.filelist[file_index]
            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            tensor = torch.from_numpy(
                np.genfromtxt(filepath,delimiter=',')
            ).to(self.loader_device).type(torch.float32)

            tensor = tensor[:,:self.num_columns]

            start_seq = [
                round((i * (tensor.shape[0] - self.window_size)) / (self.num_windows - 1))
                for i in range(self.num_windows)
            ]
            out = [
                tensor[start_seq[idx] : start_seq[idx] + self.window_size, :]
                for idx in range(self.num_windows)
            ]
            out.append(tensor)

            out = self.descriptors(out)

            self.list_of_all.append([out, self.target(label), label])

    def descriptors(self, tensor_list):
        descriptors_tensor = torch.zeros([self.num_windows+1,self.num_descriptors,self.num_columns]).to(self.loader_device)
        v = torch.zeros([1])
        for i,tensor in enumerate(tensor_list):
            descriptors_ = torch.zeros([self.num_descriptors])
            for k in range(self.num_columns):
                j = 0
                if self.maximum:
                    descriptors_[j],_ = torch.max(tensor[:,k],0)
                    j += 1
                if self.mean:
                    descriptors_[j] = torch.mean(tensor[:,k],dim=0)
                    j += 1
                if self.median:
                    descriptors_[j],_ = torch.median(tensor[:,k],dim=0)
                    j += 1
                if self.skewness:
                    v[0] = Skewness(tensor[:,k].cpu(),nan_policy='raise')
                    if not torch.isnan(v[0]): descriptors_[j] = v[0]
                    j += 1
                if self.kurtosis:
                    v[0] = Kurtosis(tensor[:,k].cpu(),nan_policy='raise')
                    if not torch.isnan(v[0]): descriptors_[j] = v[0]
                    j += 1
                descriptors_tensor[i,:,k] = descriptors_
        return descriptors_tensor

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def SVCFeaturesLoader(
    dataset_path,
    loader_device,
    num_columns,
    num_windows,
    window_size,
    batch_size,
    shuffle=True,
    num_workers=0,
    num_descriptors=5,
):
    dataset = SVCFeaturesDataset(dataset_path, loader_device, num_columns, num_windows, window_size, num_descriptors)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True
    )


class SVCUnpaddedDataset(Dataset):

    def __init__(self, dataset_path, loader_device, num_columns):
        super().__init__()
        self.loader_device = loader_device
        self.num_columns = num_columns
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in sorted(os.listdir(os.path.join(dataset_path,foldername))):
                self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.list_of_all = []
        self.full_length = len(self.filelist)

        for file_index in range(self.full_length):
            filepath = self.filelist[file_index]
            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            tensor = torch.from_numpy(
                pandas.read_csv(filepath, header=None,sep=",").to_numpy()
            ).to(self.loader_device).type(torch.float32)

            tensor = tensor[:,:self.num_columns]

            self.list_of_all.append([tensor, self.target(label), label])

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def SVCPaddedLoader(
    dataset_path,
    loader_device,
    num_columns,
    max_rows,
    batch_size,
    shuffle=True,
    num_workers=0,
    padding='center', # or end padded
):
    dataset = SVCUnpaddedDataset(dataset_path, loader_device, num_columns)
    padder = collate_by_center_padding(max_rows) if padding == 'center' else collate_by_end_padding(max_rows)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True,collate_fn=padder
    )


class SVCInterpolatedDataset(Dataset):

    def __init__(self, dataset_path, loader_device, num_columns):
        super().__init__()
        self.loader_device = loader_device
        self.num_columns = num_columns
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in sorted(os.listdir(os.path.join(dataset_path,foldername))):
                self.filelist.append(os.path.join(dataset_path,foldername,filename))

        self.list_of_all = []
        self.full_length = len(self.filelist)

        for file_index in range(self.full_length):
            filepath = self.filelist[file_index]
            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            tensor = torch.from_numpy(
                pandas.read_csv(filepath, header=None,sep=",").to_numpy()
            ).to(self.loader_device).type(torch.float32)

            tensor = tensor[:,:self.num_columns]

            self.list_of_all.append([tensor, self.target(label), label])

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def SVCRawLoader(
    dataset_path,
    loader_device,
    num_columns,
    batch_size,
    shuffle=True,
    num_workers=0
):
    dataset = SVCInterpolatedDataset(dataset_path, loader_device, num_columns)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True
    )


class SVCImageSet(Dataset):

    def __init__(
        self,
        dataset_path,
        loader_device,
        image_size,
    ):
        super().__init__()
        self.loader_device = loader_device
        self.image_size = image_size
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in os.listdir(os.path.join(dataset_path,foldername)):
                    self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x.to(self.loader_device)
                ),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1)
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.list_of_all = []
        for index in range(len(self.filelist)):
            filepath = self.filelist[index]

            x = Image.open(filepath).convert('L')
            x = self.transform_image(x).type(torch.float32)

            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            self.list_of_all.append([x, self.target(label), label])

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def SVCImageLoader(
    dataset_path,
    loader_device,
    image_size,
    batch_size,
    shuffle=True,
    num_workers=0
):
    dataset = SVCImageSet(dataset_path,loader_device,image_size)
    return DataLoader(
         dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True
    )


class ICPRFeaturesDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        loader_device,
        num_columns,
        num_windows,
        window_size,
        num_descriptors,
        minimum=True,
        maximum=True,
        mean=True,
        median=True,
        skewness=True,
        kurtosis=True,
    ):
        super().__init__()
        self.loader_device = loader_device
        self.num_columns = num_columns
        self.num_windows = num_windows
        self.window_size = window_size
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in os.listdir(os.path.join(dataset_path,foldername)):
                    self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.median = median
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.num_descriptors = sum(int(desc) for desc in [self.minimum,self.maximum,self.mean,self.median,self.skewness,self.kurtosis])
        if self.num_descriptors != num_descriptors:
            raise Exception("Invalid Number of Descriptors!")
        self.list_of_all = []
        self.full_length = len(self.filelist)

        for file_index in range(self.full_length):
            filepath = self.filelist[file_index]
            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            tensor = torch.from_numpy(
                np.genfromtxt(filepath,delimiter=',')
            ).to(self.loader_device).type(torch.float32)

            tensor = tensor[:,:self.num_columns]

            start_seq = [
                round((i * (tensor.shape[0] - self.window_size)) / (self.num_windows - 1))
                for i in range(self.num_windows)
            ]
            out = [
                tensor[start_seq[idx] : start_seq[idx] + self.window_size, :]
                for idx in range(self.num_windows)
            ]
            out.append(tensor)

            out = self.descriptors(out)

            self.list_of_all.append([out, self.target(label), label])

    def descriptors(self, tensor_list):
        descriptors_tensor = torch.zeros([self.num_windows+1,self.num_descriptors,self.num_columns]).to(self.loader_device)
        v = torch.zeros([1])
        for i,tensor in enumerate(tensor_list):
            descriptors_ = torch.zeros([self.num_descriptors])
            for k in range(self.num_columns):
                j = 0
                if self.maximum:
                    descriptors_[j],_ = torch.max(tensor[:,k],0)
                    j += 1
                if self.mean:
                    descriptors_[j] = torch.mean(tensor[:,k],dim=0)
                    j += 1
                if self.median:
                    descriptors_[j],_ = torch.median(tensor[:,k],dim=0)
                    j += 1
                if self.skewness:
                    v[0] = Skewness(tensor[:,k].cpu(),nan_policy='raise')
                    if not torch.isnan(v[0]): descriptors_[j] = v[0]
                    j += 1
                if self.kurtosis:
                    v[0] = Kurtosis(tensor[:,k].cpu(),nan_policy='raise')
                    if not torch.isnan(v[0]): descriptors_[j] = v[0]
                    j += 1
                descriptors_tensor[i,:,k] = descriptors_
        return descriptors_tensor

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def ICPRFeaturesLoader(
    dataset_path,
    loader_device,
    num_columns,
    num_windows,
    window_size,
    batch_size,
    shuffle=True,
    num_workers=0,
    num_descriptors=5,
):
    dataset = ICPRFeaturesDataset(dataset_path, loader_device, num_columns, num_windows, window_size, num_descriptors)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True
    )


class ICPRUnpaddedDataset(Dataset):

    def __init__(self, dataset_path, loader_device, num_columns):
        super().__init__()
        self.loader_device = loader_device
        self.num_columns = num_columns
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in sorted(os.listdir(os.path.join(dataset_path,foldername))):
                self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.list_of_all = []
        self.full_length = len(self.filelist)

        for file_index in range(self.full_length):
            filepath = self.filelist[file_index]
            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            tensor = torch.from_numpy(
                pandas.read_csv(filepath, header=None,sep=",").to_numpy()
            ).to(self.loader_device).type(torch.float32)

            tensor = tensor[:,:self.num_columns]

            self.list_of_all.append([tensor, self.target(label), label])

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def ICPRPaddedLoader(
    dataset_path,
    loader_device,
    num_columns,
    max_rows,
    batch_size,
    shuffle=True,
    num_workers=0,
    padding='center', # or end padded
):
    dataset = ICPRUnpaddedDataset(dataset_path, loader_device, num_columns)
    padder = collate_by_center_padding(max_rows) if padding == 'center' else collate_by_end_padding(max_rows)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True,collate_fn=padder
    )


class ICPRInterpolatedDataset(Dataset):

    def __init__(self, dataset_path, loader_device, num_columns):
        super().__init__()
        self.loader_device = loader_device
        self.num_columns = num_columns
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in sorted(os.listdir(os.path.join(dataset_path,foldername))):
                self.filelist.append(os.path.join(dataset_path,foldername,filename))

        self.list_of_all = []
        self.full_length = len(self.filelist)

        for file_index in range(self.full_length):
            filepath = self.filelist[file_index]
            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            tensor = torch.from_numpy(
                pandas.read_csv(filepath, header=None,sep=",").to_numpy()
            ).to(self.loader_device).type(torch.float32)

            tensor = tensor[:,:self.num_columns]

            self.list_of_all.append([tensor, self.target(label), label])

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def ICPRRawLoader(
    dataset_path,
    loader_device,
    num_columns,
    batch_size,
    shuffle=True,
    num_workers=0
):
    dataset = ICPRInterpolatedDataset(dataset_path, loader_device, num_columns)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True
    )


class ICPRImageSet(Dataset):

    def __init__(
        self,
        dataset_path,
        loader_device,
        image_size,
    ):
        super().__init__()
        self.loader_device = loader_device
        self.image_size = image_size
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in os.listdir(os.path.join(dataset_path,foldername)):
                    self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x.to(self.loader_device)
                ),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1)
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.list_of_all = []
        for index in range(len(self.filelist)):
            filepath = self.filelist[index]

            x = Image.open(filepath).convert('L')
            x = self.transform_image(x).type(torch.float32)

            foldername = os.path.basename(os.path.split(filepath)[0])
            label = self.folderlist.index(foldername)

            self.list_of_all.append([x, self.target(label), label])

    def target(self,label): # Return one-hot tensor
        tensor = torch.zeros([len(self.folderlist)]).to(self.loader_device)
        tensor[label] = 1.
        return tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, index):
        return self.list_of_all[index]


def ICPRImageLoader(
    dataset_path,
    loader_device,
    image_size,
    batch_size,
    shuffle=True,
    num_workers=0
):
    dataset = ICPRImageSet(dataset_path,loader_device,image_size)
    return DataLoader(
         dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True
    )

