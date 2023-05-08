import os
import torch
import torch.nn as nn
import torch.optim as optim
import loader
import model


class ReadConfig:

    fit_map = {
        0: "raw_dataset",
        1: "2d_projections",
        2: "pen_tip",
        3: "both_balls",
    }

    loader_to_loader = {
        0: "FeaturesLoader",
        1: "ImageLoader",
        2: "PaddedLoader",
        3: "RawLoader",
        4: "OrientedLoader",
        5: "OrientedFeaturesLoader"
    }

    loader_map = {
        0: "features",
        1: "2d_image",
        2: "padded",
        3: "raw_loader",
        4: "raw_loader",
        5: "raw_loader",
    }

    def __init__(self, config) -> None:
        self.config = config

        self.dataset_name = self.config["dataset"]["name"]
        self.augmented = self.config["dataset"]["augmentation"]
        self.fit_id = self.config["dataset"]["fit"]
        self.loader_id = self.config["dataset"]["loader"]
        self.architecture_name = self.config["model"]

        self.num_classes = self.config[self.dataset_name]["no_of_classes"]

        self.batch_size = self.config["training"]["batch_size"]
        self.num_workers = self.config["training"]["num_workers"]

        self.loader_device = torch.device(self.config["devices"]["dataloading"])

        self.fit = self.config[self.dataset_name]["fits"][self.fit_map[self.fit_id]]
        self.loader = self.config[self.dataset_name]["loaders"][self.loader_map[self.loader_id]]

        self.path_to_dataset = (
            os.path.join(
                self.config["paths"]["datasets"],
                self.config[self.dataset_name]["path"],
                self.fit["aug_path"]
            )
            if self.augmented
            else os.path.join(
                self.config["paths"]["datasets"],
                self.config[self.dataset_name]["path"],
                self.fit["path"]
            )
        )

        if self.fit_id != 1:
            self.num_rows = self.fit["rows"]
            self.num_columns = self.fit["columns"]
        else:
            self.image_size = self.fit["size"]

        self.Dataloader = eval(f'loader.{self.dataset_name}{self.loader_to_loader[self.loader_id]}')
        self.Model = eval(f'model.{self.architecture_name}')

    def train_loader(self):
        path_ = os.path.join(self.path_to_dataset,"Train")
        dic_ = {
            'dataset_path': path_,
            'loader_device': self.loader_device,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'shuffle': True,
        }
        if self.loader_id == 0 or self.loader_id == 5:
            dic_["num_columns"] = self.num_columns
            dic_["num_windows"] = self.loader["no_of_windows"]
            dic_["window_size"] = self.loader["window_size"]
            dic_["num_descriptors"] = self.loader["no_of_descriptors"]
        elif self.loader_id == 1:
            dic_["image_size"] = self.image_size
        elif self.loader_id == 2:
            dic_["num_columns"] = self.num_columns
            dic_["max_rows"] = self.num_rows
            dic_["padding"] = 'center' if not (self.architecture_name == 'SimpleGRU' or self.architecture_name == 'SimpleLSTM') else 'end'
        elif self.loader_id == 3 or self.loader_id == 4:
            dic_["num_columns"] = self.num_columns
        else:
            raise Exception("Invalid Dataset.")
        return self.Dataloader(**dic_)

    def val_loader(self):
        path_ = os.path.join(self.path_to_dataset,"Validation")
        dic_ = {
            'dataset_path': path_,
            'loader_device': self.loader_device,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'shuffle': True,
        }
        if self.loader_id == 0 or self.loader_id == 5:
            dic_["num_columns"] = self.num_columns
            dic_["num_windows"] = self.loader["no_of_windows"]
            dic_["window_size"] = self.loader["window_size"]
            dic_["num_descriptors"] = self.loader["no_of_descriptors"]
        elif self.loader_id == 1:
            dic_["image_size"] = self.image_size
        elif self.loader_id == 2:
            dic_["num_columns"] = self.num_columns
            dic_["max_rows"] = self.num_rows
            dic_["padding"] = 'center' if not (self.architecture_name == 'SimpleGRU' or self.architecture_name == 'SimpleLSTM') else 'end'
        elif self.loader_id == 3 or self.loader_id == 4:
            dic_["num_columns"] = self.num_columns
        else:
            raise Exception("Invalid Dataset.")
        return self.Dataloader(**dic_)

    def test_loader(self):
        path_ = os.path.join(self.path_to_dataset,"Test")
        dic_ = {
            'dataset_path': path_,
            'loader_device': self.loader_device,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'shuffle': True,
        }
        if self.loader_id == 0 or self.loader_id == 5:
            dic_["num_columns"] = self.num_columns
            dic_["num_windows"] = self.loader["no_of_windows"]
            dic_["window_size"] = self.loader["window_size"]
            dic_["num_descriptors"] = self.loader["no_of_descriptors"]
        elif self.loader_id == 1:
            dic_["image_size"] = self.image_size
        elif self.loader_id == 2:
            dic_["num_columns"] = self.num_columns
            dic_["max_rows"] = self.num_rows
            dic_["padding"] = 'center' if not (self.architecture_name == 'SimpleGRU' or self.architecture_name == 'SimpleLSTM') else 'end'
        elif self.loader_id == 3 or self.loader_id == 4:
            dic_["num_columns"] = self.num_columns
        else:
            raise Exception("Invalid Dataset.")
        return self.Dataloader(**dic_)

    def forged_loader(self):
        path_ = os.path.join(
            self.config["paths"]["datasets"],
            self.config[self.dataset_name]["path"],
            self.fit["forged_path"]
        )
        dic_ = {
            'dataset_path': path_,
            'loader_device': self.loader_device,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'shuffle': True,
        }
        if self.loader_id == 0 or self.loader_id == 5:
            dic_["num_columns"] = self.num_columns
            dic_["num_windows"] = self.loader["no_of_windows"]
            dic_["window_size"] = self.loader["window_size"]
            dic_["num_descriptors"] = self.loader["no_of_descriptors"]
        elif self.loader_id == 1:
            dic_["image_size"] = self.image_size
        elif self.loader_id == 2:
            dic_["num_columns"] = self.num_columns
            dic_["max_rows"] = self.num_rows
            dic_["padding"] = 'center' if not (self.architecture_name == 'SimpleGRU' or self.architecture_name == 'SimpleLSTM') else 'end'
        elif self.loader_id == 3 or self.loader_id == 4:
            dic_["num_columns"] = self.num_columns
        else:
            raise Exception("Invalid Dataset.")
        return self.Dataloader(**dic_)

    def model(self):
        dic_ = {
            'num_classes': self.num_classes,
        }
        if self.architecture_name in ["NewGRU","OldGRU","NewLSTM","OldLSTM"]:
            dic_["num_columns"] = self.num_columns if not (self.loader_id == 4 or self.loader_id == 5) else self.num_columns+3
            dic_["num_windows"] = self.loader["no_of_windows"] + 1 # extra window for combined descriptors
            dic_["num_descriptors"] = self.loader["no_of_descriptors"]
            dic_["dropout_rate"] = self.config["stc"]["dropout_rate"]
            dic_["hidden_size_1"] = self.config["stc"]["hidden_size_1"]
            dic_["hidden_size_2"] = self.config["stc"]["hidden_size_2"]
            dic_["hidden_size_3"] = self.config["stc"]["hidden_size_3"]
            dic_["hidden_size_4"] = self.config["stc"]["hidden_size_4"]
            dic_["layer1_bidirectional"] = self.config["stc"]["layer1_bidirectional"]
            dic_["layer2_bidirectional"] = self.config["stc"]["layer2_bidirectional"]
        elif self.architecture_name == "CNN1D":
            dic_["num_rows"] = self.num_rows
            dic_["num_columns"] = self.num_columns
        elif self.architecture_name == "VGG16":
            pass
        elif self.architecture_name == "SimpleLSTM":
            dic_["num_rows"] = self.num_rows
            dic_["num_columns"] = self.num_columns
            dic_["hidden_size_1"] = self.config["simple_lstm"]["hidden_size_1"]
            dic_["hidden_size_2"] = self.config["simple_lstm"]["hidden_size_2"]
            dic_["bidirectional"] = self.config["simple_lstm"]["bidirectional"]
        elif self.architecture_name == "SimpleGRU":
            dic_["num_rows"] = self.num_rows
            dic_["num_columns"] = self.num_columns
            dic_["hidden_size_1"] = self.config["simple_gru"]["hidden_size_1"]
            dic_["hidden_size_2"] = self.config["simple_gru"]["hidden_size_2"]
            dic_["bidirectional"] = self.config["simple_gru"]["bidirectional"]
        elif self.architecture_name == "CNN2D":
            dic_["num_rows"] = self.num_rows
            dic_["num_columns"] = self.num_columns
        elif self.architecture_name == "SliTCNN2D":
            dic_["num_rows"] = self.num_rows
            dic_["num_columns"] = self.num_columns
        else:
            raise Exception("Invalid Model.")
        return self.Model(**dic_)

    def loss_func(self):
        loss_func = self.config["training"]["loss_function"]
        if loss_func == "CrossEntropyLoss":
            return nn.CrossEntropyLoss
        elif loss_func == "BCELoss":
            return nn.BCELoss
        else:
            raise Exception("Invalid Loss Function.")

    def optimizer_func(self):
        optimizer_fn = self.config["training"]["optimizer"]
        if optimizer_fn == "Adam":
            return optim.Adam
        elif optimizer_fn == "SGD":
            return optim.SGD
        else:
            raise Exception("Invalid Optimizer.")
