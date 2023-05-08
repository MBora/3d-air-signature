import os
import pandas


class PrintRun(object):

    def __init__(self,name) -> None:
        self.name = name

    def train_start(self):
        print("")
        print(f"+{'-'*77:^77s}+")
        print(f"|{f'STARTING TRAINING {self.name}':^77s}|")
        print(f"+{'-'*77:^77s}+")
        print("")

    def train_end(self):
        print("")
        print(f"+{'-'*77:^77s}+")
        print(f"|{f'FINISHED TRAINING {self.name}':^77s}|")
        print(f"+{'-'*77:^77s}+")
        print("")

    def test_start(self):
        print("")
        print(f"+{'-'*77:^77s}+")
        print(f"|{f'STARTING TESTING {self.name}':^77s}|")
        print(f"+{'-'*77:^77s}+")
        print("")

    def test_end(self):
        print("")
        print(f"+{'-'*77:^77s}+")
        print(f"|{f'FINISHED TESTING {self.name}':^77s}|")
        print(f"+{'-'*77:^77s}+")
        print("")


class Table(object):

    def __init__(self,num_epochs=0,train_num_batches=0,val_num_batches=0,test_num_batches=0):
        self.num_epochs = num_epochs
        self.train_num_batches = train_num_batches
        self.val_num_batches = val_num_batches
        self.test_num_batches = test_num_batches

    def train_header(self,epoch):
        print(f"+{'-'*77:^77s}+")
        print(f"|{f'TRAINING EPOCH {epoch} / {self.num_epochs}':^77s}|")
        print(f"+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+")
        print(f"|{'Epoch':^12s}|{f'Batch':^12s}|{'Target':^12s}|{'Prediction':^12s}|{'Correct?':^12s}|{'Loss':^12s}|")
        print(f"+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+")

    def train_batch(self,epoch,batch,target,pred,loss):
        for i in range(len(target)):
            print(f"|{f'{epoch}/{self.num_epochs}':^12s}|{f'{batch}/{self.train_num_batches}':^12s}|{target[i]:^12d}|{pred[i]:^12d}|{str(bool(target[i]==pred[i])):^12s}|{loss:^12.6f}|")
        if (batch == self.train_num_batches): print(f"+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+")
        else: print(f"|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|")

    def train_end(self,epoch,loss,acc):
        print(f"|{f'END OF TRAINING EPOCH {epoch} / {self.num_epochs}':^77s}|")
        print(f"+{'-'*77:^77s}+")
        print(f"|{f'TRAINING LOSS = {loss:<12.6f}':^38s} {f'TRAINING ACCURACY = {acc:<12.6f}':^38s}|")
        print(f"+{'-'*77:^77s}+")

    def val_header(self,epoch):
        print(f"+{'-'*77:^77s}+")
        print(f"|{f'VALIDATION EPOCH {epoch} / {self.num_epochs}':^77s}|")
        print(f"+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+")
        print(f"|{'Epoch':^12s}|{f'Batch':^12s}|{'Target':^12s}|{'Prediction':^12s}|{'Correct?':^12s}|{'Loss':^12s}|")
        print(f"+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+")

    def val_batch(self,epoch,batch,target,pred,loss):
        for i in range(len(target)):
            print(f"|{f'{epoch}/{self.num_epochs}':^12s}|{f'{batch}/{self.val_num_batches}':^12s}|{target[i]:^12d}|{pred[i]:^12d}|{str(bool(target[i]==pred[i])):^12s}|{loss:^12.6f}|")
        if (batch == self.val_num_batches): print(f"+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+")
        else: print(f"|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|")

    def val_end(self,epoch,loss,acc):
        print(f"|{f'END OF VALIDATION EPOCH {epoch} / {self.num_epochs}':^77s}|")
        print(f"+{'-'*77:^77s}+")
        print(f"|{f'VALIDATION LOSS = {loss:<12.6f}':^38s} {f'VALIDATION ACCURACY = {acc:<12.6f}':^38s}|")
        print(f"+{'-'*77:^77s}+")

    def save_checkpoint(self,filename):
        print(f"|{f'CHECKPOINT SAVED AS':^77s}|")
        print(f"|{f'{filename[-77:]}':^77s}|")
        print(f"+{'-'*77:^77s}+")
        print("")

    def test_header(self):
        print(f"+{'-'*64:^64s}+")
        print(f"|{f'TESTING EPOCH':^64s}|")
        print(f"+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+")
        print(f"|{f'Batch':^12s}|{'Target':^12s}|{'Prediction':^12s}|{'Correct?':^12s}|{'Loss':^12s}|")
        print(f"+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+")

    def test_batch(self,batch,target,pred,loss):
        for i in range(len(target)):
            print(f"|{f'{batch}/{self.test_num_batches}':^12s}|{target[i]:^12d}|{pred[i]:^12d}|{str(bool(target[i]==pred[i])):^12s}|{loss:^12.6f}|")
        if (batch == self.test_num_batches): print(f"+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+{'-'*12:^12s}+")
        else: print(f"|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|{'-'*12:^12s}|")

    def test_end(self,loss,acc,checkpoint_path):
        print(f"|{'END OF TESTING EPOCH':^64s}|")
        print(f"+{'-'*64:^64s}+")
        print(f"|{f'TESTING LOSS = {loss:<12.6f}':^32s}{f'TESTING ACCURACY = {acc:<12.6f}':^32s}|")
        print(f"+{'-'*64:^64s}+")
        checkpoint_file = checkpoint_path.split('/')[-1]
        print(f"|{f'CHECKPOINT LOADED FROM':^64s}|")
        print(f"|{f'{checkpoint_file[-64:]}':^64s}|")
        print(f"+{'-'*64:^64s}+")
        print("")


class TrainLog:

    def __init__(self,filepath) -> None:
        self.filepath = filepath
        init_dict = {
            "Epoch": [],
            "Training Loss": [],
            "Training Accuracy": [],
            "Validation Loss": [],
            "Validation Accuracy": [],
        }
        df = pandas.DataFrame(init_dict)
        df.to_csv(self.filepath,float_format="%.6f",header=True,index=False)

    def log_epoch(self,epoch,t_loss,t_acc,v_loss,v_acc):
        current_dict = {
            "Epoch": [epoch],
            "Training Loss": [t_loss],
            "Training Accuracy": [t_acc],
            "Validation Loss": [v_loss],
            "Validation Accuracy": [v_acc],
        }
        df = pandas.DataFrame(current_dict)
        df.to_csv(self.filepath,float_format="%.6f",mode='a',header=False,index=False)


class TestLog:

    def __init__(self,filepath) -> None:
        self.filepath = filepath
        if not os.path.exists(filepath):
            init_dict = {
                "Label": [],
                "Checkpoint": [],
                "Testing Loss": [],
                "Testing Accuracy": [],
            }
            df = pandas.DataFrame(init_dict)
            df.to_csv(self.filepath,float_format="%.6f",header=True,index=False)

    def log_results(self,label,checkpoint,loss,acc):
        current_dict = {
                "Label": [label],
                "Checkpoint": [checkpoint],
                "Testing Loss": [loss],
                "Testing Accuracy": [acc],
        }
        df = pandas.DataFrame(current_dict)
        df.to_csv(self.filepath,float_format="%.6f",mode='a',header=False,index=False)
