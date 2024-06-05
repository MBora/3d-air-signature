import os
import shutil
import time
import pandas
import torch
import torch.optim as optim
from read import ReadConfig
from write import Table, TrainLog
import torch.nn as nn
import numpy as np


def clear_directory(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            try:
                os.unlink(os.path.join(root, file))
            except Exception as e:
                print(f"Error deleting file {file}: {e}")
        for dir in dirs:
            try:
                shutil.rmtree(os.path.join(root, dir))
            except Exception as e:
                print(f"Error deleting directory {dir}: {e}")


def Train(config,name,writer,checkpoints_dir,logs_dir):

    model_device = config["devices"]["model"]
    training_device = config["devices"]["training"]
    logging_file = os.path.join(logs_dir,f'TrainLog_{name}.csv')

    num_epochs = config["training"]["no_of_epochs"]
    learning_rate = config["training"]["learning_rate"]

    Config = ReadConfig(config)

    loss_function = Config.loss_func()

    optimizer_function = Config.optimizer_func()

    model = Config.model()

    model.to(model_device)

    criterion = loss_function().to(training_device)

    optimizer = optimizer_function(model.parameters(),lr=learning_rate)

    trainloader = Config.train_loader()

    valloader = Config.val_loader()

    testloader = Config.test_loader()

    del Config

    if config["training"]["learning_rate_scheduler"]["enable"]:
        optim.lr_scheduler.StepLR(optimizer,step_size=config["training"]["learning_rate_scheduler"]["every_n_epochs"],gamma=config["training"]["learning_rate_scheduler"]["factor"])

    table = Table(num_epochs=num_epochs,train_num_batches=len(trainloader),val_num_batches=len(valloader),test_num_batches=len(testloader))
    log = TrainLog(logging_file)

    best_epoch_val_acc = 0.0

    val_losses = []

    rolling_avg = 100000000.

    for epoch in range(num_epochs):

        training_loss = 0.0
        training_acc = 0.0
        total_reconstruction_loss = 0.0
        total_classification_loss = 0.0

        model.train()
        table.train_header(epoch+1)
        # if epoch % 50 == 0:
        # if epoch % 3 == 0:
        base_dir = "train_samples_two_stream" 
        clear_directory(base_dir)  # Clear the base directory at the start of each epoch
        for col,batch_data in enumerate(trainloader):
            current_data, next_data, current_label, next_label = batch_data
            # current_data = current_data[:,:,:3]
            # next_data = next_data[:, :, :3]
            current_data = current_data.to(model_device)
            next_data = next_data.to(model_device)
            output, reconstructed, elbow = model(current_data, next_data, mode="train")
            output.to(training_device)
            reconstructed.to(training_device)
            # target = target.to(training_device)

            # print(data.shape)
            # acc = torch.sum(torch.argmax(output,-1)==torch.argmax(target,1))
            # training_acc += acc.item()
            optimizer.zero_grad()
            # print(reconstructed.view(next_data.size(0), -1).shape)
            # print(next_data.to(training_device).view(next_data.size(0), -1).shape)
            # input()
            # reconstruction_loss = nn.MSELoss()(reconstructed.view(next_data.size(0), -1), next_data.to(training_device).view(next_data.size(0), -1))
            reconstruction_loss = elbow 
            # classification_loss = criterion(output,target) 
            loss = elbow
            # print(loss)
            training_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_classification_loss = 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
            optimizer.step()
            table.train_batch(epoch+1,col+1,current_label,torch.argmax(output,-1),loss)
            del loss, reconstruction_loss
            if epoch == 120:
                for i in range(current_label.size(0)):
                    label = current_label[i].item()
                    label_dir = f"train_samples_two_stream/label_{label}"
                    os.makedirs(label_dir, exist_ok=True)
                    
                    sample_id = f"sample_{col}_{i}"
                    current_sample_path = os.path.join(label_dir, f"{sample_id}_current.npy")
                    next_sample_path = os.path.join(label_dir, f"{sample_id}_next.npy")
                    reconstructed_sample_path = os.path.join(label_dir, f"{sample_id}_reconstructed.npy")
                    
                    np.save(current_sample_path, current_data[i].detach().cpu().numpy())
                    np.save(next_sample_path, next_data[i].detach().cpu().numpy())
                    np.save(reconstructed_sample_path, reconstructed[i].detach().cpu().numpy())
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {training_loss/len(trainloader):.4f}, Reconstruction Loss: {total_reconstruction_loss/len(trainloader):.4f}")

        training_loss /= float(len(trainloader.dataset))
        training_acc /= float(len(trainloader.dataset))
        table.train_end(epoch+1,training_loss,training_acc)

        writer.add_scalar('Training Loss', training_loss, epoch+1)
        writer.add_scalar('Training Accuracy', training_acc, epoch+1)
        writer.add_scalar('Training Reconstruction Loss', total_reconstruction_loss, epoch+1)
        val_loss = 0.0
        val_acc = 0.0
        val_reconstruction_loss = 0.0

        model.eval()
        with torch.inference_mode():
            table.val_header(epoch+1)
            
            base_dir = "validation_samples_two_stream"
            clear_directory(base_dir)  # Clear the base directory at the start of each epoch

            for col, batch_data in enumerate(valloader):
                # Unpack the batch data
                current_data, next_data, current_label, next_label = batch_data

                # Preprocess the data (similar to training)
                # current_data = current_data[:, :, :3]
                # next_data = next_data[:, :, :3]

                # Move data to the appropriate device
                current_data = current_data.to(model_device)
                next_data = next_data.to(model_device)

                # Forward pass
                output, reconstructed, elbow = model(current_data, next_data, mode="val")

                # Move output back to training device if necessary
                output = output.to(training_device)
                reconstructed = reconstructed.to(training_device)

                # Compute losses (Use your specific loss function)
                reconstruction_loss = elbow  # if elbow is used as loss in your context
                loss = reconstruction_loss

                # Accumulate the validation loss
                val_loss += loss.item()
                val_reconstruction_loss += reconstruction_loss.item()

                # Optionally calculate classification accuracy
                # acc = torch.sum(torch.argmax(output, -1) == torch.argmax(next_label, 1))
                # val_acc += acc.item()

                # Log batch results
                table.val_batch(epoch+1, col+1, current_label, torch.argmax(output, -1), loss)

                # Optionally save samples
                # Store samples individually
                if epoch == 120:
                    for i in range(current_label.size(0)):
                        label = current_label[i].item()
                        label_dir = f"validation_samples_two_stream/label_{label}"
                        os.makedirs(label_dir, exist_ok=True)
                        
                        sample_id = f"sample_{col}_{i}"
                        current_sample_path = os.path.join(label_dir, f"{sample_id}_current.npy")
                        next_sample_path = os.path.join(label_dir, f"{sample_id}_next.npy")
                        reconstructed_sample_path = os.path.join(label_dir, f"{sample_id}_reconstructed.npy")

                        np.save(current_sample_path, current_data[i].detach().cpu().numpy())
                        np.save(next_sample_path, next_data[i].detach().cpu().numpy())
                        np.save(reconstructed_sample_path, reconstructed[i].detach().cpu().numpy())
                
            # Normalize the loss and accuracy over the dataset
            val_loss /= len(valloader)
            val_reconstruction_loss /= len(valloader)
            # val_acc /= len(valloader.dataset)

            # End of validation epoch logging
            table.val_end(epoch+1, val_loss, val_acc)
            writer.add_scalar("Validation Loss", val_loss, epoch+1)
            writer.add_scalar("Validation Reconstruction Loss", val_reconstruction_loss, epoch+1)
            writer.add_scalar("Validation Accuracy", val_acc, epoch+1)

            for name_, param in model.named_parameters():
                writer.add_histogram(name_, param, epoch+1)
                if param.grad is not None:
                    writer.add_histogram(name_ + '/grad', param.grad, epoch+1)

        log.log_epoch(epoch+1,training_loss,training_acc,val_loss,val_acc)

        if config["training"]["learning_rate_decay_by_val_loss"]["enable"]:
            if epoch >= config["training"]["learning_rate_decay_by_val_loss"]["every_n_epochs"]-1 and sum(val_losses[-5:])/5.0 > rolling_avg:
                for g in optimizer.param_groups:
                    g['lr'] *= config["training"]["learning_rate_decay_by_val_loss"]["factor"]

        rolling_avg = sum(val_losses[-5:])/5.0

        writer.close()

        if config["checkpoint_save"] == "All":
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint,os.path.join(checkpoints_dir,f'checkpoint_epoch_{epoch+1}_{name}.pt'))

            table.save_checkpoint(f'checkpoint_epoch_{epoch+1}_{name}.pt')

        elif config["checkpoint_save"] == "BestValidationAccuracy":

            if best_epoch_val_acc < val_acc:

                best_epoch_val_acc = val_acc

                for checkpoint_name in os.listdir(checkpoints_dir):
                    os.remove(os.path.join(checkpoints_dir,checkpoint_name))

                checkpoint = {
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint,os.path.join(checkpoints_dir,f'checkpoint_epoch_{epoch+1}_{name}.pt'))

                table.save_checkpoint(f'checkpoint_epoch_{epoch+1}_{name}.pt')
