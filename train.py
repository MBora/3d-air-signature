import os
import time
import pandas
import torch
import torch.optim as optim
from read import ReadConfig
from write import Table, TrainLog


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

        model.train()
        table.train_header(epoch+1)
        for col,batch_data in enumerate(trainloader):
            data,target,label = batch_data
            output = model(data.to(model_device)).to(training_device)
            target = target.to(training_device)
            acc = torch.sum(torch.argmax(output,-1)==torch.argmax(target,1))
            training_acc += acc.item()
            optimizer.zero_grad()
            loss = criterion(output,target)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
            table.train_batch(epoch+1,col+1,label,torch.argmax(output,-1),loss)
            del loss
        training_loss /= float(len(trainloader.dataset))
        training_acc /= float(len(trainloader.dataset))
        table.train_end(epoch+1,training_loss,training_acc)

        writer.add_scalar('Training Loss', training_loss, epoch+1)
        writer.add_scalar('Training Accuracy', training_acc, epoch+1)

        val_loss = 0.0
        val_acc = 0.0

        model.eval()
        with torch.inference_mode():
            table.val_header(epoch+1)
            for col,batch_data in enumerate(valloader):
                data,target,label = batch_data
                output = model(data.to(model_device)).to(training_device)
                target = target.to(training_device)
                acc = torch.sum(torch.argmax(output,-1)==torch.argmax(target,1))
                val_acc += acc.item()
                loss = criterion(output,target)
                val_loss += loss.item()
                table.val_batch(epoch+1,col+1,label,torch.argmax(output,-1),loss)
                del loss
            val_loss /= float(len(valloader.dataset))
            val_losses.append(val_loss)
            val_acc /= float(len(valloader.dataset))
            table.val_end(epoch+1,val_loss,val_acc)

            writer.add_scalar("Validation Loss", val_loss, epoch+1)
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
