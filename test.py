import os
import torch
from read import ReadConfig
from write import Table, TestLog


def Test(config,name,checkpoints_dir,logs_dir):

    def test_checkpoint(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        model_device = config["devices"]["model"]
        training_device = config["devices"]["training"]
        logging_file = os.path.join(logs_dir,f'TestLog_{name}.csv')

        learning_rate = config["training"]["learning_rate"]

        Config = ReadConfig(config)

        loss_function = Config.loss_func()

        optimizer_function = Config.optimizer_func()

        testloader = Config.test_loader()

        model = Config.model()

        model.to(model_device)

        criterion = loss_function().to(training_device)

        optimizer = optimizer_function(model.parameters(),lr=learning_rate)

        del Config

        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        table = Table(test_num_batches=len(testloader))

        log = TestLog(logging_file)

        model.eval()

        testing_loss = 0.0
        testing_acc = 0.0

        with torch.inference_mode():
            table.test_header()
            for i,batch_data in enumerate(testloader):
                data,target,label = batch_data
                output = model(data.to(model_device)).to(training_device)
                target = target.to(training_device)
                acc = torch.sum(torch.argmax(output,1)==torch.argmax(target,1))
                testing_acc += acc.item()
                loss = criterion(output,target)
                testing_loss += loss.item()
                table.test_batch(i+1,label,torch.argmax(output,1),loss)
                del loss
            testing_loss /= float(len(testloader.dataset))
            testing_acc /= float(len(testloader.dataset))
            table.test_end(testing_loss,testing_acc,checkpoint_path)

        log.log_results(name,checkpoint_path,testing_loss,testing_acc)

    if os.path.isdir(checkpoints_dir):
        for checkpoint in os.listdir(checkpoints_dir):
            test_checkpoint(os.path.join(checkpoints_dir,checkpoint))
    else:
        test_checkpoint(checkpoints_dir)
