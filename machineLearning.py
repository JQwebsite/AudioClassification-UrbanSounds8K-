import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class machineLearning():

    def __init__(self, model, trainDL, valDL, loggerConfig):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trainDL = trainDL
        self.valDL = valDL
        self.loggerConfig = loggerConfig
        if int(loggerConfig['master_logger']):
            title = loggerConfig['title'] if loggerConfig[
                'title'] else datetime.now().strftime("%Y-%m-%d,%H-%M-%S")
            self.writer = SummaryWriter(f'./logs/{title}')
            self.writer.close()

    def getDevice(self):
        return next(self.model.parameters()).device

    def write_tb_graph(self, dataloader):
        if int(self.loggerConfig['log_graph']) and int(
                self.loggerConfig['master_logger']):
            spec, label = next(iter(dataloader))
            self.writer.add_graph(self.model, spec.to(self.device))
            self.writer.close()

    def train(self, cost, optimizer):
        dataloader = self.trainDL
        batch_size = len(next(iter(dataloader))[1])
        total_batch = len(dataloader)
        train_loss, train_accuracy = 0, 0
        train_size = batch_size * total_batch

        self.model.train()

        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(self.device), Y.to(self.device)
            optimizer.zero_grad()
            pred = self.model(X)
            batch_loss = cost(pred, Y)
            batch_loss.backward()
            optimizer.step()
            batch_accuracy = (pred.argmax(1) == Y).type(torch.float).sum()
            train_loss += batch_loss.item()
            train_accuracy += batch_accuracy.item()
            if batch % 100 == 0:
                print(
                    f"Training batch {batch}/{total_batch} -> Loss: {batch_loss.item()}  Accuracy: {batch_accuracy.item()/batch_size*100}%"
                )
        train_loss /= train_size
        train_accuracy /= train_size / 100
        return (train_loss, train_accuracy)

    def val(self, cost):
        dataloader = self.valDL
        val_size = len(dataloader.dataset)
        batch_size = len(next(iter(dataloader))[1])
        total_batch = len(dataloader)
        val_loss, val_accuracy = 0, 0

        self.model.eval()

        with torch.no_grad():
            for batch, (X, Y) in enumerate(dataloader):
                X, Y = X.to(self.device), Y.to(self.device)
                pred = self.model(X)
                batch_loss = cost(pred, Y)
                batch_accuracy = (pred.argmax(1) == Y).type(torch.float).sum()
                val_loss += batch_loss.item()
                val_accuracy += batch_accuracy.item()
                if batch % 10 == 0:
                    print(
                        f"Validation batch {batch}/{total_batch} -> Loss: {batch_loss.item()}  Accuracy: {batch_accuracy.item()/batch_size*100}%"
                    )

        val_loss /= val_size
        val_accuracy /= val_size / 100
        return (val_loss, val_accuracy)

    def tensorBoardLogging(self, train_loss, train_accuracy, val_loss,
                           val_accuracy, epoch):
        if self.loggerConfig['log_iter_params'] and int(
                self.loggerConfig['master_logger']):
            self.writer.add_scalar('1 Training/1 Model loss', train_loss,
                                   epoch)
            self.writer.add_scalar('1 Training/2 Model accuracy',
                                   train_accuracy, epoch)
            self.writer.add_scalar('2 Validate/1 Model loss', val_loss, epoch)
            self.writer.add_scalar('2 Validate/2 Model accuracy', val_accuracy,
                                   epoch)
            self.writer.close()