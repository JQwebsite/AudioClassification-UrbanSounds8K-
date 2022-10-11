import torch
from tqdm import tqdm


def write_tb_graph(writer, model, dataloader):
    
        spec, label = next(iter(dataloader))
        writer.add_graph(model, spec.to(device))
        writer.close()


def train(model, dataloader, cost, optimizer, device):
    batch_size = len(next(iter(dataloader))[1])
    total_batch = len(dataloader)
    train_loss, train_accuracy = 0, 0
    train_size = batch_size * total_batch
    model.train()
    for batch, (X, Y) in tqdm(enumerate(dataloader)):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        batch_loss = cost(pred, Y)
        batch_loss.backward()
        optimizer.step()
        batch_accuracy = (pred.argmax(1) == Y).type(torch.float).sum()
        train_loss += batch_loss.item()
        train_accuracy += batch_accuracy.item()
        if batch % 5 == 0:
            print(
                f" Loss: {batch_loss.item()}  Accuracy: {batch_accuracy.item()/batch_size*100}%"
            )
    train_loss /= train_size
    train_accuracy /= train_size / 100
    return (train_loss, train_accuracy)


def val(model, dataloader, cost, device):
    val_size = len(dataloader.dataset)
    batch_size = len(next(iter(dataloader))[1])
    total_batch = len(dataloader)
    val_loss, val_accuracy = 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
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


def tensorBoardLogging(writer, train_loss, train_accuracy, val_loss,
                       val_accuracy, epoch, loggerConfig):
    if loggerConfig['log_iter_params'] and int(loggerConfig['master_logger']):
        writer.add_scalar('1 Training/1 Model loss', train_loss, epoch)
        writer.add_scalar('1 Training/2 Model accuracy', train_accuracy, epoch)
        writer.add_scalar('2 Validate/1 Model loss', val_loss, epoch)
        writer.add_scalar('2 Validate/2 Model accuracy', val_accuracy, epoch)
        writer.close()