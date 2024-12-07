import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Return the device type
def get_device():
    # Choose a GPU if we have one on our machine
    if torch.backends.cuda.is_built():
        # if we have cuda
        device = "cuda"
    elif torch.backends.mps.is_built():
        # if we have MPS
        device = "mps"
    else:
        # if not we should use our CPU
        device = "cpu"

    return device


# Test function
def test(model, test_loader, loss_function):
    # we first move our model to the configured device
    device = get_device()
    model = model.to(device=device)

    # we make sure we are not tracking gradient
    # gradient is used in training, we do not need it for test
    with torch.no_grad():
        risk = 0
        accuracy = 0

        # here we are only evaluating the model
        model.eval()

        # loop over test mini-batches
        for i, (audio, labels) in enumerate(test_loader):
            # reshape labels to have the same form as output
            # make sure labels are of torch.long type
            labels = labels.long()

            # move tensors to the configured device
            audio = audio.to(device=device)
            labels = labels.to(device=device)
            # print(labels)

            # forward pass
            outputs = model(audio)
            loss = loss_function(outputs, labels)

            # determine the class of output from softmax output
            softmax = torch.nn.Softmax(dim=1)
            predicted_probs = softmax(outputs)
            predicted_class = torch.argmax(predicted_probs,dim =1)
            # print(f'Class: {predicted_class}')

            # compute the fraction of correctly predicted labels
            correct_predict = (predicted_class == labels).float().mean()

            risk += loss.item()
            accuracy += correct_predict.item()

        # average test risk and accuracy over the whole test dataset
        test_risk = risk / len(test_loader)
        test_accuracy = accuracy / len(test_loader)

    return test_risk, test_accuracy


# Training function
def train(model, train_loader, val_loader, num_epochs, lr, loss_fn):
    # we first move our model to the configured device
    device = get_device()
    model = model.to(device=device)

    # set loss to cross entropy loss
    loss_function = loss_fn

    # Set optimizer with optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initiate the values
    train_risk = []
    val_risk = []
    val_accuracy = []

    for epoch in range(num_epochs):
        # training risk in one epoch
        risk = 0

        # tell pytorch that you start training
        model.train()

        # loop over training data
        for i, (audio, labels) in enumerate(train_loader):

            # reshape labels to have the same form as output
            # make sure labels are of torch.long type
            labels = labels.long()

            # move tensors to the configured device
            audio = audio.to(device=device)
            labels = labels.to(device=device)


            # forward pass
            outputs = model(audio)
            loss = loss_function(outputs, labels)

            # collect the training loss
            risk += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # one step of gradient descent
            optimizer.step()

        # first we tell pytorch we are doing evaluation (reassure it, as we have already included it in test())
        model.eval()

        # test out model after update by the optimizer
        risk_epoch, accuracy_epoch = test(model, val_loader, loss_function)

        # collect losses and accuracy
        train_risk.append(risk / len(train_loader))
        val_risk.append(risk_epoch)
        val_accuracy.append(accuracy_epoch)

        # we can print a message every second epoch
        if (epoch + 1) % 2 == 0:
            print(f'Epoch {epoch + 1}: Train Risk = {train_risk[-1]:.3f}, Validation Risk = {val_risk[-1]:.3f},'
                  f'Validation Accuracy {val_accuracy[-1]:.3f}')

    # plot the training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot([i + 1 for i in range(num_epochs)], train_risk, label='train')
    plt.plot([i + 1 for i in range(num_epochs)], val_risk, label='validation')
    plt.xlim(1,10)
    plt.legend()
    plt.title('Train and Validation Risk')
    plt.xlabel('Epoch')
    plt.ylabel('Risk')
    plt.show()

    # plot the validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot([i + 1 for i in range(num_epochs)], val_accuracy)
    plt.xlim(1, 10)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    return train_risk, val_risk, val_accuracy

