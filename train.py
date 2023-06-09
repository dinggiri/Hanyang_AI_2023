from torch.autograd import Variable

def train(X_train, train_y, optimizer, model = None, epochs = 3000, verbose = True):
    train_loss = []
    train_acc = []

    model.train()
    X, y = Variable(X_train), Variable(train_y)
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss = model.calculate_objective(X, y)
        loss = loss.mean()
        acc = model.calculate_classification_error(X, y)
        train_loss.append(loss)
        train_acc.append(acc)

        loss.backward()
        optimizer.step()

        if verbose:
        	print('Epoch: {}, Loss: {:.4f}, Train acc: {:.4f}%'.format(epoch, loss, acc))

    return train_loss, train_acc