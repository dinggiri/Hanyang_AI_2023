from torch.autograd import Variable

def train(X_train, train_y, optimizer, model = None, epochs = 3000, verbose = True, f1 = False):
    train_loss = []
    train_acc = []
    train_f1 = []

    model.train()
    X, y = Variable(X_train), Variable(train_y)
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss = model.calculate_objective(X, y)
        loss = loss.mean()
        if not f1:
            acc = model.calculate_classification_error(X, y)
        else:
            acc, f1_score = model.calculate_classification_error(X, y, f1 = True)
            train_f1.append(f1_score)
        train_loss.append(loss)
        train_acc.append(acc)

        loss.backward()
        optimizer.step()

        if verbose:
            if not f1:  # f1 is false
                print('Epoch: {}, Loss: {:.4f}, Train acc: {:.4f}%'.format(epoch, loss, acc))
            else:
                print('Epoch: {}, Loss: {:.4f}, Train acc: {:.4f}%, F1 macro: {:.4f}'.format(epoch, loss, acc, f1_score))
                

    if not f1:
        return train_loss, train_acc
    else:
        return train_loss, train_acc, train_f1