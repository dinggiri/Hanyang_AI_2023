from torch.autograd import Variable

def test(X_test, y_test, model = None):
    model.eval()
    X_test, y_test = Variable(X_test), Variable(y_test)
    loss = model.calculate_objective(X_test, y_test)
    loss = loss.mean()
    acc = model.calculate_classification_error(X_test, y_test)
    print('TEST, Loss: {:.6f}, Test acc: {:.4f}%'.format(loss, acc))

    return loss, acc