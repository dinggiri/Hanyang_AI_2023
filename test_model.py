from torch.autograd import Variable

def test(X_test, y_test, model = None, f1 = False):
    model.eval()
    X_test, y_test = Variable(X_test), Variable(y_test)
    loss = model.calculate_objective(X_test, y_test)
    loss = loss.mean()
    if not f1: # f1 is false
        acc = model.calculate_classification_error(X_test, y_test)
        print('TEST, Loss: {:.4f}, Test acc: {:.4f}%'.format(loss, acc))
        return loss, acc
    else:
        acc, f1_score = model.calculate_classification_error(X_test, y_test, f1 = True)
        print('TEST, Loss: {:.4f}, Test acc: {:.4f}%, F1 macro: {:.4f}'.format(loss, acc, f1_score))
        return loss, acc, f1_score

    