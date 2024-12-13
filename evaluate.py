import torch
from sklearn.metrics import accuracy_score

def evaluate(model, train_data, test_data, val_data, outfile_path):
    train_pred = torch.argmax(model(torch.tensor(train_data.data, dtype=torch.float32)), dim=1)
    test_pred = torch.argmax(model(torch.tensor(test_data.data, dtype=torch.float32)), dim=1)
    val_pred = torch.argmax(model(torch.tensor(val_data.data, dtype=torch.float32)), dim=1)
    tl = train_data.labels
    train_acc = accuracy_score(torch.tensor(train_data.labels.values), train_pred)
    test_acc = accuracy_score(torch.tensor(test_data.labels.values), test_pred)
    val_acc = accuracy_score(torch.tensor(val_data.labels.values), val_pred)
    with open(outfile_path, "w") as f:
        f.write(f"train_acc: {train_acc}\ntest_acc: {test_acc}\nval_acc: {val_acc}")