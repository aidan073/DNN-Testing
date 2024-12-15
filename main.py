from preprocess import process
from trainer import train
from evaluate import evaluate

data_path = "/mnt/netstore1_home/aidan.bell@maine.edu/gym_members_exercise_tracking.csv"
train_data, test_data, val_data = process(data_path)
model1 = train(train_data, val_data, 32, 10, optimizer_name="AdamW")
model2 = train(train_data, val_data, 32, 10, optimizer_name="AdaBelief")
model3 = train(train_data, val_data, 32, 10, optimizer_name="SGD")
model4 = train(train_data, val_data, 32, 10, optimizer_name="Adagrad")
evaluate(model1, train_data, test_data, val_data, "results1.txt")
evaluate(model2, train_data, test_data, val_data, "results2.txt")
evaluate(model3, train_data, test_data, val_data, "results3.txt")
evaluate(model4, train_data, test_data, val_data, "results4.txt")
