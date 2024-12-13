from preprocess import process
from trainer import train
from evaluate import evaluate

data_path = "/mnt/netstore1_home/aidan.bell@maine.edu/gym_members_exercise_tracking.csv"
train_data, test_data, val_data = process(data_path)
model = train(train_data, val_data, 32, 10)
evaluate(model, train_data, test_data, val_data, "results.txt")
