import pickle
from generator import Generator


g = Generator('0123456789', 2, 3)
train_data = g.generate_data(200)
g.save("train_data.pkl")
test_data = g.generate_data(10)
g.save("test_data.pkl")
