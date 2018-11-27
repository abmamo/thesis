import pickle
from model import Trainer

train_data = pickle.load( open( "train_data.pkl", "rb" ) )
test_data = pickle.load( open( "test_data.pkl", "rb" ) )
trainer = Trainer(train_data, test_data)
model = trainer.train()
print('training accuracy = {}'.format(trainer.evaluate(model, trainer.train_data)))
print('test accuracy = {}'.format(trainer.evaluate(model, trainer.test_data)))
