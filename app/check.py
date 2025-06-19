import pickletools
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
with open(model_path, 'rb') as f:
    data = f.read()

pickletools.dis(data)