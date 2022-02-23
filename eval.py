import torch
from neuralnetwork import *
from data import *


MODEL_NAME = "2-SGD-CEL"

PATH = f"{MODEL_NAME}_save.pt"

def load_model(PATH):
	model = Convolutional().to(device)
	model.load_state_dict(torch.load(PATH))
	model.eval()
	return model


#model = torch.jit.load(f'model_scripted_{MODEL_NAME}.pt')
model = load_model(PATH)
model.eval()

data = Data()
test_set = data.test_set(1000)

graficar = False
with torch.no_grad():
	accuracy = 0
	for batch in test_set:
		images, target = batch
		output = model(images)

		matches  = [torch.argmax(i) == j for i,j in zip(output, target)]
		print(f'Matchs en el bacht {matches.count(True)}')
		accuracy += matches.count(True)
		
		if graficar:
			for i, image in enumerate(images):
				plt.title(f'Target/Predction: {data.label[target[i].cpu().item()]} - {data.label[result.cpu().item()]}')
				plt.imshow(image.cpu().view(28, 28))
				plt.show()
	print(f"Accuracy: {round(accuracy/10000,2)*(100)} %")

