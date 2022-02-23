from train_model import *

EPOCHS = 100

model, optimizer = train_model(EPOCHS=EPOCHS, lr=0.1)


print("\n Model's state_dict:")

for param_tensor in model.state_dict():
	print(param_tensor, "\t", model.state_dict()[param_tensor].size(), "\n")
	
	PATH = f"{MODEL_NAME}_save.pt"
	torch.save(model.state_dict(), PATH)
	
	model_scripted = torch.jit.script(model) 
	model_scripted.save(f'model_scripted_{MODEL_NAME}.pt')



