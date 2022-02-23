from torch import optim
from neuralnetwork import *
from data import *
import random
import time


def target_to_array(target, batch_size=1000):
    return torch.tensor(
        np.array([np.eye(10)[target[i].item()] for i in range(batch_size)]),
        dtype=torch.float32,
        device="cuda",
    )


data = Data()
batch = 1000
train_set = [(i, target_to_array(j, batch)) for i, j in data.train_set(batch)]
test_set = [(i, target_to_array(j, batch)) for i, j in data.test_set(batch)]


#loss_function = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()

MODEL_NAME = "2-SGD-CEL"

###########################
#  FUNCTIONS DEFINITIONS  #
###########################


def init_netowrk(lr=0.001):
    net = Convolutional().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    return net, optimizer


def fwd_pass(image, target, net, optimizer, train=False):
    if train:
        net.zero_grad()

    output = net(image)
    
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(output, target)]
    accuracy = matches.count(True) / len(matches)

    loss = loss_function(output, target)
    
    if train:
        loss.backward()
        optimizer.step()

    return loss, accuracy


def test_model(net, optimizer):
    batch = random.choice(test_set)
    images = batch[0]
    target = batch[1]

    with torch.no_grad():
        test_loss, test_acc = fwd_pass(images, target, net, optimizer)

    return test_loss, test_acc


def train_model(EPOCHS=1, lr=0.001):
    net, optimizer = init_netowrk(lr)
    with open(f"model-{MODEL_NAME}-{EPOCHS}.log", "w", encoding="utf-8") as f:
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        for epoch in range(EPOCHS):
            start = time.perf_counter()
            for batch in train_set:
                images = batch[0]
                target = batch[1]

                loss, acc = fwd_pass(images, target, net, optimizer, train=True)
                test_loss, test_acc = test_model(net, optimizer)

                to_file = float(loss), float(test_loss), float(acc), float(test_acc)
                f.write(
                    f"{MODEL_NAME}, {epoch}, {round(time.time(),3)}, {round(to_file[0],3)}, {round(to_file[1],3)}, {round(to_file[2],3)}, {round(to_file[3],3)}\n"
                )
            end = time.perf_counter()
            print(f"Epoca: {epoch}, {round(end-start,2)}.")
            if epoch%15==0:
                #scheduler.step()
                print('scheduler step')
    
    return net, optimizer
