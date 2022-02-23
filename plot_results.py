import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_NAME = "2-SGD-CEL"

def create_acc_loss_graph(model_name=MODEL_NAME, EPOCHS=100):
    contents = open(f"model-{MODEL_NAME}-{EPOCHS}.log", "r").read().split("\n")

    times = []

    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    epoch_ticks = []

    end_of_epoch = 0
    for c in contents:
        if model_name in c:
            name, epoch, timestamp, loss, val_loss, acc, val_acc = c.split(",")

            if end_of_epoch != int(epoch):
                end_of_epoch += 1
                epoch_ticks.append(float(timestamp))

            times.append(float(timestamp))
            
            losses.append(float(loss))
            accuracies.append(float(acc))
            
            val_losses.append(float(val_loss))
            val_accs.append(float(val_acc))

    normalize_time = np.array([i for i in range(len(times))])/60
    print(len(times)/6)
    
    i = 70
    j = 71

    mean_vacc = round(np.array(val_accs)[i*60:j*60].mean(),2)
    std_vacc = round(np.array(val_accs)[i*60:j*60].std(),2)
    mean_acc = round(np.array(accuracies)[i*60:j*60].mean(),2)
    std_acc = round(np.array(accuracies)[i*60:j*60].std(),2)

    val_mean_acc_train = [np.array(accuracies)[i*60:(i+1)*60].mean() for i in range(100)]
    val_mean_acc_test = [np.array(val_accs)[i*60:(i+1)*60].mean() for i in range(100)]
    plt.plot([i for i in range(100)], val_mean_acc_train)
    plt.plot([i for i in range(100)], val_mean_acc_test)
    plt.show() 

    print(f'val {mean_vacc} +/- {std_vacc} , train {mean_acc} +/- {std_acc}')

    fig = plt.figure()
    ax1 = plt.subplot(2,1,1)
    #ax1.set_ylim(0.005, 0.040)
    #ax1.set_xlim(0,30)
    step = 10
    ax1.plot(normalize_time[::step], losses[::step], label="Train loss")
    ax1.plot(normalize_time[::step], val_losses[::step], label="Test loss")
    ax1.legend()



    ax2 = plt.subplot(2,1,2)
    ax2.set_xlabel("EPOCHS")
    #ax2.set_ylim(0.8, 1)
    #ax2.set_xlim(0,30)
    
    ax2.plot(normalize_time[::step], accuracies[::step], label="Train Accuracy")
    ax2.plot(normalize_time[::step], val_accs[::step], label="Test Accuracy")
    #ax2.plot([i for i in range(100)], val_mean_acc_train, color='red')
    #ax2.plot([i for i in range(100)], val_mean_acc_test, color='b')
    ax2.legend()

    plt.savefig(f"loss-{EPOCHS}.png")
    plt.show()
    plt.close("all")



if __name__ == "__main__":
    create_acc_loss_graph()
    
