import numpy as np
import matplotlib.pyplot as plt

MODEL_NAME = [
    "1-ADAM-CEL",
    "1-ADAM-MSE",
    "1-SGD-CEL",
    "1-SGD-MSE",
    "2-ADAM-CEL",
    "2-ADAM-MSE",
    "2-SGD-CEL",
    "2-SGD-MSE",
]

models = [
    "M(Red 1, ADAM, CEL)",
    "M(Red 1, ADAM, MSE)",
    "M(Red 1, SGD, CEL)",
    "M(Red 1, SGD, MSE)",
    "M(Red 2, ADAM, CEL)",
    "M(Red 2, ADAM, MSE)",
    "M(Red 2, SGD, CEL)",
    "M(Red 2, SGD, MSE)",
]


for i, model in enumerate(models):
    PATH = f"Parameters exploration/{model}/model-{MODEL_NAME[i]}-100.log"
    print(PATH)

    with open(PATH, "r") as contents:

        contents = contents.read().split("\n")

        times = []

        accuracies = []
        val_accs = []
        losses = []
        val_losses = []

        for c in contents:
            if MODEL_NAME[i] in c:

                name, epoch, timestamp, loss, val_loss, acc, val_acc = c.split(",")

                times.append(float(timestamp))

                losses.append(float(loss))
                accuracies.append(float(acc))

                val_losses.append(float(val_loss))
                val_accs.append(float(val_acc))

        normalize_time = np.array([i for i in range(len(times))]) / 60

        val_mean_acc_train = [
            np.array(accuracies)[i * 60 : (i + 1) * 60].mean() for i in range(100)
        ]
        val_mean_acc_test = [
            np.array(val_accs)[i * 60 : (i + 1) * 60].mean() for i in range(100)
        ]

        ax1 = plt.subplot()
        #ax2 = plt.subplot()
        # ax1.plot([i for i in range(100)], val_mean_acc_test,  label = f'{MODEL_NAME[i]}')

        step = 20
        if i < 4:
            ax1.set_title("Red 1")
            ax1.set_ylim(0, 1)
            ax1.grid("on")
            ax1.plot(
                normalize_time[::step], val_accs[::step], label=f"M({MODEL_NAME[i]})"
            )
            ax1.plot([i for i in range(100)], val_mean_acc_train, color="black")
            ax1.legend()
        if i>1000:
            ax2.set_title("Red 2")
            ax2.set_ylim(0, 1)
            ax2.grid("on")
            ax2.plot(
                normalize_time[::step], val_accs[::step], label=f"M({MODEL_NAME[i]})"
            )
            ax2.plot([i for i in range(100)], val_mean_acc_train, color="black")

# plt.title('Models Accuracy')
# plt.xlim(0,100)
plt.legend()
plt.show()
