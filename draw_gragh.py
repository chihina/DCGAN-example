import matplotlib.pyplot as plt

with open("loss_D.log") as f:
    logs = f.readlines()

log_list = [float(i.strip()) for i in logs]
epoch_list = [i for i in range(1,len(logs)+1)]
plt.plot(epoch_list, log_list, label="D_loss")


with open("loss_G.log") as f:
    logs = f.readlines()

log_list = [float(i.strip()) for i in logs]
plt.plot(epoch_list, log_list, label="G_loss")
plt.title("Loss Gragh")
plt.legend(mode="best")
plt.savefig("loss_graph.png")