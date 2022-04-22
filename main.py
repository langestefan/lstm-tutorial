import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

N = 100     # number of samples
L = 1000    # number of time steps
T = 20    # width

x = np.empty((N, L), dtype=np.float32)
x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
y = np.sin(x/1.0/T, dtype=np.float32)



# class for the model
class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=51):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden

        # lstm1, lstm2, linear for prediction
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(n_samples, self.n_hidden).cuda()
        c_t = torch.zeros(n_samples, self.n_hidden).cuda()
        h_t2 = torch.zeros(n_samples, self.n_hidden).cuda()
        c_t2 = torch.zeros(n_samples, self.n_hidden).cuda()

        for input_t in x.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.cat(outputs, dim=1)
        return outputs

if __name__ == "__main__":
    # training
    # y = 100, 1000
    train_input = torch.from_numpy(y[3:, :-1])  # 97, 999
    train_target = torch.from_numpy(y[3:, 1:])  # 97, 999


    # testing
    test_input = torch.from_numpy(y[:3, :-1]) # 3, 999
    test_target = torch.from_numpy(y[:3, 1:])  # 3, 999

    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # model
    model = LSTMPredictor()
    model.to(device)

    criterion = nn.MSELoss().cuda()
    optimizer = optim.LBFGS(model.parameters(), lr=0.1)

    # move tensors to GPU if available
    dtype = torch.float32

    if torch.cuda.is_available():
        train_input = train_input.cuda()
        train_target = train_target.cuda()
        test_input = test_input.cuda()
        test_target = test_target.cuda()

    n_steps = 100

    # predict future steps
    for i in range(n_steps):
        print("step: ", i)

        # LBFGS requires the loss function to be a callable
        def closure():
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_target)
            print("loss: ", loss.item())
            loss.backward()
            return loss
        
        optimizer.step(closure)

        # evaluate
        with torch.no_grad():
            future = 1000
            pred = model(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            y = pred.detach().cpu().numpy()

        # plot
        plt.figure(figsize=(12, 6))
        plt.title("Step: {}".format(i))

        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        n = train_input.shape[1] # 999

        def draw(yi, color):
            plt.plot(np.arange(n), yi[:n], color, linewidth=2.0)
            plt.plot(np.arange(n, n+future), yi[n:], color + ":", linewidth=2.0)

        draw(y[0], "r")
        draw(y[1], "b")
        draw(y[2], "g")

        plt.savefig("step_{}.png".format(i))
        plt.close()




        




