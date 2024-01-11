import torch
# from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil


torch.device(0)


def train(model, train_dataloader, epochs, lr, loss_fn, val_dataloader=None, clip_grad=False, method=None,
          input_sequence=None, save_path=None):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    early_stop = 1180
    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    n = 0
    # start = time.time()

    train_losses = []
    for epoch in range(epochs):

        for step, (model_input, gt) in enumerate(train_dataloader):
            start_time = time.time()

            model_output = model(model_input)
            losses = loss_fn(model_output, gt)
            # reg_loss = model['reg_loss']
            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                train_loss = single_loss + train_loss
            # train_loss += reg_loss
            train_losses.append(train_loss.item())

            optim.zero_grad()
            train_loss.backward(retain_graph=True)
            optim.step()

            if clip_grad:
                if isinstance(clip_grad, bool):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optim.step()

            if (n + 1) % 100 == 0:

                # print("Running validation set...")
                model.eval()
                with torch.no_grad():
                    val_loss = 0.
                    for (model_input, gt) in val_dataloader:
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        val_loss += single_loss

                model.train()
                print("Epoch %d, Train loss %0.6f, Val loss %0.6f, iteration time %0.6f" % (
                    epoch, train_loss, val_loss, time.time() - start_time))

                if val_loss < dev_best_loss:
                    dev_best_loss = val_loss
                    torch.save(model.state_dict(), save_path)
                    last_improve = n

            if n - last_improve > early_stop:
                # Stop training if the loss of dev dataset has not dropped
                # exceeds early_stop batches
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            n += 1

        if flag:
            break
            # if epoch % 10==0:
            #     print("Epoch %d, Train loss %0.6f, Val loss %0.6f, iteration time %0.6f" % (epoch, train_loss, val_loss, time.time() - start_time))



if __name__ == '__main__':
    def calculate_convergence_rate(initial_loss, final_loss):
        return (1 - final_loss / initial_loss) * 100


    # 假设初始损失为1.0，最终损失为0.1

    initial_loss = 1.761875
    final_loss = 0.408715

    convergence_rate = calculate_convergence_rate(initial_loss, final_loss)
    print(f'Convergence Rate: {convergence_rate:.2f}%')