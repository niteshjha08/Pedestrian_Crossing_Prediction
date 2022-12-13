#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def find_best_model():
    min_val_loss = 10000
    best_model_count = -1
    val_losses = []
    model_count = 1
    for i in range(108):
        file_path = "nn_models/model_count_{}/history.csv".format(str(model_count))
        print(file_path)
        f = open(file_path, "r")
        lines = f.readlines()
        val_loss = float(lines[-1].split(",")[2])
        val_losses.append(val_loss)
        if(val_loss < min_val_loss):
            min_val_loss = val_loss
            best_model_count = model_count
        model_count+=1
    print("Best model count: {}".format(best_model_count))
    print("Min val loss: {}".format(min_val_loss))
    plt.plot(val_losses)
    plt.xlabel("Models")
    plt.ylabel("Validation loss")
    plt.show() 

if __name__ == "__main__":
    find_best_model()