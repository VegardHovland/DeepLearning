import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
import timeit


if __name__ == "__main__":
    start = timeit.default_timer()
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    if use_momentum:
        learning_rate = .02
    else: 
        learning_rate = .1

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    # Comparison model 4d!    
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    neurons_per_layer = [60, 60, 10]
    if use_momentum:
        learning_rate = .02
    else: 
        learning_rate = .1
    model2 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_2 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model2, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_2, val_history_2 = trainer_2.train(
        num_epochs)

    
    # Comparison model 4e!  
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    if use_momentum:
        learning_rate = .02
    else: 
        learning_rate = .1 

    model3 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_3 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model3, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_3, val_history_3 = trainer_3.train(
        num_epochs)
    

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 3 Model ", npoints_to_average=10)
    utils.plot_loss(
        train_history_2["loss"], "Task 4d Model - two x 60", npoints_to_average=10)
    utils.plot_loss(
        train_history_3["loss"], "Task 4e Model - ten x 64", npoints_to_average=10)    
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1])
    utils.plot_loss(val_history["accuracy"], "Task 3 Model")
    utils.plot_loss(
        val_history_2["accuracy"], "Task 4d Model - two hidden layers, 60 units")
    utils.plot_loss(
        val_history_3["accuracy"], "Task 4e Model - ten x 64 hidden layers")        
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4_train_loss_ten_hidden_layers_early_stop.png")

    stop = timeit.default_timer()
    print('Running Time: ', stop - start) 
    
