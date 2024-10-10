from backend.models.model_nnext import ModelNNext
from configs.config import get_cfg
import optuna
import math as m

# but then should probably separate test set and valid test... and what about cross valid?
def create_objective(model):
    n_channels_input = len(model.cfg['data']['entities'])
    L_in = model.cfg['data']['rolling_count']-1

    def objective(trial):

        # Suggest hyperparameters for Conv1d layers
        conv1_out_channels = trial.suggest_int("conv1_out_channels", 1, 128)
        conv1_kernel_size = trial.suggest_int("conv1_kernel_size", 2, 7)
        
        conv2_out_channels = trial.suggest_int("conv2_out_channels", 1, 128)
        conv2_kernel_size = trial.suggest_int("conv2_kernel_size", 2, 7)

        linear_layer_size = trial.suggest_int("linear_layer_size", 16, 128)
        linear_layer2_size = trial.suggest_int("linear_layer_size", 16, 128)

        # Model configuration
        # model.cfg['model']['training']['n_epochs'] = trial.suggest_int("n_epochs", 100, 1000)
        # model.cfg['model']['training']['learning_rate'] = trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True)
        model.cfg['model']['architecture'][0]['Conv1d']['args'] = [n_channels_input, conv1_out_channels, conv1_kernel_size]
        model.cfg['model']['architecture'][2]['MaxPool1d']['args'] = [2]  # Fixed pooling size
        # Compute L_out after the first Conv1d and MaxPool1d
        L_out_conv1 = (L_in - (conv1_kernel_size - 1)) // 2  # Use floor division for pooling
        if L_out_conv1 <= 0:
            # Prune the trial if the size is invalid
            raise optuna.TrialPruned(f"Invalid output size after first Conv1d and MaxPool1d: {L_out_conv1}")

        model.cfg['model']['architecture'][3]['Conv1d']['args'] = [conv1_out_channels, conv2_out_channels, conv2_kernel_size]
        model.cfg['model']['architecture'][5]['MaxPool1d']['args'] = [2]  # Fixed pooling size
        # Compute L_out after the second Conv1d and MaxPool1d
        L_out_conv2 = (L_out_conv1 - (conv2_kernel_size - 1)) // 2
        if L_out_conv2 <= 0:
            # Prune the trial if the size is invalid
            raise optuna.TrialPruned(f"Invalid output size after second Conv1d and MaxPool1d: {L_out_conv2}")

        # Ensure the input size for the Linear layer is valid
        flattened_input_size = conv2_out_channels * L_out_conv2
        if flattened_input_size <= 0:
            # Prune the trial if the size is invalid
            raise optuna.TrialPruned(f"Invalid flattened input size: {flattened_input_size}")

        # Update the Linear layer
        model.cfg['model']['architecture'][7]['Linear']['args'] = [flattened_input_size, linear_layer_size]
        model.cfg['model']['architecture'][9]['Linear']['args'] = [linear_layer_size, linear_layer2_size]
        model.cfg['model']['architecture'][11]['Linear']['args'] = [linear_layer2_size, 5]

        # Build and train the model
        model.build()
        loss_valid = model.train(verbose=False)

        return loss_valid
    
    return objective

if __name__ == "__main__":
    ## should propably fix the training dataset
    cfg = get_cfg("model_finance_convo.yaml")
    model = ModelNNext(cfg, flat_input = False)
    model.prepare_data()
    study = optuna.create_study(direction="minimize")
    # study.enqueue_trial({'n_epochs': 20, 'conv_out_channels': 8, 'kernel_size': 7, 'linear1_units': 176})
    objective = create_objective(model)
    study.optimize(objective, n_trials=10)

    print("Best hyperparameters: ", study.best_params)
    print("Best validation loss: ", study.best_value)

    import optuna.visualization as vis

    # Plot the optimization history
    fig = vis.plot_optimization_history(study)
    fig.write_image('optim_hist.png')
    # Plot the parameter importance
    # fig = vis.plot_param_importances(study)
    # fig.write_image('param_importance.png')
