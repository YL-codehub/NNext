from backend.models.model_nnext import ModelNNext
from configs.config import get_cfg
import optuna

# but then should probably separate test set and valid test... and what about cross valid?
def objective(trial):

    model.cfg['model']['training']['n_epochs'] = trial.suggest_int("n_epochs", 10, 10)  # Can also be suggested
    model.cfg['model']['training']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    
    # Suggest parameters for Conv2d and Linear layers
    conv_out_channels = [trial.suggest_int("conv_out_channels", 8, 16)]
    linear_units = [
        trial.suggest_int("linear1_units", 128, 128),
        trial.suggest_int("linear2_units", 256, 256),
        trial.suggest_int("linear3_units", 128, 128),
        trial.suggest_int("linear4_units", 64, 64)
    ]

    # Update model architecture with the suggested values
    model.cfg['model']['architecture'][0]['Conv2d']['args'] = [1, conv_out_channels[0], 3, 1, 1]  # Update Conv2d args
    model.cfg['model']['architecture'][1]['BatchNorm2d']['args'] = [conv_out_channels[0]]
    model.cfg['model']['architecture'][4]['Linear']['args'] = [conv_out_channels[0] * 10 * 5, linear_units[0]]  # Update first Linear layer
    model.cfg['model']['architecture'][6]['Linear']['args'] = [linear_units[0], linear_units[1]]
    model.cfg['model']['architecture'][8]['Linear']['args'] = [linear_units[1], linear_units[2]]
    model.cfg['model']['architecture'][10]['Linear']['args'] = [linear_units[2], linear_units[3]]
    model.cfg['model']['architecture'][12]['Linear']['args'] = [linear_units[3], 5]  # Final output to match the 1x5 target
    
    
    model.build()
    loss_valid = model.train(verbose = False)
    print(loss_valid)
    return loss_valid

if __name__ == "__main__":
    cfg = get_cfg("model_finance_convo.yaml")
    model = ModelNNext(cfg, flat_input = False)
    model.prepare_data()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters: ", study.best_params)
    print("Best validation loss: ", study.best_value)
