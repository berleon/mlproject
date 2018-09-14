from sacred.run import Run
import sacred

def make_ml_project(ex: sacred.Experiment, data_loader, model_loader):
    return MLProject(ex, data_loader, model_loader)

def ml_main(_run: sacred.run.Run):
    sacred.commands.print_config(_run)


    print(_run)
    print(_run.config)


def ml_load_experiment():
    pass
