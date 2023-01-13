import wandb
import random
import os


def train():
    with wandb.init(project="ciao") as run:
        print(run.id)
        print(run.name)
        for i in range(10):
            wandb.log({"metric": i * random.randint(0, 10)})


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


if __name__ == "__main__":
    sweep_id = wandb.sweep({
      "method": "random",
      "metric": {"goal": "maximize", "name": "metric"},
      "parameters":
          {
            "p1": {"values": [1, 2, 3, 4]},
            "p2": {"values": [9, 3]}}}, project="ciao")
    # find the best model with a sweep
    wandb.agent(sweep_id, function=train, count=3)
    # reset_wandb_env()
    for key in os.environ.keys():
        if key.startswith("WANDB_"):
            del os.environ[key]
    with wandb.init(project="ciao", reinit=True, id="new_id", name="run_1") as run:
        # training of the best model with a simple run
        print(run.id)
        print(run.name)

