import modules
from config import PruningConfig, LoRaConfig

class GradualWanda():
    def __init__(self, model_path):
        self.set_model_path(model_path)

    def set_model_path(self, model_path):
        self.model_path = model_path

    # def evaluate(self):
    #     modules.evaluate(self.model_path)

    # def gradual_pruning(self):
    #     modules.gradual_pruning(self.model_path)

    def lora_finetune(self, config: LoRaConfig):
        modules.lora_finetune(config, self.model_path)

    def prune_wanda(self, config: PruningConfig):
        modules.prune_wanda(config, self.model_path)

    def evaluate_model_sparsity(self):
        return modules.evaluate_model_sparsity(self.model_path)
