import modules

class GradualWanda():
    def __init__(self, model_path):
        self.set_model_path(model_path)

    def set_model_path(self, model_path):
        self.model_path = model_path

    def evaluate(self):
        modules.evaluate(self.model_path)

    def gradual_pruning(self):
        modules.gradual_pruning(self.model_path)

    def lora_finetune(self):
        modules.lora_finetune(self.model_path)

    def prune_wanda(self):
        modules.prune_wanda(self.model_path)

    def evaluate_model_sparsity(self):
        modules.evaluate_model_sparsity(self.model_path)
