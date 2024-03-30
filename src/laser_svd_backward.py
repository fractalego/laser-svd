import sys
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


class LaserLinear(torch.nn.modules.module.Module):
    U: torch.Tensor
    sigma: torch.Tensor
    V: torch.Tensor

    def __init__(self, weight: torch.Tensor, rank_fraction: float, niter: int = 2):
        super().__init__()
        max_rank = min(weight.shape)
        q = int(max_rank * rank_fraction)
        U, sigma, V = torch.svd_lowrank(weight, q=q, niter=niter)
        self.U = torch.nn.Parameter(U)
        self.sigma = torch.nn.Parameter(sigma)
        self.V = torch.nn.Parameter(V)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input @ (self.U @ torch.diag(self.sigma) @ self.V.T).T

if __name__ == "__main__":
    rank_fraction = 0.5
    mlp_names = [
        "mlp.down_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]
    sys.stdout.flush()
    for layer_index in reversed(range(len(model.base_model.layers))):
        for mlp_name in mlp_names:
            original_linear = eval(f"model.base_model.layers[layer_index].{mlp_name}")
            weight = original_linear.weight
            try:
                exec(f"model.model.layers[layer_index].{mlp_name} = LaserLinear(weight, rank_fraction)")
            except AssertionError as e:
                print(f"Layer {layer_index} on {mlp_name} with Q={rank_fraction} is rejected.\nAssertionError: {e}")
                exec(f"model.model.layers[layer_index].{mlp_name} = original_linear")
                sys.stdout.flush()
                continue

            print(f"Layer {layer_index} on {mlp_name} with Q={rank_fraction} is accepted")
            sys.stdout.flush()

        torch.save(model.state_dict(), f"mistral-7b-instruct-laser-distil-backward-{rank_fraction}")
