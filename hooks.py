import torch
import time

# def timing_hook(module, input, output):
#     module._forward_time = time.perf_counter()


# def sparsity_hook(module, input, output):
#     if hasattr(module, 'weight'):
#         weight = module.weight.data
#         num_zeros = torch.sum(weight < 1e-8).item()
#         num_elements = weight.numel()
#         weight_sparsity = num_zeros / num_elements
#         print(f"[Weight Sparsity] {module.__class__.__name__}: {weight_sparsity:.2%}")

#     # Optional: check output (activation) sparsity too
#     if isinstance(output, torch.Tensor):
#         num_zeros_out = torch.sum(output < 1e-8).item()
#         num_elements_out = output.numel()
#         output_sparsity = num_zeros_out / num_elements_out
#         print(f"[Activation Sparsity] {module.__class__.__name__}: {output_sparsity:.2%}")

def make_timing_hook():
    def hook(module, input, output):
        if not hasattr(module, "_forward_time_total"):
            module._forward_time_total = 0.0
            module._forward_time_calls = 0
        # Record start time if not already recorded
        if not hasattr(module, "_start_time"):
            module._start_time = time.perf_counter()

        # Compute elapsed time for this call
        elapsed = time.perf_counter() - module._start_time

        # Update running totals
        module._forward_time_total += elapsed
        module._forward_time_calls += 1

    return hook

def make_running_sparsity_hook():
    def hook(module, input, output):
        if not hasattr(module, "_sparsity_total"):
            module._sparsity_total = 0.0
            module._sparsity_calls = 0

        if not hasattr(module, "_activation_sparsity_total"):
            module._activation_sparsity_total = 0.0
            module._activation_sparsity_calls = 0

        if hasattr(module, 'weight'):
            weight = module.weight.data
            sparsity = (weight < 1e-8).sum().item() / weight.numel()
            module._sparsity_total += sparsity
            module._sparsity_calls += 1

        if isinstance(output, torch.Tensor):
            activation_sparsity = (output < 1e-8).sum().item() / output.numel()
            module._activation_sparsity_total += activation_sparsity
            module._activation_sparsity_calls += 1

    return hook

# Before each forward pass (start timer):
def start_timers(model):
    for module in model.modules():
        if hasattr(module, "_start_time"):
            module._start_time = time.perf_counter()

# After training:
def report_forward_times(model):
    print("\n=== Forward Pass Timing Summary ===")
    for name, module in model.named_modules():
        if hasattr(module, "_forward_time_total"):
            avg_time = module._forward_time_total / module._forward_time_calls
            print(f"{name}: {avg_time*1000:.4f} ms (avg over {module._forward_time_calls} calls)")

def report_average_sparsities(model):
    print("\n=== Average Sparsity per Layer ===")
    for name, module in model.named_modules():
        if hasattr(module, "_sparsity_total") and module._sparsity_calls > 0:
            avg_sparsity = module._sparsity_total / module._sparsity_calls
            print(f"{name} weight sparsity: {avg_sparsity:.2%}")

        if hasattr(module, "_activation_sparsity_total") and module._activation_sparsity_calls > 0:
            avg_sparsity = module._sparsity_total / module._activation_sparsity_calls 
            print(f"{name} activation sparsity: {avg_sparsity:.2%}")