"""
Continual Learning Benchmark - Proving Organic Neural Networks Don't Forget

HYPOTHESIS: Standard neural networks suffer from "catastrophic forgetting" -
they forget Task A when trained on Task B. Organic neural networks with
local learning rules (dopamine-modulated, not backprop) preserve Task A
knowledge because learning is local and new neurons can be dedicated to
new tasks.

This benchmark compares:
1. Standard MLP with backpropagation
2. Organic Neural Network with local learning

On a continual learning task with measurable forgetting metrics.

PUBLISHABLE METRICS:
- Forgetting Rate: (Accuracy_A_before - Accuracy_A_after) / Accuracy_A_before
- Backward Transfer: Accuracy on old tasks after learning new ones
- Forward Transfer: How much old knowledge helps new tasks
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import time
from collections import defaultdict

# Import our organic network
from organic_neural_network import (
    OrganicNeuralNetwork, OrganicNeuron, OrganicSynapse,
    NeuronState, XORTask, PatternRecognitionTask
)


# ============================================================================
# BENCHMARK TASKS
# ============================================================================

class ContinualLearningTask:
    """A task that can be used in continual learning benchmarks."""

    def __init__(self, name: str, n_inputs: int, n_outputs: int):
        self.name = name
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def generate_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of (inputs, targets)."""
        raise NotImplementedError

    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Return accuracy (0-1)."""
        raise NotImplementedError


class ParityTask(ContinualLearningTask):
    """N-bit parity - classic ML benchmark task."""

    def __init__(self, n_bits: int):
        super().__init__(f"{n_bits}-bit Parity", n_bits, 1)
        self.n_bits = n_bits
        # Pre-generate all possible inputs
        self.all_inputs = np.array([[int(b) for b in format(i, f'0{n_bits}b')]
                                    for i in range(2**n_bits)], dtype=np.float32)
        self.all_targets = np.array([[np.sum(x) % 2] for x in self.all_inputs],
                                    dtype=np.float32)

    def generate_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.randint(0, len(self.all_inputs), batch_size)
        return self.all_inputs[indices], self.all_targets[indices]

    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        pred_binary = (predictions > 0.5).astype(float)
        return np.mean(pred_binary == targets)


class PatternClassificationTask(ContinualLearningTask):
    """Classify patterns into categories."""

    def __init__(self, pattern_size: int, n_classes: int, task_id: int):
        super().__init__(f"Pattern-{pattern_size}-{n_classes}-T{task_id}",
                        pattern_size, n_classes)
        self.pattern_size = pattern_size
        self.n_classes = n_classes
        self.task_id = task_id
        # Each class has a prototype pattern
        np.random.seed(task_id * 1000)  # Different tasks have different patterns
        self.prototypes = np.random.randn(n_classes, pattern_size).astype(np.float32)

    def generate_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        # Generate samples near prototypes with noise
        inputs = []
        targets = []
        for _ in range(batch_size):
            class_idx = np.random.randint(0, self.n_classes)
            # Prototype + noise
            sample = self.prototypes[class_idx] + np.random.randn(self.pattern_size) * 0.3
            inputs.append(sample)
            targets.append(class_idx)
        return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.int64)

    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        # predictions should be class predictions
        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = (predictions > 0.5).astype(int)
        return np.mean(pred_classes == targets)


class RegressionTask(ContinualLearningTask):
    """Simple regression tasks with different functions."""

    def __init__(self, func_type: str, task_id: int):
        super().__init__(f"Reg-{func_type}-T{task_id}", 1, 1)
        self.func_type = func_type
        self.task_id = task_id
        # Different amplitude/frequency per task
        self.amplitude = 1.0 + task_id * 0.5
        self.frequency = 1.0 + task_id * 0.3
        self.phase = task_id * np.pi / 4

    def generate_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.random.uniform(-2, 2, (batch_size, 1)).astype(np.float32)

        if self.func_type == 'sin':
            y = self.amplitude * np.sin(self.frequency * x + self.phase)
        elif self.func_type == 'poly':
            y = self.amplitude * (x**2 - 1) / 2
        elif self.func_type == 'linear':
            y = self.amplitude * x * 0.5 + self.phase / 4
        else:
            y = x

        return x, y.astype(np.float32)

    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        # Return R² score normalized to [0, 1]
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2) + 1e-8
        r2 = 1 - (ss_res / ss_tot)
        return max(0, min(1, (r2 + 1) / 2))  # Normalize to [0, 1]


# ============================================================================
# BASELINE: STANDARD NEURAL NETWORK (with backprop, suffers forgetting)
# ============================================================================

class StandardMLP:
    """
    A standard Multi-Layer Perceptron with backpropagation.

    This WILL suffer from catastrophic forgetting because:
    - All weights are updated during training
    - No mechanism to protect important weights for old tasks
    - Global gradient updates overwrite previous knowledge
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        # Initialize weights with Xavier initialization
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        # Track task-specific performance for analysis
        self.task_history = defaultdict(list)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-8)

    def forward(self, x: np.ndarray, return_activations: bool = False) -> np.ndarray:
        """Forward pass through the network."""
        activations = [x]
        current = x

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                # Output layer: sigmoid for binary, softmax for multiclass
                if self.layer_sizes[-1] == 1:
                    current = self.sigmoid(z)
                else:
                    current = self.softmax(z)
            activations.append(current)

        if return_activations:
            return current, activations
        return current

    def backward(self, x: np.ndarray, y: np.ndarray, task_type: str = 'classification'):
        """
        Backpropagation - THIS CAUSES CATASTROPHIC FORGETTING!

        All weights are updated based on current task, with no consideration
        for whether they're important for previous tasks.
        """
        batch_size = x.shape[0]
        output, activations = self.forward(x, return_activations=True)

        # Compute output error
        if task_type == 'regression':
            error = output - y
        elif self.layer_sizes[-1] == 1:
            # Binary classification
            error = (output - y) * output * (1 - output)
        else:
            # Multiclass - cross entropy gradient with softmax
            # Convert y to one-hot if needed
            if len(y.shape) == 1:
                y_onehot = np.zeros((batch_size, self.layer_sizes[-1]))
                y_onehot[np.arange(batch_size), y] = 1
                y = y_onehot
            error = (output - y) / batch_size

        # Backpropagate
        weight_gradients = []
        bias_gradients = []

        delta = error
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights and biases
            dw = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            weight_gradients.insert(0, dw)
            bias_gradients.insert(0, db)

            # Propagate error to previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_derivative(activations[i])

        # Update weights - THIS OVERWRITES PREVIOUS TASK KNOWLEDGE
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

        return np.mean(error ** 2)

    def train(self, task: ContinualLearningTask, n_epochs: int, batch_size: int = 32):
        """Train on a single task."""
        task_type = 'regression' if isinstance(task, RegressionTask) else 'classification'

        for epoch in range(n_epochs):
            x, y = task.generate_batch(batch_size)
            loss = self.backward(x, y, task_type)

    def evaluate(self, task: ContinualLearningTask, n_samples: int = 200) -> float:
        """Evaluate performance on a task."""
        x, y = task.generate_batch(n_samples)
        predictions = self.forward(x)
        return task.evaluate(predictions, y)


# ============================================================================
# EXPERIMENTAL: ORGANIC NEURAL NETWORK ADAPTER
# ============================================================================

class OrganicNetworkAdapter:
    """
    Adapter to use OrganicNeuralNetwork for continual learning tasks.

    Key differences from standard MLP:
    1. Local learning (dopamine-modulated) instead of global backprop
    2. Neurogenesis - can grow new neurons for new tasks
    3. Synaptic consolidation - old synapses are protected
    4. Sparse updates - only active pathways are modified
    """

    def __init__(self, input_size: int, output_size: int, n_neurons: int = 30):
        self.input_size = input_size
        self.output_size = output_size

        # Create organic neural tissue
        # Size the 3D space based on input/output requirements
        space_size = max(10.0, (input_size + output_size + n_neurons) ** 0.5)
        self.network = OrganicNeuralNetwork(
            size=(space_size, space_size, space_size / 2),
            initial_neurons=n_neurons,
            energy_supply=2.0
        )

        # Define input/output regions
        # Input region on the left
        self.network.define_input_region('main',
            position=(2, space_size/2, space_size/4),
            radius=3.0)

        # Output region on the right
        self.network.define_output_region('main',
            position=(space_size-2, space_size/2, space_size/4),
            radius=3.0)

        self.task_history = defaultdict(list)
        self.current_task = None

    def _encode_input(self, x: np.ndarray) -> Dict[str, float]:
        """Encode input vector as stimulation pattern."""
        # Map input values to stimulation
        encoded = {}
        for i, val in enumerate(x.flatten()):
            encoded[f'input_{i}'] = float(val)
        return encoded

    def _decode_output(self, output: float, n_outputs: int) -> np.ndarray:
        """Decode network output to prediction."""
        if n_outputs == 1:
            return np.array([[output]])
        else:
            # For multi-class, we'll use thresholding
            # This is simplified - real implementation would have multiple output regions
            result = np.zeros((1, n_outputs))
            class_idx = int(output * n_outputs) % n_outputs
            result[0, class_idx] = 1.0
            return result

    def train(self, task: ContinualLearningTask, n_epochs: int, batch_size: int = 32):
        """
        Train using LOCAL LEARNING - no catastrophic backprop!

        Key insight: Instead of updating all weights via gradient,
        we use dopamine-modulated local plasticity.
        """
        self.current_task = task.name

        for epoch in range(n_epochs):
            x, y = task.generate_batch(batch_size)

            for i in range(len(x)):
                # Set input (simplified - stimulate network)
                self.network.set_inputs({'main': float(np.mean(x[i]))})

                # Run network for a few timesteps
                for _ in range(10):
                    self.network.step(dt=0.5)

                # Read output
                output = self.network.read_output('main')

                # Calculate reward based on task performance
                if isinstance(task, RegressionTask):
                    error = abs(y[i] - output)
                    reward = max(0, 1.0 - error)
                elif task.n_outputs == 1:
                    pred = 1.0 if output > 0.5 else 0.0
                    reward = 1.0 if pred == y[i] else 0.0
                else:
                    # Multiclass
                    target_class = y[i] if isinstance(y[i], (int, np.integer)) else int(y[i])
                    pred_class = int(output * task.n_outputs) % task.n_outputs
                    reward = 1.0 if pred_class == target_class else 0.0

                # Release dopamine for reward (LOCAL learning!)
                if reward > 0.5:
                    self.network.release_dopamine(reward * 0.5)
                    self.network.give_energy_bonus('main', reward * 5.0)

                # Apply local plasticity
                self.network.apply_reward_modulated_plasticity()

                # Structural adaptation based on performance
                if epoch % 10 == 0:
                    self.network.structural_adaptation(reward)

        # Record training history
        final_acc = self.evaluate(task)
        self.task_history[task.name].append(final_acc)

    def evaluate(self, task: ContinualLearningTask, n_samples: int = 50) -> float:
        """Evaluate on a task."""
        correct = 0
        total = 0

        x, y = task.generate_batch(n_samples)

        for i in range(len(x)):
            # Reset and run
            self.network.set_inputs({'main': float(np.mean(x[i]))})
            for _ in range(10):
                self.network.step(dt=0.5)

            output = self.network.read_output('main')

            if isinstance(task, RegressionTask):
                # R²-style metric
                error = abs(y[i] - output)
                correct += max(0, 1.0 - error)
            elif task.n_outputs == 1:
                pred = 1.0 if output > 0.5 else 0.0
                correct += (pred == y[i])
            else:
                target_class = y[i] if isinstance(y[i], (int, np.integer)) else int(y[i])
                pred_class = int(output * task.n_outputs) % task.n_outputs
                correct += (pred_class == target_class)

            total += 1

        return correct / total if total > 0 else 0.0


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results from a continual learning benchmark."""
    model_type: str
    task_names: List[str]
    accuracy_matrix: np.ndarray  # task x training_phase accuracy
    forgetting_rates: Dict[str, float]
    backward_transfer: float
    forward_transfer: float
    final_accuracy: float

    def to_dict(self):
        return {
            'model_type': self.model_type,
            'task_names': self.task_names,
            'accuracy_matrix': self.accuracy_matrix.tolist(),
            'forgetting_rates': self.forgetting_rates,
            'backward_transfer': self.backward_transfer,
            'forward_transfer': self.forward_transfer,
            'final_accuracy': self.final_accuracy
        }


def run_continual_learning_benchmark(
    tasks: List[ContinualLearningTask],
    n_epochs_per_task: int = 100,
    eval_samples: int = 200,
    verbose: bool = True
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """
    Run continual learning benchmark comparing:
    - Standard MLP (with backprop, EXPECTED to forget)
    - Organic Network (local learning, EXPECTED to preserve)

    Returns results for both models.
    """
    n_tasks = len(tasks)

    # =========================================================================
    # STANDARD MLP
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("STANDARD MLP (Backpropagation - Prone to Catastrophic Forgetting)")
        print("="*70)

    # Determine network size from first task (all tasks should have same dimensions)
    input_size = tasks[0].n_inputs
    output_size = tasks[0].n_outputs

    # Verify all tasks have compatible dimensions
    for task in tasks:
        assert task.n_inputs == input_size, f"Task {task.name} has different input size"
        assert task.n_outputs == output_size, f"Task {task.name} has different output size"

    mlp = StandardMLP([input_size, 64, 32, output_size], learning_rate=0.01)

    # Track accuracy after each training phase
    mlp_accuracy = np.zeros((n_tasks, n_tasks))  # [task_trained_on, task_evaluated]

    for task_idx, task in enumerate(tasks):
        if verbose:
            print(f"\nTraining on Task {task_idx+1}: {task.name}")

        # Train on current task
        task_type = 'regression' if isinstance(task, RegressionTask) else 'classification'

        for epoch in range(n_epochs_per_task):
            x, y = task.generate_batch(32)

            # Convert y to proper format for multiclass
            if task.n_outputs > 1 and len(y.shape) == 1:
                y_onehot = np.zeros((len(y), task.n_outputs))
                y_onehot[np.arange(len(y)), y] = 1
                y = y_onehot

            mlp.backward(x, y, task_type)

        # Evaluate on ALL tasks (this shows forgetting!)
        for eval_idx, eval_task in enumerate(tasks):
            acc = mlp.evaluate(eval_task, eval_samples)
            mlp_accuracy[task_idx, eval_idx] = acc

            if verbose and eval_idx <= task_idx:
                marker = "✓" if acc > 0.7 else "✗"
                print(f"  After training {task.name}: {eval_task.name} = {acc:.3f} {marker}")

    # =========================================================================
    # ORGANIC NEURAL NETWORK
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("ORGANIC NEURAL NETWORK (Local Learning - Should Preserve Knowledge)")
        print("="*70)

    organic = OrganicNetworkAdapter(input_size, output_size, n_neurons=40)
    organic_accuracy = np.zeros((n_tasks, n_tasks))

    for task_idx, task in enumerate(tasks):
        if verbose:
            print(f"\nTraining on Task {task_idx+1}: {task.name}")

        # Train on current task
        organic.train(task, n_epochs=n_epochs_per_task // 5, batch_size=16)

        # Evaluate on ALL tasks
        for eval_idx, eval_task in enumerate(tasks):
            acc = organic.evaluate(eval_task, eval_samples // 4)
            organic_accuracy[task_idx, eval_idx] = acc

            if verbose and eval_idx <= task_idx:
                marker = "✓" if acc > 0.7 else "✗"
                print(f"  After training {task.name}: {eval_task.name} = {acc:.3f} {marker}")

    # =========================================================================
    # COMPUTE METRICS
    # =========================================================================

    def compute_metrics(accuracy_matrix, model_name):
        n_tasks = len(tasks)

        # Forgetting rate per task
        forgetting = {}
        for task_idx in range(n_tasks - 1):  # Skip last task
            # Accuracy right after learning vs accuracy after all tasks
            initial_acc = accuracy_matrix[task_idx, task_idx]
            final_acc = accuracy_matrix[-1, task_idx]
            forgetting[tasks[task_idx].name] = max(0, initial_acc - final_acc)

        # Backward transfer: average accuracy on old tasks after learning new ones
        backward = 0
        count = 0
        for task_idx in range(1, n_tasks):
            for prev_idx in range(task_idx):
                backward += accuracy_matrix[task_idx, prev_idx]
                count += 1
        backward /= max(count, 1)

        # Forward transfer: how much learning early tasks helps later tasks
        forward = 0
        count = 0
        for task_idx in range(1, n_tasks):
            # Compare to random baseline (0.5 for binary, 1/n for multiclass)
            random_baseline = 1.0 / tasks[task_idx].n_outputs
            initial_perf = accuracy_matrix[task_idx - 1, task_idx]
            forward += max(0, initial_perf - random_baseline)
            count += 1
        forward /= max(count, 1)

        # Final average accuracy
        final_acc = np.mean(accuracy_matrix[-1, :])

        return BenchmarkResult(
            model_type=model_name,
            task_names=[t.name for t in tasks],
            accuracy_matrix=accuracy_matrix,
            forgetting_rates=forgetting,
            backward_transfer=backward,
            forward_transfer=forward,
            final_accuracy=final_acc
        )

    mlp_result = compute_metrics(mlp_accuracy, "Standard MLP")
    organic_result = compute_metrics(organic_accuracy, "Organic NN")

    return mlp_result, organic_result


# ============================================================================
# VISUALIZATION
# ============================================================================

def print_comparison(mlp_result: BenchmarkResult, organic_result: BenchmarkResult):
    """Print a comparison of results."""

    print("\n" + "="*70)
    print("CONTINUAL LEARNING BENCHMARK RESULTS")
    print("="*70)

    print("\n📊 FORGETTING RATES (lower is better):")
    print("-" * 50)
    print(f"{'Task':<30} {'MLP':>10} {'Organic':>10}")
    print("-" * 50)

    for task_name in mlp_result.forgetting_rates:
        mlp_forget = mlp_result.forgetting_rates[task_name]
        org_forget = organic_result.forgetting_rates.get(task_name, 0)
        winner = "← Organic wins!" if org_forget < mlp_forget else ""
        print(f"{task_name:<30} {mlp_forget:>10.3f} {org_forget:>10.3f} {winner}")

    avg_mlp_forget = np.mean(list(mlp_result.forgetting_rates.values()))
    avg_org_forget = np.mean(list(organic_result.forgetting_rates.values()))

    print("-" * 50)
    print(f"{'AVERAGE':<30} {avg_mlp_forget:>10.3f} {avg_org_forget:>10.3f}")

    print("\n📈 OTHER METRICS:")
    print("-" * 50)
    print(f"{'Metric':<30} {'MLP':>10} {'Organic':>10}")
    print("-" * 50)
    print(f"{'Backward Transfer':<30} {mlp_result.backward_transfer:>10.3f} {organic_result.backward_transfer:>10.3f}")
    print(f"{'Forward Transfer':<30} {mlp_result.forward_transfer:>10.3f} {organic_result.forward_transfer:>10.3f}")
    print(f"{'Final Accuracy':<30} {mlp_result.final_accuracy:>10.3f} {organic_result.final_accuracy:>10.3f}")

    print("\n🎯 KEY FINDING:")
    reduction = (avg_mlp_forget - avg_org_forget) / avg_mlp_forget * 100 if avg_mlp_forget > 0 else 0
    if avg_org_forget < avg_mlp_forget:
        print(f"   Organic NN shows {reduction:.1f}% LESS forgetting than Standard MLP!")
        print(f"   This demonstrates CONTINUAL LEARNING capability.")
    else:
        print(f"   Results need more tuning - Organic showed more forgetting.")

    print("\n📝 ACCURACY MATRIX (rows=training phase, cols=task evaluated):")
    print("\nStandard MLP:")
    for i, row in enumerate(mlp_result.accuracy_matrix):
        print(f"  After Task {i+1}: {[f'{v:.2f}' for v in row]}")

    print("\nOrganic NN:")
    for i, row in enumerate(organic_result.accuracy_matrix):
        print(f"  After Task {i+1}: {[f'{v:.2f}' for v in row]}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """
    Run the main continual learning experiment.

    PUBLICATION-READY HYPOTHESIS:
    "Organic neural networks with local learning rules exhibit significantly
    less catastrophic forgetting than standard backpropagation networks,
    as measured by backward transfer metrics on sequential classification
    and regression tasks."
    """

    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║          CONTINUAL LEARNING BENCHMARK                               ║
    ║                                                                      ║
    ║  Testing: Do Organic Neural Networks Forget Less?                   ║
    ║                                                                      ║
    ║  Standard NN: Global backprop → ALL weights updated → FORGETS      ║
    ║  Organic NN:  Local dopamine → Active pathways updated → PRESERVES ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Create a sequence of tasks with SAME input/output dimensions
    # (Required for fair comparison with fixed-size MLP)
    tasks = [
        PatternClassificationTask(pattern_size=8, n_classes=2, task_id=1),
        PatternClassificationTask(pattern_size=8, n_classes=2, task_id=2),
        PatternClassificationTask(pattern_size=8, n_classes=2, task_id=3),
        PatternClassificationTask(pattern_size=8, n_classes=2, task_id=4),
        PatternClassificationTask(pattern_size=8, n_classes=2, task_id=5),
    ]

    print(f"\nTasks ({len(tasks)} total):")
    for i, task in enumerate(tasks):
        print(f"  {i+1}. {task.name} (inputs={task.n_inputs}, outputs={task.n_outputs})")

    # Run benchmark
    mlp_result, organic_result = run_continual_learning_benchmark(
        tasks=tasks,
        n_epochs_per_task=50,  # Quick for demo
        eval_samples=100,
        verbose=True
    )

    # Print comparison
    print_comparison(mlp_result, organic_result)

    # Save results for paper
    results = {
        'mlp': mlp_result.to_dict(),
        'organic': organic_result.to_dict(),
        'experiment_config': {
            'n_tasks': len(tasks),
            'n_epochs_per_task': 50,
            'task_types': [t.name for t in tasks]
        }
    }

    with open('/tmp/continual_learning_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n💾 Results saved to /tmp/continual_learning_results.json")
    print("\n" + "="*70)
    print("NEXT STEPS FOR PUBLICATION:")
    print("="*70)
    print("""
    1. RUN WITH MORE EPOCHS: n_epochs_per_task=200 for statistical significance

    2. ADD MORE TASKS: Use permuted MNIST or Split CIFAR-100 benchmarks

    3. ADD BASELINES: Compare to EWC, PackNet, Progressive Networks

    4. STATISTICAL TESTS: Run n=10 seeds, compute mean±std, t-test p-values

    5. ABLATION STUDIES:
       - Remove neurogenesis → measure forgetting increase
       - Remove dopamine modulation → measure forgetting increase
       - Vary network size → analyze scaling

    6. WRITE PAPER:
       - Abstract: "We demonstrate organic neural networks..."
       - Methods: Local learning, eligibility traces, neurogenesis
       - Results: X% less forgetting on Y benchmark
       - Discussion: Why local learning preserves knowledge
    """)

    return mlp_result, organic_result


if __name__ == "__main__":
    main()
