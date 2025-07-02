# MNIST Digit Classification Using Spiking Neural Networks (SNNs)

## Step-by-Step Explanation of the Approach

### 1. **Data Preprocessing and Spike Encoding**
- **Dataset:** MNIST handwritten digits (28x28 grayscale images, 10 classes).
- **Preprocessing:** Images are normalized to range.
- **Spike Encoding:** Each pixel’s intensity is converted to a spike train using **Poisson rate coding**. For each time step, a pixel fires (produces a spike) with probability proportional to its intensity. This creates a temporal sequence of binary events for each input image.

### 2. **SNN Model Architecture**
- **Layers:**
  - Input layer: Receives spike trains (one neuron per pixel).
  - Hidden layer: Leaky Integrate-and-Fire (LIF) neurons.
  - Output layer: LIF neurons, one per digit class.
- **Neuron Model:** LIF neurons accumulate input until a threshold is reached, then emit a spike and reset.
- **Learning Rule:** Supervised learning with surrogate gradients, allowing the use of backpropagation through the non-differentiable spike function.

### 3. **Training and Evaluation**
- **Training:** The SNN is trained using mini-batch stochastic gradient descent. Loss is computed using cross-entropy between the sum of output spikes for each class and the true label.
- **Evaluation:** After training, the model is evaluated on the test set by summing output spikes over all time steps and selecting the class with the highest count as the prediction.

## Visualizations

see .ipynb

## Observations, Challenges, and Solutions

### **Observations**
- **SNNs can learn to classify MNIST digits, but require more careful tuning than ANNs.**
- **Accuracy:** With proper tuning, SNNs can reach 90–98% accuracy on MNIST.
- **Spiking Activity:** Most neurons are silent at any given time step, leading to sparse activity.

### **Challenges Faced**
1. **Low Initial Accuracy (~10–15%):**
   - *Cause:* Too few simulation time steps, poor weight initialization, or low spike rates.
   - *Solution:* Increased time steps (25–50), used Kaiming initialization, verified spike rates.
2. **Training Instability:**
   - *Cause:* Surrogate gradient not propagating well.
   - *Solution:* Used a well-tested surrogate gradient (fast sigmoid), tuned learning rate.
3. **Slow Convergence:**
   - *Cause:* SNNs require more epochs than ANNs.
   - *Solution:* Trained for 10+ epochs, reduced batch size for more frequent updates.
4. **Device Mismatch Errors:**
   - *Cause:* Tensors on different devices (CPU vs CUDA).
   - *Solution:* Ensured all tensors are created on the correct device.

## Reflections: SNN vs. ANN

### **Information Processing**

| Feature         | ANN                                  | SNN                                      |
|-----------------|--------------------------------------|------------------------------------------|
| Signal Type     | Continuous activations (e.g., ReLU)  | Discrete spikes (binary events)          |
| Temporal Coding | None (static per input)              | Temporal (spikes over time)              |
| Biological Plausibility | Low                         | High (closer to real neural behavior)    |

- **ANNs** process all information in one forward pass using continuous values.
- **SNNs** process information over multiple time steps using spikes, mimicking biological neurons.

### **Training Dynamics**

| Aspect            | ANN                               | SNN                                          |
|-------------------|-----------------------------------|----------------------------------------------|
| Backpropagation   | Standard, well-optimized          | Requires surrogate gradients for spikes      |
| Convergence Speed | Fast (few epochs)                 | Slower (more epochs, more tuning needed)     |
| Stability         | High                              | Sensitive to hyperparameters, spike rates    |

- SNNs are harder to train due to non-differentiable spike functions and temporal dynamics.

### **Computational Characteristics**

| Characteristic  | ANN                                 | SNN                                       |
|-----------------|-------------------------------------|-------------------------------------------|
| Sparsity        | Low (all neurons active per layer)  | High (most neurons silent at each step)   |
| Energy Usage    | High (many MAC operations)          | Low (event-driven, fewer operations)      |
| Hardware        | Standard CPUs/GPUs                  | Neuromorphic chips (for max efficiency)   |

- SNNs are more energy-efficient, especially when implemented on neuromorphic hardware, due to their sparse and event-driven computation.
- SNNs can process temporal data naturally, making them suitable for real-time and sensory applications.

## References

https://neuro4ml.github.io/

https://github.com/adianandgit/MNIST-digit-classification-using-SNN/tree/main


## **Summary Table: SNN vs. ANN**

| Feature              | ANN                          | SNN                               |
|----------------------|------------------------------|-----------------------------------|
| Signal               | Continuous                   | Spikes (binary, temporal)         |
| Computation          | Dense                        | Sparse, event-driven              |
| Training             | Standard backpropagation     | Surrogate gradients, BPTT         |
| Energy Efficiency    | Lower                        | Higher (on neuromorphic hardware) |
| Biological Plausibility | Low                        | High                              |
| Temporal Processing  | Limited                      | Natural fit                       |
