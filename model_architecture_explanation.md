# Smashifix Deep Learning Architecture: Complete Technical Guide

This document provides a detailed breakdown of the deep learning architecture driving Smashifix. It is designed to act as a reference guide for defending the project, answering technical questions, and understanding the core concepts behind every layer of the model.

---

## 1. High-Level Overview: Late-Fusion Dual-Stream Hybrid Model

Smashifix uses a **Dual-Stream** architecture, meaning it processes two entirely different types of data simultaneously before making a decision.
1. **The Visual Stream:** Looks at the actual pixels/images (cropped video frames) to understand context like racket position, grip, and body orientation.
2. **The Pose Stream:** Looks only at the 33 3D skeletal landmarks (extracted by MediaPipe) to understand pure biomechanical movement.

**"Late Fusion"** means that these two streams are processed independently by their own dedicated sequence models. They are only combined ("fused") at the very end right before the final classification. This prevents the model from getting confused by trying to mix raw pixels with raw coordinates.

---

## 2. The Visual Backbone: RSN (Residual-Shuffle Network)

Before we can analyze a *sequence* of frames, we must first extract meaningful features from *each individual frame*. We use a custom **RSN (Residual-Shuffle Network)** for this, which acts as the visual feature extractor.

### Concepts to Know:
*   **What is a CNN (Convolutional Neural Network)?** A neural network designed to scan images using "filters" (like a magnifying glass) to detect edges, textures, and eventually complex shapes (like a racket head). RSN is a type of highly optimized CNN.
*   **Why not use MobileNetV2 or ResNet?** Standard models are too computationally expensive for real-time capabilities on standard hardware. RSN is specifically designed to be lightweight and fast without sacrificing accuracy.

### Core Components of RSN:
1.  **Grouped Convolutions:** Instead of having every input channel connect to every output channel (expensive), the channels are split into "groups." Each group is processed independently, massively reducing the number of calculations.
2.  **Channel Shuffling:** The problem with Grouped Convolutions is that the groups never communicate. Channel Shuffling solves this by taking the output of the groups and literally "shuffling" them (like a deck of cards) before the next layer. This ensures cross-group information flow.
3.  **Residual/Skip Connections:** Inspired by ResNet. In deep networks, the learning signal (gradient) can vanish as it passes through many layers. A residual connection takes the input of a layer and adds it directly to the output ($Output = F(x) + x$). This creates a "shortcut" for the gradient, making training deep models much more stable.
4.  **Global Average Pooling (GAP):** Takes the final 2D feature map and squashes it into a flat 1D vector by averaging the values. This 1D vector (projected to **64 dimensions** in our model) represents the core "visual essence" of that single frame.

---

## 3. The Sequence Models: Understanding Time

Once we have a sequence of 64-D visual features and 99-D pose features (33 joints $\times$ 3 coordinates) for, say, 40 continuous frames, we need models that understand **Time**. 

### A. The Visual Stream: Attention-TCN (Temporal Convolutional Network)

Instead of using traditional RNNs or LSTMs for the visual sequence, we use a **TCN**. 

**What is a TCN?** 
A TCN is a convolutional network adapted for sequential/time-series data. It has two strict rules:
1.  **Causal Convolutions:** The network is not allowed to look into the future. To predict frame $T$, it can only look at frame $T$ and past frames ($T-1$, $T-2$). This is crucial for making the model capable of real-time inference.
2.  **Dilated Convolutions:** In a normal convolution, you look at adjacent frames (e.g., frames 1, 2, 3). In a *dilated* convolution, you skip frames to see further back in time without needing massive filters. 
    *   Our model uses dilations of `[1, 2, 4, 8]`. 
    *   This exponentially increasing dilation means the network achieves a **Receptive Field of 31 frames**. In other words, when analyzing the current frame, it has the context of the previous 31 frames simultaneously, completely avoiding the short-term memory limits of LSTMs.

### B. Multi-Head Self-Attention (Used in both Visual and Pose Streams)

**What is Attention?**
Not all frames in a 40-frame sequence are equally important. A frame where the player is waiting to serve is less important than the exact millisecond the racket strikes the shuttle. The Attention mechanism allows the neural network to dynamically assign mathematically higher "weights" (importance) to the critical frames.

**What is "Multi-Head" Self-Attention?**
Instead of just looking for one type of important event, the model uses 8 "heads." 
*   Head 1 might learn to look exclusively for racket-shuttle contact.
*   Head 2 might look for the moment of maximum wrist flexion.
*   Head 3 might look for the foot planting on the ground.
By combining these multiple heads, the model builds an incredibly rich understanding of the entire stroke sequence.

### C. GRU (Gated Recurrent Unit)

After the TCN and Attention layers have processed the sequence, a **GRU** is used to compress the entire sequence down into a single summary vector. 
*   **What is a GRU?** It is a type of Recurrent Neural Network (RNN), very similar to an LSTM, but simpler and faster to train. It uses "gates" (update gate, reset gate) to decide what information to keep in its memory and what to throw away as it reads through the sequence frame by frame.

---

## 4. Late Fusion & Final Classification

At this point, the network has generated two summary vectors:
1.  A summary of the Visual data (processed by TCN $\rightarrow$ Attention $\rightarrow$ GRU).
2.  A summary of the Pose data (processed by Conv1D $\rightarrow$ Attention $\rightarrow$ GRU).

### The Fusion Process:
1.  **Concatenation:** The two vectors are glued together end-to-end to create one massive **163-Dimensional Hybrid Feature Vector**.
2.  **Dense Layers (Fully Connected Layers):** This combined vector is passed through standard Dense layers. Every neuron connects to every other neuron, allowing the network to find correlations between what the visual stream saw and what the pose stream saw.
3.  **Regularization (Dropout & L2):** To prevent the model from "memorizing" the training data (overfitting), we use Dropout (randomly turning off a percentage of neurons during training) and L2 Regularization (penalizing the model for having weights that are too large).
4.  **Softmax Classifier:** The final layer has 6 neurons (one for each stroke class). The `Softmax` mathematical function takes the raw outputs and converts them into a probability distribution that sums to 1.0 (e.g., 90% Smash, 5% Clear, 5% Drop).

---

## 5. Summary Cheat Sheet for Q&A

*   **Q: Why use RSN instead of pre-trained Imagenet models?**
    *   *A:* RSN is highly efficient due to Channel Shuffling and depthwise convolutions, allowing us to hit real-time latency targets without needing massive server GPUs.
*   **Q: Why use a TCN instead of an LSTM for the visual stream?**
    *   *A:* LSTMs process data sequentially, which can be slow, and they suffer from vanishing gradients over long sequences. TCNs process sequences in parallel making them faster, and dilated causal convolutions give them a massive, stable memory (receptive field) of the past 31 frames.
*   **Q: What exactly does the Attention mechanism do in this context?**
    *   *A:* It acts as a temporal spotlight. It learns to ignore the idle frames and mathematically magnifies the features of the frames containing the actual stroke execution and contact point.
*   **Q: Why separate the Visual and Pose pipelines until the very end (Late Fusion)?**
    *   *A:* Visual features (textures, objects) and geometric pose features (math coordinates) have very different statistical distributions. Late fusion allows each specialized network branch to optimize its own representations before merging them for the final decision.
