# Models

This directory contains the neural network components for the Conditional Graph Variational Autoencoder (CGVAE).

## Graph Encoder

The GraphEncoder is a Graph Neural Network (GNN) that maps molecular graphs to a continuous latent space, producing the parameters of a latent distribution for the VAE.

#### Architecture Overview:
- **Base Architecture**: Graph Attention Networks (GAT) with multi-head attention
- **Layers**: 4 stacked GAT layers with residual connections
- **Attention Heads**: 4 heads per layer for multi-perspective aggregation
- **Regularization**: Batch normalization, dropout, and residual connections

#### How It Works:

1. **Input**: Receives a molecular graph with:
   - Node features $x \in \mathbb{R}^{N \times d_{\text{node}}}$ (atom properties: element, degree, valence, hybridization, aromaticity, etc.)
   - Edge indices (bond connectivity)
   - Edge attributes $e \in \mathbb{R}^{E \times d_{\text{edge}}}$ (bond type, conjugation, ring membership)

2. **Embedding Layers**:
   - Node embedding: $h^{(0)} = \text{ReLU}(W_{\text{node}} x)$ projects atoms to hidden dimension
   - Edge embedding: $e' = W_{\text{edge}} e$ projects bond features to hidden dimension

3. **Message Passing** (4 iterations):
   - For each layer $\ell$:
     - **Attention**: $h^{(\ell)} = \text{GAT}(h^{(\ell-1)}, \text{edge\_index}, e')$ computes attention-weighted messages
     - **Normalization**: $h^{(\ell)} = \text{BatchNorm}(h^{(\ell)})$
     - **Residual**: $h^{(\ell)} = h^{(\ell)} + h^{(\ell-1)}$ (skip connections for gradient flow)
     - **Activation**: $h^{(\ell)} = \text{ReLU}(\text{Dropout}(h^{(\ell)}))$

4. **Graph-Level Pooling**:
   - Multi-strategy pooling combines three approaches:
     - $h_{\text{mean}} = \frac{1}{N}\sum_{i=1}^N h_i$ (mean aggregation)
     - $h_{\text{max}} = \max_{i=1}^N h_i$ (max aggregation)
     - $h_{\text{add}} = \sum_{i=1}^N h_i$ (sum aggregation)
   - Concatenate: $h_{\text{graph}} = [h_{\text{mean}} \,||\, h_{\text{max}} \,||\, h_{\text{add}}] \in \mathbb{R}^{3d_{\text{hidden}}}$

5. **Latent Projection**:
   - Pre-processing: $h' = \text{ReLU}(\text{Dropout}(W_{\text{pre}} h_{\text{graph}}))$
   - Mean: $\mu = W_\mu h'$ 
   - Log-variance: $\log\sigma^2 = W_{\log\sigma} h'$
   - Output: $(\mu, \log\sigma^2) \in \mathbb{R}^{d_{\text{latent}}}$ parameterizes the latent distribution $q(z|G)$

6. **Sampling**: The VAE samples $z \sim \mathcal{N}(\mu, \sigma^2)$ using the reparameterization trick: $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$

#### Why These Design Choices?

- **GAT over GCN**: Attention mechanism learns which neighbors are important for each atom
- **Multi-head attention**: Captures different chemical relationships simultaneously
- **Residual connections**: Prevents gradient vanishing in deep networks
- **Multi-pooling**: Captures different graph-level statistics (robust representation)
- **Batch normalization**: Stabilizes training and improves convergence

## Property Predictor
The PropertyPredictor is a simple multi-layer perceptron (MLP) that takes the latent vector $z$ (output by the encoder) and maps it to our target properties -- in this case, BBB permeability.

#### How It Works:
1. **Input**: recieves the latent code $z$ from the VAE encoder, one vector per graph: $[\mu,\sigma]$ but typically we use the sampled latent $z$. 
2. **Hidden Layers**: First linear layer expands from `latent_dim` to h`idden_dim`, followed by ReLU and Dropout for regularization. Second linear layer reduces to `hidden_dim//2`, again with ReLU and Dropout.
3. **Output Later**: Final linear layer maps to `num_properties` outputs (1 in our case: BBBP)
4. **Training Objective**: During VAE training, the *Property Prediction Loss* is $L_\text{prop} = \text{MSE}(\hat{y}, y)$ where $\hat{y}$ is the predictor output and $y$ is the true label.
5. **Integration with MGCVAE**: In the forward of the full MGCVAE, after sampling $z$, we call `y_pred = self.property_predictor(z)`; the loss term $\gamma L_\text{prop} encourages the latent space to capture information predictive of these properties.

This simple MLP on top of the learned latent space lets the VAE jointly learn to reconstruct molecular structure and to encode property-relevant features in $z$, enabling us to condition molecule generation on desired BBBP values.

## Graph Decoder
The GraphDecoder is the most complex component—it takes a latent vector $z$ plus target property conditions and reconstructs a molecular graph.

#### How It Works:
1. **Input Conditioning**: Takes latent vector $z$ from encoder. Concatenates with target properties (e.g., \[$BBBP=0.67$\]). Projects to hidden representation that "knows" what kind of molecule to generate.
2. **Graph Size Prediction**: Predicts how many atoms the molecule should have. Outputs probability distribution over possible sizes (1 to `max_nodes`). Smaller molecules might be more likely for certain property targets.
3. **Node Generation**: For each possible node position, predicts atom type probabilities. Uses one-hot encoding matching your dataset's node features. E.g., `[H, C, N, O, F, ...]` probabilities for each position.
4. **Edge Generation**: For every pair of nodes (i,j), predicts: Existence probability: Should there be a bond between atoms $i$ and $j$? Bond type: if bond exists, what type? (single, double, triple, aromatic).Uses node features + positional context to make this decision.
5. **Positional Awareness**: Adds learned positional embeddings to help with molecular topology. Different positions in the generation order can have different "roles."

#### Key Considerations
- **Autoregressive vs. One-Shot:** This is a "one-shot" decoder that generates all nodes/edges simultaneously, which is faster but potentially less flexible than autoregressive approaches.

- **Conditioning Strategy:** Properties are injected at the input level and influence all generation decisions through the global context vector.

- **Differentiable Sampling:** Uses Gumbel-Softmax and sigmoid probabilities to maintain differentiability during training while still producing discrete-like outputs.

The decoder learns to map from the property-conditioned latent space to valid molecular structures that satisfy the target BBBP and toxicity constraints!