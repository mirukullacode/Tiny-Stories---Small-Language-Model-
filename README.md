## Small Language Model - Built from Scratch

A GPT style transformer language model trained from scratch on the TinyStories dataset demonstratinng deep understanding of modern LLM architecure, training techniques and optimization strategies. 

### Overview

Thhis projject implements a decodeer-only transformer archhitecture (similar to GPT-2) trained to generate coherent children's stories. Built entirely from scratch using PyTorch, this demonstrates:
 - Deep understanding of transformer archhitecure
 - Practical experience with trainning optimizations
 - Knowledge of modern NLP optimizations
 - Ability to implement research papers in production code.
** Key Metrics: **
- Model Size - 24M Parameters
- Training Timme: 4.5 hours on T4 GPU
- Final Training Loss: 2.39
- Final Validation Loss: 2.39
- Context Window: 128 tokens

## Architecture
Model Configuration
- Vocabulary Size: 50,257 (GPT-2 tokenizer)
- Embedding Dimension: 384
- Number of Layers: 6
- Attention Heads: 6
- Context Length: 128 tokens
- Dropout: 0.1
- Total Parameters: ~24 million

### Core Components

**1. Multi-Head Self-Attention**
- Implements scaled dot-product attention
- Uses Flash Attention when available (GPU optimization)
- Casual maskking for autoregressive generation
- Attention dropout for regularization

**Key Features:** 

- Q, K, V projections with linear layers
- Efficient attention computation (O(n²d) complexity)
- Causal masking prevents looking ahead
- Residual connections for gradient flow

 **2. Position-wise Feed-Forward Network:**
 - 4X expansion ratio (384 -> 1536 -> 384)
 - GELU activation function
 - Drpout for regularization
 - Residual connections

**3. Layer Normalization:**
- Pre-norm architecture (LayerNorm before attention/MLP)
- Improves training stability
- Custom implementation with learnable scale and bias

**4. Positional Embeddings:**
- Learned absolute positional encodings
- Added to token embeddings
- Allows model to understand sequence order

**5. Weight Tying:**
- Shared weights between token embeddings and output layer
- Reduces parameter count by ~ 13M parameters
- Improves generalization

## Dataset - TinyStories Dataset (roneneldan/TinyStories)

- Synthetic dataset of simple children's stories
- Generated using GPT-3.5 and GPT-4
- Vocabulary limited to words a 3-4 year old would understand
- Perfect for training small language models

###Data Processing Pipeline

**1. Tokenization**

- Used GPT-2 BPE tokenizer (tiktoken)
- Vocabulary size: 50,257 tokens
- Efficient subword tokenization


**2. Binary Storage**

- Converted to binary format for efficient loading
- Used memory-mapped arrays (np.memmap)
- Prevents RAM overflow with large datasets
- **dtype:** uint16 (sufficient for vocab size < 65,536)


**3. Batch Generation**

- Dynamic batch sampling
- Context window: 128 tokens
- Batch size: 32
- Efficient GPU utilization

## Training Pipeline 
Optimization Strategy
**Learning Rate Schedule**
- Initial LR: 1e-4
- Warmup Steps: 1,000 (linear warmup)
- Decay Schedule: Cosine annealing
- Min LR: 5e-4
- Total Steps: 20,000

**Rationale:** Warmup prevents instability in early training, cosine decay smoothly reduces learning rate fro fine-grained optimization. 

AdamW Optimizer
- Beta1: 0.9
- Beta2: 0.95 (reduced from default 0.999)
- Weight Decay: 0.1 (L2 regularization)
- Epsilon: 1e-9
  
Why AdamW? Decouples weight decay from gradient updates, leading to better generalization than standard Adam.

Gradient Accumulation
- Accumulation Steps: 32
- Effective Batch Size: 32 × 32 = 1,024
- Gradient Clipping: max_norm=0.5

Benefit: Simulates large batch training on limited GPU memory, stabilizes training, prevents gradient explosion.

Mixed Precision Training
- Precision: bfloat16 (if supported), else float16
- GradScaler for automatic loss scaling
- 2x faster training, 50% less memory


### Training Configuration
Hyperparameters:
├── Batch Size: 32
├── Context Length: 128
├── Gradient Accumulation: 32 steps
├── Effective Batch: 1,024 samples
├── Max Iterations: 20,000
├── Evaluation Interval: 500 steps
├── Dropout: 0.1
└── Device: CUDA (T4 GPU)

Loss Curve Analysis
The training achieved excellent convergence:

Smooth descent: No significant overfitting
Final loss ~2.39: Strong performance for this dataset size
Train-val gap: Minimal, indicating good generalization

## Key Implementation Details
1. Flash Attention Integration
 ```
 if hasattr(F, "scaled_dot_product_attention"):
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
 ```
- 2-4x faster than manual atttention
- Memory efficient (O(n) insttead of O(n^2))
- Automatically fused operations

2. Efficient Data Loading
```
data = np.memmap('train.bin', dtype=np.uint16, mode='r')
ix = torch.randint(len(data) - block_size, (batch_size,))
```

- Memory-mapped files prevent RAM overflow
- Random sampling for better generalization
- Pin memory for async GPU transfer

3. Weight Initialization
- Linear layers: Normal(0, 0.02)
- Embeddings: Normal(0, 0.02)
- Output projection: Normal(0, 0.02/sqrt(2*n_layers))
GPT-2 initialization scheme for stable training at depth.


4. Model Checkpoint
```
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best_model_params.pt")
```
Saves only the best model based on validation loss.


### Quantitative Results
Training Duration: 3 hours 18 minutes
Final Training Loss: 2.3979
Final Validation Loss: 2.3949
Loss Reduction: 75% (from ~9.5 to ~2.4)
Convergence: Achieved by iteration 15,000



Qualitative Results
Prompt: "Once upon a time there was a pumpkin."
Generated Story: 
``` Once upon a time there was a pumpkin. He was so excited to go outside. 
He had brave and would explore. His parents warned him about it. 
He wasn't so excited and waved.

They grabbed the brick, he started running and laughed. He began to rest, 
until the Linda was something gloomy pond. She had never decided to go 
down and try "I want you, now! It's so soft!"

The girl was so excited and ready to go back home. From that day on, 
she always welcomed his shop and enjoyed carefully walking to leave 
the cool adventure...
```
### Analysis
Strengths:

Coherent sentence structure
Appropriate vocabulary for children's stories
Maintains narrative flow
Proper grammar and punctuation


Weaknesses:

Some pronoun inconsistencies
Occasional semantic drift
Limited long-range coherence (expected for 128 context)


## Research Insights
**Papers & Concepts Applied**

**1. "Attention Is All You Need" (Vaswani et al., 2017)**

- Implemented multi-head self-attention
- Scaled dot-product attention mechanism
- Position-wise feed-forward networks

**2. "Language Models are Unsupervised Multitask Learners" (GPT-2)**

- Decoder-only transformer architecture
- Weight tying between embeddings and output
- Pre-norm layer normalization

**3. "Decoupled Weight Decay Regularization" (AdamW)**

- Proper weight decay implementation
- Better generalization than Adam
- Separate learning rate and weight decay

**4. "FlashAttention: Fast and Memory-Efficient Exact Attention"**

- IO-aware attention implementation
- Memory optimization for long sequences
- Significant speedup on modern GPUs

**5. "TinyStories: How Small Can Language Models Be?"**

- Demonstrated that small models can be surprisingly capable
- Quality of data > quantity for learning
- Synthetic datasets can be highly effective

### Key Learnings from Implementation

**Learning Rate Warmup is Critical**
**Without warmup:** Training unstable, high initial loss
**With warmup:** Smooth convergence from start


**Gradient Accumulation Enables Large Batches**
- Simulates batch size of 1,024 on limited GPU
- More stable gradients
- Better generalization


**Mixed Precision = 2x Speedup**
- Essential for efficient training
- Minimal accuracy loss
- Requires GradScaler for stability


**Data Storage Matters**
- Memory-mapped arrays prevent OOM errors
- Binary format faster than text
- Preprocessing saves training time


## Technical Learnings
**What I Learned**

**1. Transformer Architecture Internals**
- How attention mechanisms actually work
- Importance of residual connections
- Layer normalization placement affects training


**2. Training Optimization**
- Learning rate schedules impact convergence
- Gradient clipping prevents exploding gradients
- Early stopping prevents overfitting


**3. PyTorch Best Practices**
- Proper use of .to() and device management
- Memory-efficient data loading
- Automatic mixed precision training


**4. Model Evaluation**
- Perplexity as a metric
- Difference between training and validation loss
- Importance of monitoring both metrics


**5. Debugging Deep Learning**
- Checking gradient flow
- Monitoring loss curves
- Validating attention patterns


## References

- Vaswani et al. (2017) - "Attention Is All You Need"
- Radford et al. (2019) - "Language Models are Unsupervised Multitask Learners"
- Loshchilov & Hutter (2019) - "Decoupled Weight Decay Regularization"
- Dao et al. (2022) - "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Eldan & Li (2023) - "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"
- Karpathy's nanoGPT - Code reference and best practices

## Author 
**Manjunath Irukulla**
Email : imanjunaad@gmail.com
LinkedIn: https://www.linkedin.com/in/irumanjunath/

##Acknowledgments
- Andrej Karpathy for nanoGPT inspiration
- Hugging Face for the TinyStories dataset
- PyTorch team for excellent documentation
- OpenAI for GPT-2 tokenizer

