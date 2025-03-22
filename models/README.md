# SALMONN Models

This directory contains the implementation of various models used in the SALMONN (Speech Audio Language Music Open Neural Network) project, including Whisper, and BEATs, Qformer, LLaMA

## 1. Whisper Architecture

Whisper is a speech processing model used in SALMONN for encoding speech inputs. It's a Transformer-based encoder-decoder model originally designed for automatic speech recognition (ASR) and translation, but in SALMONN, primarily the encoder component is utilized to extract rich representations from speech.

### 1.1 Class Diagram

#### Class Inheritance Hierarchy
```
PreTrainedModel
│
└──> WhisperPreTrainedModel
     │
     └──> WhisperModel
          │
          └──> WhisperForConditionalGeneration
```

#### Component Composition Structure
```
WhisperModel
│
└──> WhisperEncoder
     │
     ├──> WhisperEncoderLayer (multiple layers)
     │    │
     │    ├──> WhisperAttention
     │    │    │
     │    │    ├──> Linear projections (q_proj, k_proj, v_proj)
     │    │    └──> Linear projection (out_proj)
     │    │
     │    ├──> LayerNorm (self_attn_layer_norm)
     │    │
     │    ├──> Feed-Forward Network
     │    │    │
     │    │    ├──> Linear (fc1)
     │    │    ├──> Activation function (GELU)
     │    │    └──> Linear (fc2)
     │    │
     │    └──> LayerNorm (final_layer_norm)
     │
     └──> LayerNorm (layer_norm)
```

## 2. BEATs Architecture

BEATs (Bidirectional Encoder representation from Audio Transformers) is an audio processing model used in SALMONN for encoding general audio inputs. It's a transformer-based model designed to extract rich representations from audio spectrograms, complementing Whisper's speech-focused features.

### 2.1 Class Diagram

#### Class Inheritance Hierarchy
```
nn.Module
│
└──> BEATs
```

#### Component Composition Structure
```
BEATs
│
├──> nn.Conv2d (patch_embedding)
│
├──> nn.Linear (post_extract_proj)
│
├──> LayerNorm (layer_norm)
│
└──> TransformerEncoder
     │
     └──> ModuleList of TransformerEncoderLayer
          │
          ├──> MultiheadAttention
          │    │
          │    └──> Linear projections (q_proj, k_proj, v_proj, out_proj)
          │
          ├──> LayerNorm (self_attn_layer_norm)
          │
          ├──> FeedForward
          │    │
          │    ├──> Linear (fc1)
          │    ├──> Activation function (GELU)
          │    ├──> Dropout
          │    └──> Linear (fc2)
          │
          └──> LayerNorm (final_layer_norm)
```
### 2.2 Use in SALMONN

In the SALMONN architecture, BEATs serves as a secondary audio encoder that:
1. Processes raw audio waveforms into audio features that complement Whisper's speech features
2. Provides additional context for non-speech audio elements like music, environmental sounds, and acoustic events

These features are then combined with Whisper features before being processed by the Qformer, enhancing SALMONN's ability to understand the full audio context.

In SALMONN, BEATs processes audio inputs in parallel with Whisper:
```python
beats_embeds = self.audio_encoder(waveform, return_dict=True)[0]
```

The BEATs features can then be combined with Whisper features to provide a more comprehensive audio understanding:
```python
if self.use_audio_Qformer:
    speech_embeds = torch.cat([speech_embeds, beats_embeds], dim=-1)
    speech_embeds = self.speech_beats_proj(speech_embeds)
```

This multi-encoder approach allows SALMONN to leverage both speech-specific features from Whisper and general audio features from BEATs, creating a more robust representation of the audio input.


## 3. Qformer Architecture

The Qformer (Query Transformer) is a key component in SALMONN that serves as a modality adapter between speech/audio inputs and the language model. It's based on a modified BERT architecture and is designed to efficiently extract and transform features from one modality to be compatible with another.

Qformer uses learnable query tokens that interact with input features through cross-attention mechanisms. These query tokens act as information bottlenecks that extract the most relevant information from the input modality. The number of query tokens is typically much smaller than the input sequence length, allowing for efficient information extraction and dimensionality reduction.

### 3.1 Class Diagram

#### Class Inheritance Hierarchy
```
PreTrainedModel
│
└──> BertPreTrainedModel
     ├──> BertModel
     ├──> BertLMHeadModel (Qformer)
     └──> BertForMaskedLM
```

#### Component Composition Structure
```
Composition Structure (has-a relationships):

BertLMHeadModel
│
├──> BertModel
│    │
│    ├──> BertEmbeddings
│    │
│    ├──> BertEncoder
│    │    │
│    │    └──> a stack of BertLayer, each is a transformer block containing
│    │         │
│    │         ├──> BertAttention 
│    │         │    │
│    │         │    └──> BertSelfAttention (can be made into both self/cross-attention)
│    │         │
│    │         ├──> BertIntermediate
│    │         │
│    │         └──> BertOutput
│    │
│    └──> BertPooler
│
└──> ClassificationHead (cls)
```

### 3.2 Use in SALMONN

The Qformer architecture consists of:

- **Query Tokens**: Learnable parameters that serve as the interface between modalities
- **Self-Attention Layers**: Standard BERT-style self-attention for processing query tokens
- **Cross-Attention Layers**: Allow query tokens to attend to input features from another modality

The rest of BERT architecture is not used in Qformer.
- **Word/Position Embeddings**: Removed
- **Layer Outputs/Intermediates (FFN)**: Removed
- **BertPooler**: Not used
- **CLS head**: Removed

The Qformer can operate at different levels:
- **Window-level**: Processing fixed-length windows of speech/audio features
- **Sequence-level**: Processing the entire sequence at once

## 4. LLaMA Architecture

LLaMA (Large Language Model Meta AI) is the foundation language model used in SALMONN for text generation. It's a decoder-only transformer architecture optimized for efficient inference and strong language understanding capabilities.

### 4.1 Class Diagram

#### Class Inheritance Hierarchy
```
PreTrainedModel
│
└──> LlamaPreTrainedModel
     │
     └──> LlamaForCausalLM
```

#### Component Composition Structure
```
LlamaForCausalLM
│
├──> LlamaModel
│    │
│    ├──> nn.Embedding (embed_tokens)
│    │
│    ├──> nn.ModuleList of LlamaDecoderLayer
│    │    │
│    │    ├──> LlamaAttention
│    │    │    │
│    │    │    ├──> Linear projections (q_proj, k_proj, v_proj, o_proj)
│    │    │    └──> LlamaRotaryEmbedding (rotary_emb)
│    │    │
│    │    ├──> LlamaRMSNorm (input_layernorm)
│    │    │
│    │    ├──> LlamaMLP
│    │    │    │
│    │    │    ├──> Linear (gate_proj)
│    │    │    ├──> Linear (up_proj)
│    │    │    ├──> ACT2FN[hidden_act] activation
│    │    │    └──> Linear (down_proj)
│    │    │
│    │    └──> LlamaRMSNorm (post_attention_layernorm)
│    │
│    └──> LlamaRMSNorm (norm)
│
└──> nn.Linear (lm_head)
```

In SALMONN, LLaMA is integrated with speech features through a projection layer that maps Qformer outputs to the LLaMA embedding space. The model processes inputs in this sequence:

1. BOS token embeddings
2. Projected speech embeddings from Qformer
3. Text token embeddings (for training)
