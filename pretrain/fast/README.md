---
library_name: transformers
license: apache-2.0
tags:
- robotics
- tokenizer
---

# FAST: Efficient Action Tokenization for Vision-Language-Action Models

This is the official repo for the [FAST action tokenizer](https://www.pi.website/research/fast).

The action tokenizer maps any sequence of robot actions into a sequence of dense, discrete **action tokens** for training autoregressive VLA models.

Here, we provide:
1. FAST+, our *universal* action tokenizer, trained on 1M real robot action sequences.
2. Code for quickly training *new* action tokenizers on your custom dataset.

## Installation

FAST can be used as a convenient HuggingFace AutoProcessor. To use it, simply install the `transformers` package (and `scipy` for the underlying DCT algorithm).

```
pip install transformers scipy
```

## Using the Universal Action Tokenizer

We recommend applying the tokenizer to 1-second action "chunks" that have been pre-normalized to a range of [-1...1] 
(we use quantile normalization for this step -- check our paper). Encoding and decoding support batched inference.

```
import numpy as np
from transformers import AutoProcessor

# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Tokenize & decode action chunks (we use dummy data here)
action_data = np.random.rand(256, 50, 14)    # one batch of action chunks
tokens = tokenizer(action_data)              # tokens = list[int]
decoded_actions = tokenizer.decode(tokens)
```

**Note**: During decoding, the tokenizer needs to map the decoded sequence of actions back into a `[time_horizon, action_dim]` matrix. 
There are multiple ways to provide the necessary dimensions to the tokenizer: (1) they automatically get saved on the first `forward()` call, (2) you can set them manually as arguments to the `decode()` call


## Training a new Action Tokenizer on Your Own Data

In our experiments, we found the FAST+ universal tokenizer to work well across a wide range of robot setups, action dimensions, and control frequencies.
If you, however, want to train a custom FAST tokenizer for your dataset at hand, it is very easy using the `.fit()` convenience function we provide.
When called on a dataset of action chunks (of the same or different lengths), it returns a new tokenizer instance, which you can save and optionally push 
to the HuggingFace hub. Training should typically only take a few seconds to minutes.

```
# First, we download the tokenizer from the Hugging Face model hub
# Here, we will not use the pre-trained tokenizer weights, but only the source code
# to train a new tokenizer on our own data.
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Load your action data for tokenizer training
# Chunks do not need to be of the same length, we will use dummy data
action_data = np.random.rand(4000, 50, 14)

# Train the new tokenizer, depending on your dataset size this can take a few minutes
tokenizer = tokenizer.fit(action_data)

# Save the new tokenizer, optionally push it to the Hugging Face model hub
tokenizer.save_pretrained("<your_local_path>")
tokenizer.push_to_hub("YourUsername/my_new_tokenizer")
```