---
title: "Face Embedding Conditioned Transformer For 3D Face Reconstruction"
date: "2025-01-26"
ShowToc: true
TocOpen: false
tags: ["transformer", "3D", "paper", "attention", "face embedding", "morphable model"]
---


Here I will describe one of my recent experiments with 3D face reconstruction, based on this awesome [Disney paper](https://assets.studios.disneyresearch.com/app/uploads/2022/03/Shape-Transformers.pdf). Using paper's transformer decoder architecture, with cross-covariance attention [(XCiT)](https://github.com/facebookresearch/xcit?tab=readme-ov-file) conditioned and modulated with face embeddings, I was able to improve the quality of reconstructed faces. Since I was lacking a sufficiently large 3D head shape dataset, I decided to generate my own dataset using [HRN](https://github.com/youngLBW/HRN) which scores among the highest on 3D Face Reconstruction [benchmark](https://paperswithcode.com/sota/3d-face-reconstruction-on-realy).

### Shape Transformers and XCiT Blocks

In their paper, Disney Research proposed a new approach to modeling 3D shapes using a transformer-based architecture. At the core of their work is the Shape Transformer, which includes:

- A transformer-based model that is topology-independent, since the encoder’s input consists of learned tokens for each point sampled from the mesh surface.
- Cross-Covariance Attention (XCiT) blocks are used for learning non-linear spatial correlations.

XCiT blocks differ from traditional self-attention mechanisms by operating on the cross-covariance of token features, reducing computational overhead. This makes them ideal for 3D tasks, where input resolution can become a bottleneck. In this particular case, ~5k points are sampled from the mesh surface, and each one is translated into a token of dimension 512 (same as the face embedding) — so using vanilla self-attention mechanisms would be sort of infeasible.

{{< figure src="/posts/shape-former/architecture.png" caption="Figure 1: Shape Transformers pipeline" class=custom-caption >}}

In Shape Transformer, the encoder model gets two 3D inputs: one, points sampled from a canonical shape (averaged head, for example), and the other, points from a target shape. Both inputs are encoded into latent space and passed through encoder blocks to generate the shape code.

The decoder’s input consists of the shape code and points sampled from the canonical shape. The decoder’s job is to predict offsets for canonical points.

Both the encoder and decoder include several consecutive XCiT blocks.

### A Hierarchical Representation Network (HRN)

HRN is one of the top-performing methods for 3D face reconstruction. It achieves near state-of-the-art results for single-photo face reconstruction. The key idea behind HRN is to reconstruct the face shape in a coarse-to-fine manner. The first stage includes morphable model prediction (identity and expression) based on the given photo. The second stage refines low-frequency details by predicting a deformation map (per-vertex offsets). The final stage refines high-frequency details (wrinkles, etc.) by predicting a displacement map (similar to a normal map).

{{< figure src="/posts/shape-former/teaser.jpg" caption="Figure 2: HRN pipeline" class=custom-caption >}}

I used HRN to generate approximately ~10k diverse 3D face shapes. Unfortunately, that means face expressions are baked into the geometry, so the transformer also learns those. But, in general, if the transformer learns to approximate HRN’s output well, I get a near-SOTA solution that can be run in a single inference step (compared to HRN, whose 3 stages take approximately 9 seconds on my RTX 4090).

### Face embeddings coming into play

To adapt Disney’s Shape Transformer architecture for 3D face reconstruction from a single photo, I made several modifications to their pipeline.

1. *Remove the encoder part*

Since I was working with face embeddings (which is much simpler than training my own face encoder), I dropped the encoder entirely. Instead, the face embedding is fed directly into the decoder to condition the reconstruction process.

2. *Conditioning with Face Embeddings*

Inside the decoder, face embeddings are used in a couple of different ways. One, they’re used for modulation between attention blocks. And also, they serve as conditioning inputs in XCiT blocks.

3. *Dynamic canonical shapes*

Instead of using a fixed canonical shape, I used a morphable model trained to reconstruct face shapes from Dlib face encodings.

4. *Augmentations*

The more data, the better, so I augmented the dataset by interpolating between pairs of samples (face embedding and geometry) to create new, synthetic data points. Also, random noise was added to face embeddings during training. Both of these helped a lot with overfitting.

{{< figure src="/posts/shape-former/interpolation.png" caption="Figure 3: Data sugmentation via linear interpolation" class=custom-caption >}}

5. *Cross Attention and Self attention*

Alongside vanilla XCiT blocks, I added XCiT blocks conditioned with face embeddings (in a manner similar to cross attention), to enrich latent representations with face information and to capture intra-token relationships. I noticed that due to this modification, the quality improvement is minimal, but the model converges significantly faster.

{{< figure src="/posts/shape-former/xcit.png" caption="Figure 4: XCiT block" class=custom-caption >}}


{{< figure src="/posts/shape-former/xattention.png" caption="Figure 5: XCiT block conditioned on face embedding" class=custom-caption >}}


### Results

Below is a comparison of the reconstruction results, with entries as follows:

1. Output of the morphable model that’s used as a canonical shape generator for the transformer.
2. Output of the transformer
3. HRN prediction
4. The original photo (not used during model training or testing)

{{< figure src="/posts/shape-former/comparison.png" caption="Figure 6: Morphable model vs Transformer vs HRN prediction" class=custom-caption >}}

And here’s another visualization, where I tried to show the L2 loss between face shape predictions and HRN, averaged across ~60 photos and rendered over an average head shape. Red regions represent higher reconstruction errors. Some of the red areas for morphable model are expected, like neck region for example. But the inner face regions are the ones that are more important – and there I see the real improvement.

{{< figure src="/posts/shape-former/loss.png" caption="Figure 7: L2 loss for transformer output (left) and morphable model output (right)" class=custom-caption >}}


### Conclusion

This experiment shows how Disney’s Shape Transformer architecture could be adapted for single-photo 3D face reconstruction. In my particular case, this model is able to approximate HRN’s high-quality outputs, improving upon a simpler morphable model. The good thing is, the whole thing runs in one inference step and usually produces quite a decent result.
