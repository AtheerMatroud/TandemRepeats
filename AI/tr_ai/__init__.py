"""Learned reconstruction of the ancestor of two tandem repeat sequences.

simdata    simulated families (A; D1, D2) from the generator of the MPE manuscript
fastalign  fast symmetric alignment of D1 and D2, column labels, symbol edit distance
data       tensors, batching and augmentation for the three candidate models
models     Transformer, BiLSTM sequence-to-sequence, alignment-guided tagger
train      training loop (resumable, mixed precision)
infer      reconstruction of the ancestor by a trained model
evaluate   accuracy measures of the MPE manuscript, baseline pipeline, distance estimates
"""
