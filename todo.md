- Sanity test pairwise batch statistics (e.g. sample-wise energy/variance, alignment)
- Single frame dataloader for contrastive learning
    - simclr, byol, dino

arches/setups
- try out RNN/LSTM baseline on entire IDMT dataset. Vacuum tube sim is too easy
- after:
    - wavenet (neural amp modeler baseline)
        - If things go well, use DP recurrent formulation for inference
    - SSMs
        - should theoretically match wavenets
    - flow matching
        - very steerable


