# Contributing to streamSAM

Contributions should preserve the bounded-memory and ordered-state contracts
that define streamSAM.

## Pull requests

1. Fork the repo and create your branch from `main`.
2. Add contract checks for changes to queues, ownership, ordering or state.
3. Add mask-fidelity evidence for changes that affect model execution.
4. Update the relevant documentation when ownership or synchronization changes.
5. Run the relevant checks and include the exact commands and results in the PR.

## Issues

Include the model config, checkpoint, PyTorch and CUDA versions, GPU, input
shape, queue capacities and the smallest reproducible command. State whether the
problem also occurs on the upstream EdgeTAM checkout.

## License

Contributions are licensed under the [Apache 2.0 license](LICENSE). The original
EdgeTAM and SAM 2 attribution remains intact.
