# SeqInsight · Sequence Models for Sentiment and Generation

NUS CEG5304 coursework portfolio: hand-written RNN/LSTM blocks, SST-2 sentiment classification, embedding visualisation, gradient saliency, and small-corpus autoregressive text generation.

## What is included

- `seqinsight.ipynb`: complete implementation cells with historical outputs. Course submission instructions and question text have been removed.
- `models/`: retained RNN/LSTM classifier and generator state dictionaries.
- `corpuses/short.parquet`: small story corpus used in the notebook.
- `tokenizer/word_level.json`: the 3,072-token generation vocabulary.
- `tokenizer/sst2_vocab.json`: the classifier's token-to-ID mapping, reconstructed and checked against all 67,349 retained encoded records.
- `figures/`, `generated_text/`: historical training curve and generation examples.
- `scripts/smoke_check.py`: both backends' forward/backward checks, retained-checkpoint loading checks and corpus/tokenizer checks.

## Setup

Use Python 3.8 for compatibility with the original TorchText/PyTorch pair.

```bash
conda env create -f environment.yml
conda activate seqinsight
python scripts/smoke_check.py
jupyter lab seqinsight.ipynb
```

Use retained weights directly without downloading SST-2 or running training cells:

```bash
python scripts/infer.py --task sentiment --backend rnn --text "This movie was wonderful."
python scripts/infer.py --task generate --backend lstm --text "Once upon a time" --length 40
```

Run the notebook with the repository root as its working directory. The sentiment section downloads SST-2 using TorchText and therefore requires internet access on first use. Training cells overwrite the corresponding saved model files; make a copy first if you want to preserve the supplied historical weights. The generation section uses the bundled Parquet corpus.

The requirements use standard PyTorch packages. For CUDA, install a compatible official PyTorch 2.2.0 build while keeping TorchText 0.17.0 and TorchData 0.7.1 aligned. A newer PyTorch version may break this legacy TorchText pipeline.

## Evidence and limits

| Experiment | Retained evidence |
|---|---|
| RNN sentiment classification | 67,349 training records; best development accuracy **81.88%** at epoch 3 of the retained five-epoch run |
| RNN generation | 583 story segments; WordLevel vocabulary 3,072; training loss **8.2744 → 6.1275** over 50 epochs |

These are historical results, not new reproductions. The saved classifier corresponds to the last training epoch, not automatically the best validation epoch. No verified LSTM-vs-RNN benchmark is claimed. Training loss does not establish generation quality; the supplied generated text is frequently incoherent.

The notebook implements masked mean pooling and raw-gradient saliency. It does not implement a bidirectional LSTM, attention, Integrated Gradients, top-k/top-p decoding, or validation-based early stopping. In the generation loop, hidden states are retained across shuffled batches; reset them for independent samples if extending the project.

## Verification performed during packaging

Both RNN and LSTM classifier/generator definitions passed forward and backward checks. All four retained checkpoints passed architecture loading and inference checks. The bundled Parquet file and tokenizer were readable. Full dataset training and metric reproduction were not rerun.

See [PROVENANCE.md](PROVENANCE.md) for the distinction between course scaffolding and completed implementation. This repository was initially created **private**; no blanket licence over course materials or datasets is asserted.
