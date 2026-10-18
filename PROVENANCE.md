# Provenance and scope

This repository packages a NUS CEG5304 coursework submission. The course supplied the task scaffolding, portions of data loading/training utilities, and model diagrams. Student work includes RNN/LSTM internals, classifier and generation forward passes, training updates, sampling, and analysis utilities.

This is coursework, not a production NLP service. No ownership or open-source licence is claimed over the course material or third-party datasets/code. The repository is initially private. Course assignment questions, submission instructions, matriculation identifiers, and local virtual environments are omitted.

Included weights and notebook outputs are historical artifacts. The saved classifier is the last epoch, whereas 81.88% is the best development accuracy observed at epoch 3; do not label that number as the saved checkpoint's measured score. LSTM weights exist, but a verified RNN-vs-LSTM comparison is not supplied. Text generation samples have limited coherence.

SST-2 is loaded through TorchText; the story Parquet file is the small corpus supplied with the coursework. Its upstream dataset provenance has not been independently identified. `tokenizer/word_level.json` is a WordLevel tokenizer, not SentencePiece.
