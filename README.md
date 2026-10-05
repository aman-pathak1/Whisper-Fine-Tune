# Whisper Fine-Tuning

A speech-to-text project that fine-tunes OpenAI Whisper on a custom dataset using Hugging Face Transformers and a Kaggle T4 GPU.

## Overview

The project fine-tunes a pre-trained Whisper model for Automatic Speech Recognition (ASR).

```text
Hugging Face Dataset
        |
        v
Audio Preprocessing
        |
        v
Whisper Processor
        |
        v
Whisper Fine-Tuning
        |
        v
Fine-Tuned Model
        |
        v
Evaluation
        |
        +---- WER: 4.8%
        |
        +---- CER: 3.4%
```

## Features

- Whisper fine-tuning
- Dynamic dataset loading from Hugging Face
- Audio preprocessing at 16 kHz
- Kaggle T4 GPU training
- WER and CER evaluation
- Speech-to-text inference
- Hugging Face compatible model

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- Whisper
- Accelerate
- JiWER

## Project Structure

```text
Whisper-Fine-Tune/
├── train/
│   └── fine-tune_on_hf_dataset.py
├── inference/
│   └── eval.py
├── requirements.txt
└── README.md
```

## Dataset

The dataset is loaded dynamically from Hugging Face.

```python
from datasets import load_dataset

dataset = load_dataset(
    dataset_name,
    config_name,
    split="train"
)
```

The project does not use `load_from_disk()`.

Audio is resampled to 16 kHz before being processed by Whisper.

## Training

The model is trained on a Kaggle T4 GPU.

Run:

```bash
python train/fine-tune_on_hf_dataset.py
```

Check GPU availability:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Evaluation

The fine-tuned model is evaluated using Word Error Rate (WER) and Character Error Rate (CER).

```bash
python inference/eval.py \
    --hf_model <whisper-small> \
    --language hi \
    --config hi \
    --split test
```

## Results

| Metric | Score |
|---|---:|
| WER | 4.8% |
| CER | 3.4% |

Lower WER and CER indicate better transcription performance.

## Inference

The fine-tuned model can be used with the Transformers ASR pipeline:

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="<whisper-small",
    device=0
)

result = pipe("audio.wav")
print(result["text"])
```

## Hugging Face Authentication

For private datasets or models:

```bash
huggingface-cli login
```

For Kaggle, the Hugging Face token should be stored using Kaggle Secrets.

Do not hard-code tokens in the source code.

## Future Improvements

- Real-time speech recognition
- FastAPI deployment
- Web-based interface
- Larger training datasets
- Further hyperparameter optimization

## Author

**Aman Pathak**

GitHub: https://github.com/aman-pathak1

## Acknowledgements

- OpenAI Whisper
- Hugging Face
- Kaggle
