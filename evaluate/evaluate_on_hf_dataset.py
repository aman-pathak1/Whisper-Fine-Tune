import os
import argparse
import evaluate
from tqdm import tqdm

from datasets import load_dataset, Audio
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

whisper_norm = BasicTextNormalizer()


def is_target_text_in_range(ref):
    if ref is None:
        return False

    ref = str(ref).strip()

    if ref == "ignore time segment in scoring":
        return False

    return ref != ""


def get_text(sample):
    possible_columns = [
        "text",
        "sentence",
        "normalized_text",
        "transcript",
        "transcription"
    ]

    for column in possible_columns:
        if column in sample:
            return sample[column]

    raise ValueError(
        f"No transcript column found. Expected one of: "
        f"{possible_columns}. Available columns: {list(sample.keys())}"
    )


def get_text_column_name(column_names):
    possible_columns = [
        "text",
        "sentence",
        "normalized_text",
        "transcript",
        "transcription"
    ]

    for column in possible_columns:
        if column in column_names:
            return column

    raise ValueError(
        f"No transcript column found. Available columns: {column_names}"
    )


def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch


def main(args):

    print("Loading Whisper model...")
    print(f"Model: {args.model}")

    whisper_asr = pipeline(
        task="automatic-speech-recognition",
        model=args.model,
        device=0,
        torch_dtype="auto"
    )

    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language=args.language,
            task="transcribe"
        )
    )

    print("\nLoading dataset from Hugging Face...")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")

    if args.config:
        dataset = load_dataset(
            args.dataset,
            args.config,
            split=args.split
        )
    else:
        dataset = load_dataset(
            args.dataset,
            split=args.split
        )

    print("\nDataset loaded successfully.")
    print(f"Number of samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    text_column_name = get_text_column_name(
        dataset.column_names
    )

    print(f"Transcript column: {text_column_name}")

    if "audio" not in dataset.column_names:
        raise ValueError(
            "Dataset must contain an 'audio' column."
        )

    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=16000)
    )

    dataset = dataset.map(
        normalise,
        num_proc=2
    )

    dataset = dataset.filter(
        is_target_text_in_range,
        input_columns=[text_column_name],
        num_proc=2
    )

    print(f"Samples after filtering: {len(dataset)}")

    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []

    print("\nStarting transcription...")

    for sample in tqdm(
        dataset,
        desc="Evaluation"
    ):

        audio = sample["audio"]

        output = whisper_asr(
            {
                "array": audio["array"],
                "sampling_rate": audio["sampling_rate"]
            }
        )

        prediction = output["text"]
        reference = get_text(sample)

        normalized_prediction = whisper_norm(
            prediction
        )

        normalized_reference = whisper_norm(
            reference
        )

        predictions.append(prediction)
        references.append(reference)

        normalized_predictions.append(
            normalized_prediction
        )

        normalized_references.append(
            normalized_reference
        )

    print("\nCalculating metrics...")

    wer = wer_metric.compute(
        references=references,
        predictions=predictions
    )

    cer = cer_metric.compute(
        references=references,
        predictions=predictions
    )

    normalized_wer = wer_metric.compute(
        references=normalized_references,
        predictions=normalized_predictions
    )

    normalized_cer = cer_metric.compute(
        references=normalized_references,
        predictions=normalized_predictions
    )

    wer = round(100 * wer, 2)
    cer = round(100 * cer, 2)
    normalized_wer = round(100 * normalized_wer, 2)
    normalized_cer = round(100 * normalized_cer, 2)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"WER              : {wer}%")
    print(f"CER              : {cer}%")
    print(f"Normalized WER   : {normalized_wer}%")
    print(f"Normalized CER   : {normalized_cer}%")

    print("=" * 50)

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    dataset_name = args.dataset.replace(
        "/",
        "_"
    )

    model_name = args.model.replace(
        "/",
        "_"
    )

    result_file_path = os.path.join(
        args.output_dir,
        f"{dataset_name}_{args.split}_{model_name}.txt"
    )

    with open(
        result_file_path,
        "w",
        encoding="utf-8"
    ) as result_file:

        result_file.write(
            "WHISPER EVALUATION RESULTS\n"
        )

        result_file.write(
            "=" * 50 + "\n\n"
        )

        result_file.write(
            f"Model: {args.model}\n"
        )

        result_file.write(
            f"Dataset: {args.dataset}\n"
        )

        result_file.write(
            f"Config: {args.config}\n"
        )

        result_file.write(
            f"Split: {args.split}\n\n"
        )

        result_file.write(
            f"WER: {wer}%\n"
        )

        result_file.write(
            f"CER: {cer}%\n"
        )

        result_file.write(
            f"Normalized WER: {normalized_wer}%\n"
        )

        result_file.write(
            f"Normalized CER: {normalized_cer}%\n\n"
        )

        result_file.write(
            "=" * 50 + "\n\n"
        )

        for reference, prediction in zip(
            references,
            predictions
        ):

            result_file.write(
                f"REF: {reference}\n"
            )

            result_file.write(
                f"HYP: {prediction}\n"
            )

            result_file.write(
                "-" * 50 + "\n"
            )

    print(
        f"\nResults saved to: {result_file_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Whisper model or fine-tuned model path"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hugging Face dataset ID"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Hugging Face dataset configuration"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="hi",
        help="Whisper language code"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions_dir",
        help="Directory for evaluation results"
    )

    args = parser.parse_args()

    main(args)