import numpy as np
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from pyctcdecode import build_ctcdecoder
import pandas as pd
from typing import Dict, Any, Optional
from datasets import load_dataset, load_from_disk
import torch
import argparse


SSCLangs = [ # Languages to be trained with Spontaneous Speech Corpus data
    "aln", "bew", "bxk", "el-CY", "cgg", "hch", "kcn", "koo", "led", "lke",
    "lth", "meh", "mmc", "pne", "ruc", "rwm", "sco", "tob", "top", "ttj", "ukv",
]
CVLangs = [ # Languages to be trained with Common Voice data
    "ady", "bas", "kbd", "qxp", "ush"
]

all_langs = SSCLangs + CVLangs

USERNAME = "ctaguchi"


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--results_id",
        type=str,
        help="Results identifier for separating folders."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model name."
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=50,
        help="Beam width."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Model directory."
    )
    parser.add_argument(
        "--load_remote_model",
        action="store_true",
        help="Load from the hub."
    )
    parser.add_argument(
        "--ngram",
        action="store_true",
        help="Use ngram decoding."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Alpha hyperparam (how much we trust LM)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Beta hyperparam. How much we prioritize inserting a new word (space)"
    )
    return parser.parse_args()


def get_logits(batch: Dict[str, Any],
               processor: Wav2Vec2Processor,
               model: Wav2Vec2ForCTC,
               device: str) -> Dict[str, Any]:
    """Inference with the main Wav2Vec2ForCTC model."""
    array = batch["audio"]["array"]
    inp = processor(
        array,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = model(inp.input_values.to(device)).logits

    batch["logits"] = logits
    batch["path"] = os.path.basename(batch["audio"]["path"])
    return batch


def prepare_decoder(processor: Wav2Vec2Processor,
                    ngram: bool,
                    alpha: float = 0.2,
                    beta: float = 0.0):
    # id -> token list in order
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])

    full_labels = [tok for tok, _ in sorted_vocab]

    # 2. Fix special tokens
    pad_id = processor.tokenizer.pad_token_id
    unk_id = processor.tokenizer.unk_token_id
    bos_id = vocab_dict.get("<s>")
    eos_id = vocab_dict.get("</s>")
    word_delim_id = getattr(processor.tokenizer,
                            "word_delimiter_token_id",
                            None)
    drop_ids = {i for i in [eos_id] if i is not None}
    # Keep everything else (including PAD = CTC blank)
    keep_ids = [i for i in range(len(full_labels)) if i not in drop_ids]
    labels = [full_labels[i] for i in keep_ids]

    # Map PAD to CTC blank in *reduced* index space
    if pad_id in keep_ids:
        blank_pos = keep_ids.index(pad_id)
        labels[blank_pos] = ""

    # Optional: map word delimiter to space if present
    word_delim_id = getattr(processor.tokenizer, "word_delimiter_token_id", None)
    if word_delim_id is not None and word_delim_id in keep_ids:
        delim_pos = keep_ids.index(word_delim_id)
        labels[delim_pos] = " "

    print("len(full_labels) =", len(full_labels))  # should be 82
    print("len(labels)      =", len(labels))       # should be 79

    if ngram:
        kenlm_model_path = os.path.join("ngram", f"5gram_correct_{lang}.binary")
        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=kenlm_model_path,  # or "char.arpa"
            alpha=alpha, # LM weight
            beta=beta, # word insertion / length penalty; lower -> less space insertion
        )
    else:
        decoder = build_ctcdecoder(labels)
    return decoder, keep_ids


def decode(batch: Dict[str, Any],
           decoder,
           keep_ids,
           beam_width: int = 50,
           ngram: bool = False):
    """Beam-search decoding.
    Time complexity is T x B x V, where
        - T: time (array size)
        - B: beam width
        - V: vocabulary size
    """
    logits = np.array(batch["logits"][0])
    logits_reduced = logits[:, keep_ids]
    if ngram:
        decoded = decoder.decode(logits_reduced)
    else:
        decoded = decoder.decode(logits_reduced, beam_width=beam_width).replace("⁇", "")
    batch["decoded"] = decoded
    return batch


def process_language(lang: str,
                     model_suffix: str,
                     beam_width: int = 50,
                     ngram: bool = False,
                     results_dir: str = "results",
                     model_dir: Optional[str] = None,
                     load_remote_model: bool = False,
                     alpha: float = 0.2,
                     beta: float = 0.0):
    # test = test_data.filter(lambda x: x["language"] == lang)
    test = load_dataset(f"{USERNAME}/mcv-sps-test-{lang}", split="train")
    try:
        if load_remote_model:
            model_name = f"{USERNAME}/ssc-{lang}-{model_suffix}"
        else:
            model_name = os.path.join(model_dir, f"ssc-{lang}-{model_suffix}")
        model = Wav2Vec2ForCTC.from_pretrained(model_name,
                                               ignore_mismatched_sizes=True).to(device)
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        
    except:
        try:
            model = Wav2Vec2ForCTC.from_pretrained(model_name,
                                               ignore_mismatched_sizes=True).to(device)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                model_name,
                unk_token="[UNK]",
                pad_token="[PAD]",
                word_delimiter_token="|",
                target_lang=lang
            )
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=True
            )
            processor = Wav2Vec2Processor(tokenizer=tokenizer,
                                          feature_extractor=feature_extractor)
        except:
            raise ValueError
        # model_name = f"{USERNAME}/ssc-{lang}-mms-model-mix-adapt-max"
        # processor = Wav2Vec2Processor.from_pretrained(model_name)
        # model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    print(f"Using model: {model_name}")

    logits = test.map(get_logits,
                      fn_kwargs={"processor": processor,
                               "model": model,
                               "device": device},
                      remove_columns=["audio"])

    decoder, keep_ids = prepare_decoder(processor,
                                        ngram=ngram,
                                        alpha=alpha,
                                        beta=beta)
    preds = logits.map(decode,
                       fn_kwargs={"decoder": decoder,
                                  "keep_ids": keep_ids,
                                  "beam_width": beam_width},
                       num_proc=6)
    return preds


if __name__ == "__main__":
    args = get_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    logits_dir = f"results/{args.results_id}/logits"
    os.makedirs(logits_dir, exist_ok=True)
    results_dir = f"results/{args.results_id}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    for lang in all_langs:
        if os.path.exists(os.path.join(results_dir, f"{lang}.tsv")):
            print(f"Found the results for {lang}, skipping...")
            continue
        print(f"Working on {lang}...")
        if os.path.exists(os.path.join(logits_dir, f"{lang}.logits")):
            print(f"Found existing logits: {lang}.logits")
            preds = load_from_disk(os.path.join(logits_dir, f"{lang}.logits"))
        else:
            preds = process_language(lang,
                                     model_suffix=args.model,
                                     beam_width=args.beam_width,
                                     ngram=args.ngram,
                                     model_dir=args.model_dir,
                                     load_remote_model=args.load_remote_model,
                                     alpha=args.alpha,
                                     beta=args.beta)
            preds.save_to_disk(os.path.join(logits_dir, f"{lang}.logits"))

        # Load tsv
        if lang in SSCLangs:
            tsv_file = os.path.join("data", "multilingual-general", f"{lang}.tsv")
        elif lang in CVLangs:
            tsv_file = os.path.join("data", "unseen-langs", f"{lang}.tsv")
        
        df = pd.read_csv(tsv_file, sep="\t", index_col=False)

        transcriptions = [x["decoded"] for x in preds]
        paths = [x["path"] for x in preds]
        mapping = dict(zip(paths, transcriptions))
        df["sentence"] = df["audio_file"].map(mapping)

        # tsv, preds = process_language(lang)
        df.to_csv(os.path.join(results_dir,
                                f"{lang}.tsv"),
                sep="\t",
                index=False)

        print(f"Evaluation on {lang} done.")
        