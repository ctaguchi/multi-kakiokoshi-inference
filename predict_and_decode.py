import numpy as np
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from pyctcdecode import build_ctcdecoder
import pandas as pd
from typing import Dict, Any, Optional
from datasets import load_dataset, load_from_disk, Dataset
import torch
import argparse
import jiwer
import json


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
    parser.add_argument(
        "--reuse_logits_path",
        type=str,
        default=None,
        help="Logits path if you want to reuse them."
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="If turned on, do grid search of hyperparameters (alpha and beta)."
    )
    parser.add_argument(
        "-l", # CURRENTLY UNUSED
        "--lang",
        type=str,
        default=None,
        help="If specified, this language will be used for cross-validation (grid search)."
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
    if batch["audio"]["path"]:
        # dev doesn't have path
        batch["path"] = os.path.basename(batch["audio"]["path"])
    return batch


def prepare_decoder(lang: str,
                    processor: Wav2Vec2Processor,
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
           beam_width: int = 50,):
    """Beam-search decoding.
    Time complexity is T x B x V, where
        - T: time (array size)
        - B: beam width
        - V: vocabulary size
    """
    logits = np.array(batch["logits"][0])
    logits_reduced = logits[:, keep_ids]
    # if ngram:
    #     decoded = decoder.decode(logits_reduced)
    # else:
    decoded = decoder.decode(logits_reduced, beam_width=beam_width).replace("⁇", "")
    batch["decoded"] = decoded
    return batch


def process_language(lang: str,
                     dataset: Dataset,
                     model: Wav2Vec2ForCTC,
                     processor: Wav2Vec2Processor,
                     beam_width: int = 50,
                     ngram: bool = False,
                     alpha: float = 0.2,
                     beta: float = 0.0,
                     reuse_logits: bool = False,
                     logits: Optional[Dataset] = None):
    # test = test_data.filter(lambda x: x["language"] == lang)
    if not reuse_logits:
        logits = dataset.map(get_logits,
                             fn_kwargs={"processor": processor,
                                        "model": model,
                                        "device": device},
                             remove_columns=["audio"])
        
    else:
        assert logits is not None

    decoder, keep_ids = prepare_decoder(lang=lang,
                                        processor=processor,
                                        ngram=ngram,
                                        alpha=alpha,
                                        beta=beta)
    preds = logits.map(decode,
                       fn_kwargs={"decoder": decoder,
                                  "keep_ids": keep_ids,
                                  "beam_width": beam_width},
                       num_proc=6)
    return preds


def grid_search_alpha_beta(lang: str,
                           model: Wav2Vec2ForCTC,
                           processor: Wav2Vec2Processor,
                           device: str,
                           beam_width: int,
                           ngram: bool,
                           alpha_values: list[float],
                           beta_values: list[float],
                           results_id: str) -> tuple[float, float]:
    """
    Simple cross-validation / grid search for (alpha, beta) for one language.
    Reuses the same logits and calls `process_language` repeatedly with different
    alpha/beta, then picks the pair that minimizes WER on the given TSV.

    Returns: (best_alpha, best_beta)
    """
    best_alpha = None
    best_beta = None
    best_wer = float("inf")

    print(f"[{lang}] Starting grid search over alpha, beta...")
    print("  alpha grid:", alpha_values)
    print("  beta  grid:", beta_values)
    
    # First, get logits for dev
    dev = load_dataset(f"{USERNAME}/mcv-sps-{lang}-segmented", split="train")
    # Pick up just the first 20 samples
    dev = dev.filter(lambda x: x["split"] == "dev").select(range(20))
    refs = dev["transcription"]
    logits = dev.map(get_logits,
                     fn_kwargs={"processor": processor, # define?
                                "model": model, # define?
                                "device": device # define?
                                },
                     remove_columns=["audio"])

    stats_list = []
    # We assume `logits` is a Dataset with columns: 'logits', 'path', etc.
    for alpha in alpha_values:
        for beta in beta_values:
            print(f"[{lang}] Trying alpha={alpha}, beta={beta} ...")

            preds = process_language(lang=lang,
                                     dataset=dev,
                                     model=model,
                                     processor=processor,
                                     beam_width=beam_width,
                                     ngram=ngram,
                                     alpha=alpha,
                                     beta=beta,
                                     reuse_logits=True,
                                     logits=logits)
            
            wer = jiwer.wer(refs, [x["decoded"] for x in preds])
            cer = jiwer.cer(refs, [x["decoded"] for x in preds])
            
            stats = {"language": lang,
                     "alpha": alpha,
                     "beta": beta,
                     "wer": wer,
                     "cer": cer}
            stats_list.append(stats)

            for k, v in stats.items():
                print(f"{k}: {v}")
            print("-" * 20)
            
            if wer < best_wer:
                best_wer = wer
                best_alpha = alpha
                best_beta = beta
                print(f"[{lang}]   New best: WER={best_wer:.4f} (alpha={best_alpha}, beta={best_beta})")
    
    results_dir = os.path.join("results", results_id)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"grid_search_results_{lang}.json"), "w") as f:
        json.dump(stats_list, f, indent=4)

    print(f"[{lang}] Grid search done. Best WER={best_wer:.4f}, alpha={best_alpha}, beta={best_beta}")
    return best_alpha, best_beta


def load_processor(model_name: str,
                   lang: str):
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
    return processor



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
        model_name = os.path.join(args.model_dir, f"ssc-{lang}-{args.model}")
        processor = load_processor(model_name=model_name,
                                   lang=lang)
        model = Wav2Vec2ForCTC.from_pretrained(model_name,
                                               ignore_mismatched_sizes=True).to(device)
        print("Model and processor loaded.")
        print("Using model:", model_name)
        
        # Grid search
        alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        beta_grid  = [-3.0, -2.0, -1.0, 0.0, 1.0]
        
        if args.ngram and args.grid_search:
            print("Running grid search to find the best alpha and best beta...")
            best_alpha, best_beta = grid_search_alpha_beta(lang=lang,
                                                           model=model,
                                                           processor=processor,
                                                           device=device,
                                                           beam_width=args.beam_width,
                                                           ngram=args.ngram,
                                                           alpha_values=alpha_grid,
                                                           beta_values=beta_grid,
                                                           results_id=args.results_id)
        else:
            best_alpha = args.alpha
            best_beta = args.beta
        
        
        # Test phase
        if os.path.exists(os.path.join(logits_dir, f"{lang}.logits")):
            print(f"Found existing logits: {lang}.logits")
            preds = load_from_disk(os.path.join(logits_dir, f"{lang}.logits")) 
            
        elif args.reuse_logits_path:
            assert os.path.exists(args.reuse_logits_path), "Make sure the logits dir exists."
            print(f"Reusing the logits from {args.reuse_logits_path}...")
            logits = load_from_disk(os.path.join(args.reuse_logits_path, f"{lang}.logits")) # it might contain preds
            
            preds = process_language(lang,
                                     dataset=None,
                                     model=None,
                                     processor=processor,
                                     beam_width=args.beam_width,
                                     ngram=args.ngram,
                                     alpha=best_alpha,
                                     beta=best_beta,
                                     reuse_logits=True,
                                     logits=logits)
            
        else:            
            # Main test inference
            test = load_dataset(f"{USERNAME}/mcv-sps-test-{lang}", split="train")
            preds = process_language(lang,
                                     dataset=test,
                                     model=model,
                                     processor=processor,
                                     beam_width=args.beam_width,
                                     ngram=args.ngram,
                                     alpha=best_alpha,
                                     beta=best_beta)
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
        