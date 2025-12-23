from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset, load_from_disk, load_dataset
import torch
import numpy as np
from pyctcdecode import build_ctcdecoder, BeamSearchDecoderCTC
import jiwer
import json
import argparse
from typing import Tuple, Dict, Optional
import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_DIR = os.path.join(THIS_DIR, "../../../multi-kakiokoshi/src/data/mcv-sps-st-09-2025/")
REMOTE_DATA_ID = "ctaguchi/mcv-sps-{lang}-segmented"
LOCAL_MODEL_DIR = "/afs/crc/group/nlp/11/ctaguchi/multi-kakiokoshi-models"
KENLM_MODEL_PATH = os.path.join(THIS_DIR, "../{n}gram/{n}gram_{lang}_correct.binary")
USERNAME = "ctaguchi"


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        choices=["ady", "bas", "kbd", "qxp", "ush",
                "aln", "bew", "bxk", "cgg", "el-CY",
                "hch", "kcn", "koo", "led", "lke",
                "lth", "meh", "mmc", "pne", "ruc",
                "rwm", "sco", "tob", "top", "ttj", "ukv"],
        required=True,
        help="Language."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["max3"],
        help="Model group, e.g. max3."
    )
    parser.add_argument(
        "--use_local_model",
        action="store_true",
        help="If set, the model/processor saved locally will be used."
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=50,
        help="Width in beam search."
    )
    return parser.parse_args()


MODEL_MAP = {"max3": "ssc-{lang}-mms-model-mix-adapt-max3"}


def load_model_and_processor(lang: str,
                             model_name: str,
                             use_local_model: bool) -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
    """Load the pretrained model and the processor.
    The model path looks something like: `ctaguchi/ssc-sco-mms-model-mix-adapt-max3` if loaded remotely.
    If loaded locally, we need to use the local path.
    """
    model_id = MODEL_MAP[model_name].format(lang=lang)
    if use_local_model:
        model_dir = os.path.join(LOCAL_MODEL_DIR, model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_dir)
        processor = Wav2Vec2Processor.from_pretrained(model_dir)
    else:
        
        model = Wav2Vec2ForCTC.from_pretrained(f"{USERNAME}/{model_id}")
        processor = Wav2Vec2Processor.from_pretrained(f"{USERNAME}/{model_id}")
    return model, processor


def load_eval_data(lang: str,
                   use_dev: bool = True,
                   use_local_dataset: bool = False) -> Dataset:
    if use_local_dataset:
        ds_path = os.path.join(LOCAL_DATA_DIR, lang, "segmented_dataset")
        ds = load_from_disk(ds_path)
    else:
        # load from Hugging Face Hub
        ds = load_dataset(REMOTE_DATA_ID.format(lang=lang))
    
    if use_dev:
        dev = ds.filter(lambda x: x["split"] == "dev")
    else:
        raise NotImplementedError("Test data is not available yet.")
    return dev


def logits_from_audio_array(audio_array: np.ndarray,
                            model: Wav2Vec2ForCTC,
                            processor: Wav2Vec2Processor,
                            device: str,
                            sampling_rate: int = 16000) -> torch.Tensor:
    """Compute logits from the audio array."""
    input_values = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    ).input_values.to(device)
    
    with torch.no_grad():
        logits = model(input_values).logits
    return logits


def prepare_decoder(processor: Wav2Vec2Processor,
                    kenlm_model: Optional[str] = None,
                    alpha: Optional[float] = None,
                    beta: Optional[float] = None):
    """Prepare a CTC decoder with pyctcdecode.
    If kenlm_model is not None, an n-gram language model will be used.
    """
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

    if kenlm_model:
        decoder = build_ctcdecoder(labels,
                                   kenlm_model_path=kenlm_model,
                                   alpha=alpha,
                                   beta=beta)
    else:
        decoder = build_ctcdecoder(labels)
    return decoder, keep_ids


def multidecode(batch: dict,
                model: Wav2Vec2ForCTC,
                processor: Wav2Vec2Processor,
                device: str,
                decoders: Dict[str, BeamSearchDecoderCTC],
                beam_width: int = 50,
                sampling_rate: int = 16000) -> dict:
    """Decode with multiple decoding methods."""
    logits = logits_from_audio_array(
        audio_array=batch["audio"]["array"],
        model=model,
        processor=processor,
        device=device,
        sampling_rate=sampling_rate)
    
    # Greedy
    pred_ids = torch.argmax(logits, dim=-1)
    greedy_pred = processor.batch_decode(pred_ids)[0]
    batch["greedy_pred"] = greedy_pred
    
    # Beam search, N-gram
    logits = logits.cpu().numpy()[0] # pyctcdecode doesn't accept torch.Tensor
    for decoder_name, decoder in decoders.items():
        decoded = decoder.decode(logits, beam_width=beam_width)
        batch[f"{decoder_name}_pred"] = decoded
    
    return batch


def eval(ds: Dataset) -> Dict[str, float]:
    """Evaluate the predictions.
    Compute WER and CER."""
    pred_column_names = [c for c in ds.column_names if c.endswith("_pred")]
    stats = {}
    golds = ds["transcription"]
    for c in pred_column_names:
        preds = ds[c]
        wer = jiwer.wer(reference=golds, hypothesis=preds)
        cer = jiwer.cer(reference=golds, hypothesis=preds)
        stats[c] = {"wer": wer,
                    "cer": cer,
                    "references": golds,
                    "predicitons": preds}
    return stats


def main(args: argparse.Namespace) -> None:
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model, processor = load_model_and_processor(lang=args.language,
                                                model_name=MODEL_MAP[args.model],
                                                use_local_model=args.use_local_model)
    model.to(device)
    
    # load dev data
    ds = load_eval_data(lang=args.language,
                        use_dev=True,
                        use_local_dataset=False)
    
    # prepare decoders
    beam_search_decoder = prepare_decoder(processor=processor)
    bigram_decoder = prepare_decoder(processor=processor,
                                     kenlm_model=KENLM_MODEL_PATH.format(n=2, lang=args.language))
    pentagram_decoder = prepare_decoder(processor=processor,
                                        kenlm_model=KENLM_MODEL_PATH.format(n=5, lang=args.language))
    decoders = {"beam_search_decoder": beam_search_decoder,
                "2gram_decoder": bigram_decoder,
                "5gram_decoder": pentagram_decoder}
    
    # run inference
    ds = ds.map(multidecode,
                fn_kwargs={"model": model,
                           "processor": processor,
                           "device": device,
                           "decoders": decoders,
                           "beam_width": args.beam_width,
                           "sampling_rate": 16000})
    # `ds` now should contain `transcription` (gold), `greedy_pred`, `beam_search_decoder_pred`, `2gram_decoder_pred`, `5gram_decoder_pred`.
    
    # evaluate
    stats = eval(ds)
    # save
    with open(os.path.join(THIS_DIR, f"stats_{args.language}.json"), "w") as f:
        json.dump(stats, f, indent=4)
    

if __name__ == "__main__":
    args = get_args()
    main(args)