import torch
from transformers import CLIPProcessor, CLIPModel
import datasets
import aiohttp
from tqdm import tqdm

from dmx.compressor.utils.benchmark import (
    measure_model_runtime,
    measure_model_accuracy,
    measure_model_error,
    EVALUATION_MODE,
)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def clip_evaluator(model, device, desc):
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    ds = datasets.load_dataset(
        "HuggingFaceM4/COCO",
        split="test",
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )
    # Take the first 1000
    ds = ds.take(1000)
    dl = torch.utils.data.DataLoader(torch.arange(len(ds)), batch_size=8)
    all_image_embeds = []
    all_text_embeds = []
    print(f"evaluating clip model {desc}")
    for indices in tqdm(dl):
        batch = ds[indices]

        model_input = processor(
            text=[x["raw"] for x in batch["sentences"]],
            images=batch["image"],
            return_tensors="pt",
            padding=True,
        )
        model_input = {k: v.to(device) for k, v in model_input.items()}
        with torch.no_grad():
            model_output = model(**model_input)

        all_image_embeds.append(model_output.image_embeds)
        all_text_embeds.append(model_output.text_embeds)

    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    text_img_sim = all_text_embeds @ all_image_embeds.t()

    ordered_winners = torch.argsort(text_img_sim, dim=-1, descending=True)
    correct_winner_mask = (
        ordered_winners
        == torch.arange(ordered_winners.shape[0])
        .unsqueeze(1)
        .to(ordered_winners.device)
    ).long()
    top_K = [1, 5, 10]
    metrics = {
        f"top_{k}": correct_winner_mask[:, :k].sum(-1).float().mean() for k in top_K
    }
    return metrics


def clip_model_maker(return_kwargs=False):
    device = torch.device("cuda")

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    ds = datasets.load_dataset(
        "HuggingFaceM4/COCO",
        split="test",
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )

    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=ds[:2]["image"],
        return_tensors="pt",
        padding=True,
    )

    model_forward_kwargs = {
        "input_ids": inputs["input_ids"].to(device),
        "pixel_values": inputs["pixel_values"].to(device),
    }

    def model_runner(m):
        return m(**model_forward_kwargs)

    if return_kwargs:
        return [model, model_runner, clip_evaluator, device, model_forward_kwargs]
    else:
        return [model, model_runner, clip_evaluator, device]


def clip_text_model_maker():
    model, _, clip_evaluator, device, model_forward_kwargs = clip_model_maker(
        return_kwargs=True
    )

    model = model.text_model
    del model_forward_kwargs["pixel_values"]

    def model_runner(m):
        return m(**model_forward_kwargs)

    return model, model_runner, clip_evaluator, device


def clip_vision_model_maker():
    model, _, clip_evaluator, device, model_forward_kwargs = clip_model_maker(
        return_kwargs=True
    )

    model = model.vision_model
    del model_forward_kwargs["input_ids"]

    def model_runner(m):
        return m(**model_forward_kwargs)

    return model, model_runner, clip_evaluator, device


def main():
    active_modes = [
        EVALUATION_MODE.VANILLA,
        EVALUATION_MODE.BASELINE,
        EVALUATION_MODE.FP8,
        EVALUATION_MODE.BASIC,
        EVALUATION_MODE.BASIC_NOVSIMD,
    ]

    print("********Error analysis clip")
    measure_model_error(clip_model_maker, active_modes, EVALUATION_MODE.BASELINE)

    print("********Accuracy measurement of CLIP")
    measure_model_accuracy(clip_model_maker, active_modes)

    print("********RUNTIME measurments of CLIP TEXT")
    measure_model_runtime(clip_text_model_maker, active_modes)

    print("********RUNTIME measurments of CLIP VISION")
    measure_model_runtime(clip_vision_model_maker, active_modes)


if __name__ == "__main__":
    main()
