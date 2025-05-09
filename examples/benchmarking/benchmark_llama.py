import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline

from dmx.compressor.utils.benchmark import (
    measure_model_runtime,
    measure_model_error,
    EVALUATION_MODE,
)


def llama_model_maker():
    device = torch.device("cuda")

    model = "d-matrix/Llama-3.2-1b"
    task = "text-generation"
    pipe = pipeline(
        task=task,
        model=model,
        trust_remote_code=True,
        device_map="auto",
    )

    def model_runner(m):
        return pipe(
            text_inputs="The unanimous Declaration of the thirteen united States of America, ",
            max_new_tokens=1,
        )

    return [pipe.model, model_runner, None, device]


def main():
    active_modes = [EVALUATION_MODE.BASIC, EVALUATION_MODE.BASELINE]

    print("********RUNTIME measurments of Llama")
    measure_model_runtime(llama_model_maker, active_modes)

    print("********Error analysis Llama")
    measure_model_error(llama_model_maker, active_modes, EVALUATION_MODE.BASELINE)


if __name__ == "__main__":
    main()
