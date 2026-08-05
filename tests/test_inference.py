import torch

from scripts.infer import generate


class TokenBatch(dict):
    def to(self, device):
        return TokenBatch({key: value.to(device) for key, value in self.items()})


class StubProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, audios, instruction=None, formatted_prompt=None):
        self.calls.append(
            {
                "audios": audios,
                "instruction": instruction,
                "formatted_prompt": formatted_prompt,
            }
        )
        return TokenBatch(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            audio_features=torch.zeros(1, 10, 128),
            audio_lengths=torch.tensor([10]),
            audio_counts=torch.tensor([1]),
        )

    def decode(self, token_ids):
        assert token_ids.tolist() == [7]
        return "A synthetic tone."


class StubModel:
    device = torch.device("cpu")

    def generate(self, **inputs):
        assert inputs["input_ids"].device.type == "cpu"
        assert inputs["audio_features"].device.type == "cpu"
        assert inputs["max_new_tokens"] == 4
        assert inputs["do_sample"] is False
        return torch.tensor([[7]])


def test_generate_uses_instruction_for_simple_prompt():
    processor = StubProcessor()
    response = generate(
        StubModel(),
        processor,
        ["tone.wav"],
        "Describe the audio.",
        max_new_tokens=4,
    )

    assert response == "A synthetic tone."
    assert processor.calls == [
        {
            "audios": ["tone.wav"],
            "instruction": "Describe the audio.",
            "formatted_prompt": None,
        }
    ]


def test_generate_preserves_explicit_audio_placement():
    processor = StubProcessor()
    prompt = "<audio>Compare this recording with <audio>this example."
    response = generate(
        StubModel(),
        processor,
        ["main.wav", "example.wav"],
        prompt,
        max_new_tokens=4,
    )

    assert response == "A synthetic tone."
    assert processor.calls[0]["instruction"] is None
    assert processor.calls[0]["formatted_prompt"] == prompt
