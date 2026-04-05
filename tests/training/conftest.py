import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

from bit_axon.training.tokenizer import QwenTokenizerWrapper


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


@pytest.fixture
def test_tokenizer(tmp_path):
    """Create minimal Qwen2.5-compatible tokenizer for testing."""
    byte_vocab = {char: i for i, char in enumerate(_bytes_to_unicode().values())}
    tok = Tokenizer(models.BPE(vocab=byte_vocab, merges=[], unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    tok.add_special_tokens(
        [
            AddedToken("<｜end▁of▁text｜>", special=True),
            AddedToken("<|im_start|>", special=True),
            AddedToken("<|im_end|>", special=True),
        ]
    )

    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return QwenTokenizerWrapper(str(path))
