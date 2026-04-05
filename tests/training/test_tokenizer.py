import mlx.core as mx

from bit_axon.training.tokenizer import QwenTokenizerWrapper


class TestQwenTokenizerWrapper:
    def test_encode_returns_list_of_ints(self, test_tokenizer: QwenTokenizerWrapper):
        ids = test_tokenizer.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_decode_roundtrip(self, test_tokenizer: QwenTokenizerWrapper):
        text = "hello world"
        ids = test_tokenizer.encode(text)
        decoded = test_tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded.strip() == text

    def test_decode_accepts_mx_array(self, test_tokenizer: QwenTokenizerWrapper):
        ids = test_tokenizer.encode("test")
        arr = mx.array(ids)
        mx.eval(arr)
        decoded = test_tokenizer.decode(arr)
        assert isinstance(decoded, str)

    def test_apply_chat_template_user_only(self, test_tokenizer: QwenTokenizerWrapper):
        messages = [{"role": "user", "content": "What is 2+2?"}]
        ids = test_tokenizer.apply_chat_template(messages)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        decoded = test_tokenizer.decode(ids, skip_special_tokens=False)
        assert "<|im_start|>user" in decoded
        assert "What is 2+2?" in decoded
        assert "<|im_end|>" in decoded

    def test_apply_chat_template_with_system(self, test_tokenizer: QwenTokenizerWrapper):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        ids = test_tokenizer.apply_chat_template(messages)
        decoded = test_tokenizer.decode(ids, skip_special_tokens=False)
        assert "<|im_start|>system" in decoded
        assert "You are helpful." in decoded
        assert "<|im_start|>user" in decoded
        assert "Hi" in decoded

    def test_apply_chat_template_add_generation_prompt(self, test_tokenizer: QwenTokenizerWrapper):
        messages = [{"role": "user", "content": "Hello"}]
        ids_no_prompt = test_tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        ids_with_prompt = test_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        assert len(ids_with_prompt) > len(ids_no_prompt)
        decoded = test_tokenizer.decode(ids_with_prompt, skip_special_tokens=False)
        assert "<|im_start|>assistant" in decoded

    def test_apply_chat_template_multi_turn(self, test_tokenizer: QwenTokenizerWrapper):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        ids = test_tokenizer.apply_chat_template(messages)
        decoded = test_tokenizer.decode(ids, skip_special_tokens=False)
        assert decoded.count("<|im_start|>user") == 2
        assert decoded.count("<|im_start|>assistant") == 1
        assert "Hello" in decoded
        assert "Hi there!" in decoded
        assert "How are you?" in decoded

    def test_pad_token_id(self, test_tokenizer: QwenTokenizerWrapper):
        pad_id = test_tokenizer.pad_token_id
        assert isinstance(pad_id, int)
        assert pad_id >= 0

    def test_eos_token_id(self, test_tokenizer: QwenTokenizerWrapper):
        eos_id = test_tokenizer.eos_token_id
        assert isinstance(eos_id, int)
        assert eos_id >= 0

    def test_vocab_size(self, test_tokenizer: QwenTokenizerWrapper):
        vs = test_tokenizer.vocab_size
        assert isinstance(vs, int)
        assert vs > 0
