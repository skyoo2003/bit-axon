def test_import_version():
    from bit_axon import __version__

    assert isinstance(__version__, str)


def test_import_config():
    from bit_axon import BitAxonConfig

    config = BitAxonConfig()
    assert config.hidden_dim == 2560


def test_import_model():
    from bit_axon import BitAxonModel

    assert BitAxonModel is not None


def test_import_tokenizer():
    from bit_axon import QwenTokenizerWrapper

    assert QwenTokenizerWrapper is not None
