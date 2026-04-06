from bit_axon.training.packing import PackedBatch, SequencePacker

MAX_SEQ_LEN = 16
EOS_ID = 99


class TestSequencePacker:
    def test_single_example_fills_exactly(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        ids = list(range(MAX_SEQ_LEN))
        mask = [1] * MAX_SEQ_LEN

        result = packer.add_example(ids, mask)
        assert len(result) == 1
        assert result[0].token_ids == ids
        assert result[0].loss_mask == mask
        assert packer.flush() is None

    def test_two_examples_fit_in_one_sequence(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        ids1 = list(range(8))
        mask1 = [1] * 8
        ids2 = list(range(100, 107))
        mask2 = [1] * 7

        packer.add_example(ids1, mask1)
        batches = packer.add_example(ids2, mask2)

        assert len(batches) == 1
        batch = batches[0]
        assert len(batch.token_ids) == MAX_SEQ_LEN
        assert batch.token_ids[:8] == ids1
        assert batch.token_ids[8] == EOS_ID
        assert batch.token_ids[9:16] == ids2[:7]
        assert batch.loss_mask[:8] == [1] * 8
        assert batch.loss_mask[8] == 0
        assert batch.loss_mask[9:16] == [1] * 7
        assert packer.flush() is None

    def test_example_spans_two_sequences(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)

        ids1 = list(range(10))
        mask1 = [1] * 10
        packer.add_example(ids1, mask1)

        ids2 = list(range(100, 120))
        mask2 = [1] * 20
        batches = packer.add_example(ids2, mask2)

        assert len(batches) == 1
        assert len(batches[0].token_ids) == MAX_SEQ_LEN

        flushed = packer.flush()
        assert flushed is not None
        real_tokens = 10 + 1 + 16 - MAX_SEQ_LEN
        assert sum(flushed.loss_mask) == real_tokens

    def test_flush_partial_buffer(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        ids = [1, 2, 3]
        mask = [1, 1, 0]

        packer.add_example(ids, mask)
        flushed = packer.flush()

        assert flushed is not None
        assert len(flushed.token_ids) == MAX_SEQ_LEN
        assert flushed.token_ids[:3] == [1, 2, 3]
        assert flushed.token_ids[3:] == [0] * 13
        assert flushed.loss_mask[:3] == [1, 1, 0]
        assert flushed.loss_mask[3:] == [0] * 13

    def test_flush_empty_buffer(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        assert packer.flush() is None

    def test_truncate_oversized_example(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        ids = list(range(20))
        mask = [1] * 20

        batches = packer.add_example(ids, mask)
        assert len(batches) == 1
        assert batches[0].token_ids == list(range(16))
        assert batches[0].loss_mask == [1] * 16
        assert packer.flush() is None

    def test_reset_clears_buffer(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        packer.add_example([1, 2, 3], [1, 1, 1])
        packer.reset()
        assert packer.flush() is None

    def test_loss_mask_preserved(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        mask = [1, 0, 1, 0, 1, 1, 0, 0]
        ids = list(range(8))

        packer.add_example(ids, mask)
        flushed = packer.flush()

        assert flushed is not None
        assert flushed.loss_mask[:8] == mask

    def test_eos_inserted_between_examples(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        ids1 = [1, 2]
        ids2 = [3, 4]

        packer.add_example(ids1, [1, 1])
        packer.add_example(ids2, [1, 1])

        flushed = packer.flush()
        assert flushed is not None
        assert flushed.token_ids[:2] == [1, 2]
        assert flushed.token_ids[2] == EOS_ID
        assert flushed.token_ids[3:5] == [3, 4]

    def test_eos_has_zero_loss_mask(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        packer.add_example([1], [1])
        packer.add_example([2], [1])

        flushed = packer.flush()
        assert flushed is not None
        assert flushed.token_ids[1] == EOS_ID
        assert flushed.loss_mask[1] == 0

    def test_padding_has_zero_loss_mask(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        packer.add_example([5, 6], [1, 1])

        flushed = packer.flush()
        assert flushed is not None
        for i in range(2, MAX_SEQ_LEN):
            assert flushed.token_ids[i] == 0
            assert flushed.loss_mask[i] == 0

    def test_many_small_examples_packing_efficiency(self):
        packer = SequencePacker(max_seq_len=MAX_SEQ_LEN, eos_token_id=EOS_ID)
        example_len = 3
        all_batches: list[PackedBatch] = []

        for i in range(20):
            ids = [i * 10 + j for j in range(example_len)]
            mask = [1] * example_len
            all_batches.extend(packer.add_example(ids, mask))

        final = packer.flush()
        if final is not None:
            all_batches.append(final)

        total_tokens = sum(len(b.token_ids) for b in all_batches)
        total_loss_tokens = sum(sum(b.loss_mask) for b in all_batches)

        assert total_tokens == len(all_batches) * MAX_SEQ_LEN
        assert total_loss_tokens == 20 * example_len
        assert len(all_batches) > 1
