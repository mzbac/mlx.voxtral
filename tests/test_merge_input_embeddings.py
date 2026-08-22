"""Unit tests for VoxtralForConditionalGeneration._merge_input_embeddings.

The merge is fully vectorized in MLX (no numpy round-trip): audio embeddings are
ranked within each row, gathered, and written in with a single `where`. The audio
features describe one stream shared by every batch row, so that stream is
broadcast to every row that contains [AUDIO] placeholders.

Follows the approach used on the `fix/scatter-audio-embeddings` branch.

Run with:
    python tests/test_merge_input_embeddings.py
or:
    pytest tests/test_merge_input_embeddings.py
"""

import types

import mlx.core as mx
import numpy as np

from mlx_voxtral.modeling_voxtral import VoxtralForConditionalGeneration

AUDIO_TOKEN = 32000
TOKENS_PER_CHUNK = 2


def make_model(embed_dtype=mx.bfloat16, audio_dtype=mx.float32):
    """Build a minimal model exposing only what _merge_input_embeddings needs."""
    model = VoxtralForConditionalGeneration.__new__(VoxtralForConditionalGeneration)
    model.config = types.SimpleNamespace(audio_token_id=AUDIO_TOKEN)

    def embed_tokens(input_ids):
        batch, seq = input_ids.shape
        return mx.array(
            np.random.randn(batch, seq, 4).astype(np.float32), dtype=embed_dtype
        )

    def get_audio_embeds(input_features):
        num_audio_tokens = input_features.shape[0] * TOKENS_PER_CHUNK
        rows = np.arange(num_audio_tokens)[:, None]
        cols = np.arange(4)[None, :]
        return mx.array((rows * 10 + cols).astype(np.float32)[None, :, :], dtype=audio_dtype)

    model.embed_tokens = embed_tokens
    model.get_audio_embeds = get_audio_embeds
    return model


def positions_of(ids, token):
    ids = np.asarray(ids)
    return [np.nonzero(row == token)[0] for row in ids]


def to_np(arr):
    return np.array(arr.astype(mx.float32), dtype=np.float32)


def test_single_batch_places_audio_and_promotes_dtype():
    model = make_model()
    input_ids = mx.array(
        [[1, AUDIO_TOKEN, 2, 3, AUDIO_TOKEN, AUDIO_TOKEN, 4, AUDIO_TOKEN]]
    )
    inputs_embeds = model.embed_tokens(input_ids)
    assert inputs_embeds.dtype == mx.bfloat16

    out = model._merge_input_embeddings(
        input_ids=input_ids,
        input_features=mx.zeros((2, 2)),  # 2 chunks -> 4 audio tokens expected
        inputs_embeds=inputs_embeds,
    )

    assert out.dtype == mx.float32
    pos = positions_of(input_ids, AUDIO_TOKEN)[0]
    assert len(pos) == 4
    for j in range(input_ids.shape[1]):
        if j not in pos:
            assert np.allclose(to_np(out[0, j]), to_np(inputs_embeds[0, j]))
    audio_embeds = model.get_audio_embeds(mx.zeros((2, 2)))
    for k, j in enumerate(pos):
        assert np.allclose(to_np(out[0, int(j)]), to_np(audio_embeds[0, int(k)]))


def test_batch_broadcasts_single_audio_stream():
    model = make_model()
    input_ids = mx.array(
        [
            [1, AUDIO_TOKEN, 2, AUDIO_TOKEN, AUDIO_TOKEN, AUDIO_TOKEN],
            [3, AUDIO_TOKEN, AUDIO_TOKEN, 4, AUDIO_TOKEN, AUDIO_TOKEN],
            [5, 6, 7, 8, 9, 10],
        ]
    )
    inputs_embeds = model.embed_tokens(input_ids)
    out = model._merge_input_embeddings(
        input_ids=input_ids,
        input_features=mx.zeros((2, 2)),
        inputs_embeds=inputs_embeds,
    )

    audio_block = to_np(model.get_audio_embeds(mx.zeros((2, 2)))[0])
    pos = positions_of(input_ids, AUDIO_TOKEN)
    assert np.allclose(to_np(out[0, mx.array(pos[0])]), audio_block)
    assert np.allclose(to_np(out[1, mx.array(pos[1])]), audio_block)
    assert np.allclose(to_np(out[2]), to_np(inputs_embeds[2]))


def test_no_audio_tokens_preserves_dtype_and_does_not_mutate():
    model = make_model()
    input_ids = mx.array([[1, 2, 3, 4, 5]])
    inputs_embeds = model.embed_tokens(input_ids)
    assert inputs_embeds.dtype == mx.bfloat16

    out = model._merge_input_embeddings(
        input_ids=input_ids,
        input_features=mx.zeros((1, 2)),
        inputs_embeds=inputs_embeds,
    )

    assert out.dtype == mx.bfloat16
    assert out is inputs_embeds


def test_caller_provided_inputs_embeds_is_not_mutated():
    model = make_model()
    input_ids = mx.array(
        [[1, AUDIO_TOKEN, 2, AUDIO_TOKEN, 3, AUDIO_TOKEN, 4, AUDIO_TOKEN]]
    )
    caller = model.embed_tokens(input_ids)
    snapshot = to_np(caller).copy()

    out = model._merge_input_embeddings(
        input_ids=input_ids,
        input_features=mx.zeros((2, 2)),
        inputs_embeds=caller,
    )

    assert np.array_equal(to_np(caller), snapshot)
    assert not np.allclose(to_np(out[0]), to_np(caller[0]))


def test_mismatched_audio_token_count_raises():
    model = make_model()
    # 3 [AUDIO] tokens but the audio features encode 2 chunks -> 4 expected.
    input_ids = mx.array([[1, AUDIO_TOKEN, 2, AUDIO_TOKEN, AUDIO_TOKEN, 3]])
    inputs_embeds = model.embed_tokens(input_ids)
    try:
        model._merge_input_embeddings(
            input_ids=input_ids,
            input_features=mx.zeros((2, 2)),
            inputs_embeds=inputs_embeds,
        )
    except ValueError as exc:
        assert "Expected 4 audio tokens" in str(exc)
    else:
        raise AssertionError("expected ValueError for audio token count mismatch")


def test_non_single_audio_stream_raises():
    model = make_model()

    def get_audio_embeds(input_features):
        return mx.array(np.random.randn(2, 4, 4).astype(np.float32))

    model.get_audio_embeds = get_audio_embeds
    input_ids = mx.array([[1, AUDIO_TOKEN, 2, AUDIO_TOKEN, 3]])
    inputs_embeds = model.embed_tokens(input_ids)
    try:
        model._merge_input_embeddings(
            input_ids=input_ids,
            input_features=mx.zeros((2, 2)),
            inputs_embeds=inputs_embeds,
        )
    except ValueError as exc:
        assert "single audio stream" in str(exc)
    else:
        raise AssertionError("expected ValueError for multi-stream audio")


if __name__ == "__main__":
    test_single_batch_places_audio_and_promotes_dtype()
    test_batch_broadcasts_single_audio_stream()
    test_no_audio_tokens_preserves_dtype_and_does_not_mutate()
    test_caller_provided_inputs_embeds_is_not_mutated()
    test_mismatched_audio_token_count_raises()
    test_non_single_audio_stream_raises()
    print("All tests passed.")
