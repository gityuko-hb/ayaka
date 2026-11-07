"""Concrete tokenizers behind :class:`~ayaka.tokenizers.ports.TokenizerLike`.

``hf``       HuggingFace fast or slow tokenizers (needs ``transformers``).
``tiktoken`` OpenAI-style BPE loaded from a local vocabulary file (needs ``tiktoken``).

Third-party packages are imported lazily inside the adapters, so importing this
package costs nothing and a missing package fails with a clear, actionable error.
"""
