use crate::error::{TokenizerError, TokenizerResult};
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug)]
pub struct SentencePieceBackend {
    processor: sentencepiece::SentencePieceProcessor,
    /// Pre-built set of special token IDs (BOS, EOS, PAD, UNK) for O(1)
    /// lookup in the per-token generation hot path.
    ///
    /// The original implementation called `special_token_ids()` — which
    /// `collect()`s a new `Vec<u32>` — on every single token. This pre-builds
    /// the set once at load time.
    special_ids: HashSet<u32>,
}

impl SentencePieceBackend {
    pub fn from_path(root: &Path) -> TokenizerResult<Self> {
        let processor = sentencepiece::SentencePieceProcessor::open(root.join("tokenizer.model"))
            .map_err(|err| TokenizerError::BackendError(err.to_string()))?;

        // P1 fix: build HashSet once at load time.
        let special_ids = [
            processor.bos_id(),
            processor.eos_id(),
            processor.pad_id(),
            Some(processor.unk_id()),
        ]
        .into_iter()
        .flatten()
        .collect();

        Ok(Self {
            processor,
            special_ids,
        })
    }

    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> TokenizerResult<Vec<u32>> {
        let mut ids: Vec<u32> = self
            .processor
            .encode(text)
            .map_err(|err| TokenizerError::BackendError(err.to_string()))?
            .into_iter()
            .map(|piece| piece.id)
            .collect();

        if add_special_tokens {
            if let Some(bos) = self.processor.bos_id() {
                ids.insert(0, bos);
            }
            if let Some(eos) = self.processor.eos_id() {
                ids.push(eos);
            }
        }

        Ok(ids)
    }

    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> TokenizerResult<String> {
        let filtered: Vec<u32> = if skip_special_tokens {
            ids.iter()
                .copied()
                .filter(|id| !self.is_special_token_id(*id))
                .collect()
        } else {
            ids.to_vec()
        };

        self.processor
            .decode_piece_ids(&filtered)
            .map_err(|err| TokenizerError::BackendError(err.to_string()))
    }

    pub fn token_to_id(
        &self,
        token: &str,
    ) -> Option<u32> {
        self.processor.piece_to_id(token).ok().flatten()
    }

    /// Returns the decoded text for a single token ID.
    ///
    /// NOTE: calls `decode_piece_ids` which runs the full SentencePiece
    /// decoding pipeline. The return value is the decoded text (e.g. `" Hello"`
    /// for the `▁Hello` piece), not the raw piece string (`"▁Hello"`). This
    /// diverges from the HuggingFace convention where `id_to_token` returns the
    /// raw vocabulary token. The `sentencepiece` crate does not expose an
    /// `id_to_piece` method, so this is the closest available approximation.
    pub fn id_to_token(
        &self,
        id: u32,
    ) -> Option<String> {
        self.processor.decode_piece_ids(&[id]).ok()
    }

    pub fn vocab_size(&self) -> usize {
        self.processor.len()
    }

    pub fn token_bytes(
        &self,
        id: u32,
    ) -> TokenizerResult<Vec<u8>> {
        self.decode(&[id], false).map(String::into_bytes)
    }

    /// O(1) check against the pre-built special-ID set.
    ///
    /// Safe to call in the per-token generation hot path.
    pub fn is_special_token_id(
        &self,
        id: u32,
    ) -> bool {
        self.special_ids.contains(&id)
    }

    pub fn special_token_ids(&self) -> Vec<u32> {
        self.special_ids.iter().copied().collect()
    }
}
