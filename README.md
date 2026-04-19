# `speechbrain-fbank`

[![Crates.io Version](https://img.shields.io/crates/v/speechbrain-fbank)](https://crates.io/crates/speechbrain-fbank)

Rust speech feature extraction crate providing SpeechBrain-style FBANKs, MFCC, and STFT/ISTFT built on top of `realfft`/`rustfft`.

## Features
- SpeechBrain-compatible FBANK extraction with configurable `n_mels`, `f_min`, `f_max`, `n_fft`, filter shape, deltas, and context windows.
- MFCC extraction with configurable frame/mel options.
- STFT/ISTFT utilities and raw frame copies for debugging.

## Quick start
Add the crate to your project (path or git as needed), then compute SpeechBrain-style FBANKs:

```rust
use speechbrain_fbank::fbank::{FbankComputer, FbankOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = FbankOptions::default();
    let mut comp = FbankComputer::new(opts.clone())?;
    let wave = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / opts.frame_opts.samp_freq).sin())
        .collect::<Vec<_>>();

    let feats = comp.compute_waveform(&wave);
    println!("{} frames x {} bins", feats.len(), feats[0].len());
    Ok(())
}
```

For frame-by-frame input, `FbankComputer::compute` remains available. The utterance-level `compute_waveform` path is the closest match to SpeechBrain because top-db clipping, deltas, and context are sequence-level operations.

## Running tests
```
cargo test --tests -- --nocapture
```

## Also see

- https://github.com/RustedBytes/kaldi-native-fbank
