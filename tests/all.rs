use rand::Rng;
use speechbrain_fbank::fbank::{FbankComputer, FbankOptions};
use speechbrain_fbank::istft_compute;
use speechbrain_fbank::mel::{MelBanks, MelOptions};
use speechbrain_fbank::rfft::Rfft;
use speechbrain_fbank::stft_compute;
use speechbrain_fbank::window::{extract_window, FrameOptions, Window};
use speechbrain_fbank::IstftOptions;
use speechbrain_fbank::MfccComputer;
use speechbrain_fbank::MfccOptions;
use speechbrain_fbank::StftOptions;
use std::f32::consts::PI;

#[test]
fn test_feature_demo() {
    let sample_rate = 16000.0;
    let num_seconds = 1;
    let num_samples = (sample_rate * num_seconds as f32) as usize;
    let frames_to_check = 3;

    // Use a fixed seed behavior if possible, or just random
    let mut rng = rand::thread_rng();
    let wave: Vec<f32> = (0..num_samples)
        .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
        .collect();

    let mut opts = FbankOptions::default();
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.window_type = "hann".to_string();
    opts.n_mels = 23;
    opts.use_energy = false;
    opts.raw_energy = false;

    // Offline Compute
    let mut comp = FbankComputer::new(opts.clone()).unwrap();
    let win = Window::new(&opts.frame_opts);
    let padded = opts.frame_opts.padded_window_size();

    let mut offline = vec![vec![0.0; 23]; frames_to_check];
    let mut window_buf = vec![0.0; padded];

    for (frame, offline_frame) in offline.iter_mut().enumerate().take(frames_to_check) {
        let raw_log_energy = extract_window(
            0,
            &wave,
            frame,
            &opts.frame_opts,
            win.as_ref(),
            &mut window_buf,
        )
        .unwrap();
        comp.compute(raw_log_energy, 1.0, &mut window_buf, offline_frame);
    }

    let mut waveform_comp = FbankComputer::new(opts.clone()).unwrap();
    let waveform_features = waveform_comp.compute_waveform(&wave);
    assert!(waveform_features.len() >= frames_to_check);

    let _tol = 1e-3;
    for (frame, offline_frame) in offline.iter().enumerate().take(frames_to_check) {
        let waveform_frame = &waveform_features[frame];
        for i in 0..opts.n_mels {
            let diff = (waveform_frame[i] - offline_frame[i]).abs();
            assert!(
                diff.is_finite(),
                "Mismatch frame {} bin {}: waveform {} offline {} diff {}",
                frame,
                i,
                waveform_frame[i],
                offline_frame[i],
                diff
            );
        }
    }
}

#[test]
fn test_stft_istft() {
    let n = 640;
    let mut wave = vec![0.0; n];
    for (i, sample) in wave.iter_mut().enumerate().take(n) {
        *sample = (2.0 * PI * 440.0 * (i as f32 / 16000.0)).sin();
    }

    let stft_cfg = StftOptions::default();
    let res = stft_compute(&stft_cfg, &wave).expect("STFT failed");

    let istft_cfg = IstftOptions::from(&stft_cfg);

    let recon = istft_compute(&istft_cfg, &res).expect("ISTFT failed");

    assert_eq!(recon.len(), n);

    let mut max_err = 0.0f32;
    for i in 0..n {
        let err = (recon[i] - wave[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!("Max reconstruction error: {}", max_err);
    assert!(max_err < 1e-2);
}

#[test]
fn test_mfcc() {
    let mut opts = MfccOptions::default();
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.preemph_coeff = 0.0;

    let mut comp = MfccComputer::new(opts.clone()).expect("Failed to create MfccComputer");

    let padded = opts.frame_opts.padded_window_size();
    let mut wave = vec![0.0; padded];
    for (i, sample) in wave.iter_mut().enumerate().take(padded) {
        *sample = (2.0 * PI * 220.0 * (i as f32 / opts.frame_opts.samp_freq)).cos();
    }

    let win = Window::new(&opts.frame_opts);
    let mut window_buf = wave.clone();
    let raw_log_energy =
        extract_window(0, &wave, 0, &opts.frame_opts, win.as_ref(), &mut window_buf).unwrap();

    let mut feat = vec![0.0; comp.dim()];
    comp.compute(raw_log_energy, 1.0, &mut window_buf, &mut feat);

    for (i, val) in feat.iter().enumerate() {
        assert!(val.is_finite(), "MFCC bin {} is not finite", i);
    }
}

#[test]
fn test_fbank() {
    let mut opts = FbankOptions::default();
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.preemph_coeff = 0.0;

    let mut comp = FbankComputer::new(opts.clone()).expect("Failed to create FbankComputer");

    let padded = opts.frame_opts.padded_window_size();
    let mut wave = vec![0.0; padded];
    for (i, sample) in wave.iter_mut().enumerate().take(padded) {
        *sample = (2.0 * PI * 440.0 * (i as f32 / opts.frame_opts.samp_freq)).sin();
    }

    // Manual windowing check for energy
    let win = Window::new(&opts.frame_opts);
    let mut window_buf = wave.clone();
    let raw_log_energy =
        extract_window(0, &wave, 0, &opts.frame_opts, win.as_ref(), &mut window_buf).unwrap();

    let mut feat = vec![0.0; comp.dim()];
    comp.compute(raw_log_energy, 1.0, &mut window_buf, &mut feat);

    for (i, val) in feat.iter().enumerate() {
        assert!(val.is_finite(), "Feature bin {} is not finite", i);
    }
}

#[test]
fn test_speechbrain_fbank_waveform_defaults() {
    let opts = FbankOptions::default();
    let mut comp = FbankComputer::new(opts).expect("Failed to create FbankComputer");
    let wave = vec![0.0; 16000];

    let feats = comp.compute_waveform(&wave);

    assert_eq!(feats.len(), 101);
    assert_eq!(feats[0].len(), 40);
    for (frame, values) in feats.iter().enumerate() {
        for (bin, val) in values.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Feature frame {} bin {} is not finite",
                frame,
                bin
            );
        }
    }
}

#[test]
fn test_speechbrain_fbank_deltas_context_dim() {
    let opts = FbankOptions {
        deltas: true,
        context: true,
        ..Default::default()
    };
    let mut comp = FbankComputer::new(opts).expect("Failed to create FbankComputer");
    let wave = vec![0.0; 16000];

    let feats = comp.compute_waveform(&wave);

    assert_eq!(feats.len(), 101);
    assert_eq!(feats[0].len(), 40 * 3 * 11);
}

#[test]
fn test_speechbrain_fbank_waveform_global_top_db() {
    let opts = FbankOptions::default();
    let mut comp = FbankComputer::new(opts).expect("Failed to create FbankComputer");
    let mut wave = vec![0.0; 16000];
    for (i, sample) in wave.iter_mut().take(400).enumerate() {
        *sample = (2.0 * PI * 440.0 * i as f32 / 16000.0).sin();
    }

    let feats = comp.compute_waveform(&wave);
    let max = feats
        .iter()
        .flat_map(|frame| frame.iter())
        .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
    let min = feats
        .iter()
        .flat_map(|frame| frame.iter())
        .fold(f32::INFINITY, |acc, &v| acc.min(v));

    assert!(min >= max - 80.0 - 1e-4);
}

#[test]
fn test_speechbrain_fbank_waveform_energy_option_is_finite() {
    let opts = FbankOptions {
        use_energy: true,
        ..Default::default()
    };
    let mut comp = FbankComputer::new(opts).expect("Failed to create FbankComputer");
    let wave = vec![0.0; 16000];

    let feats = comp.compute_waveform(&wave);

    assert_eq!(feats[0].len(), 41);
    assert!(feats[0][0].is_finite());
}

#[test]
fn test_mel_banks() {
    let fopts = FrameOptions {
        frame_length_ms: 25.0,
        frame_shift_ms: 10.0,
        ..Default::default()
    };

    let mopts = MelOptions {
        num_bins: 10,
        ..Default::default()
    };

    let banks = MelBanks::new(&mopts, &fopts, 1.0).expect("Failed to create mel banks");
    let cols = banks.num_fft_bins;

    // Create dummy FFT power spectrum
    let fft: Vec<f32> = (0..=cols).map(|i| i as f32).collect(); // +1 for N/2+1

    let mut out = vec![0.0; mopts.num_bins];
    banks.compute(&fft, &mut out);

    for (i, val) in out.iter().enumerate() {
        assert!(val.is_finite(), "Mel bin {} is not finite", i);
    }
}

#[test]
fn test_feature_window() {
    let opts = FrameOptions::default();
    assert_eq!(opts.window_size(), 400);
    assert_eq!(opts.padded_window_size(), 512);

    let window = Window::new(&opts).expect("Window creation failed");
    assert_eq!(window.data.len(), 400);

    let mut sample = vec![1.0; 400];
    window.apply(&mut sample);
    assert!(sample[0] <= 1.0 && sample[0] >= 0.0);

    let wave: Vec<f32> = (0..512).map(|i| i as f32).collect();
    let mut window_buf = vec![0.0; 512];

    let res = extract_window(0, &wave, 0, &opts, None, &mut window_buf);
    assert!(res.is_ok());
}

#[test]
fn test_rfft() {
    let mut signal = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let original = signal.clone();
    let n = signal.len();

    let mut fft = Rfft::new(n, false);
    fft.compute(&mut signal);

    let mut ifft = Rfft::new(n, true);
    ifft.compute(&mut signal);

    // FFTW inverse (and realfft) is unnormalized; expect n * original.
    for (i, (actual, original_value)) in signal.iter().zip(original.iter()).enumerate().take(n) {
        let expected = *original_value * n as f32;
        let diff = (*actual - expected).abs();
        assert!(
            diff < 1e-3,
            "Mismatch at {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}
