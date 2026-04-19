use crate::rfft::Rfft;
use crate::utils::{compute_power_spectrum_inplace, inner_product, log_energy};
use crate::window::{FrameOptions, Window};

#[derive(Clone, Debug)]
pub struct FbankOptions {
    pub frame_opts: FrameOptions,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: Option<f32>,
    pub n_fft: usize,
    pub filter_shape: String,
    pub deltas: bool,
    pub context: bool,
    pub requires_grad: bool,
    pub param_change_factor: f32,
    pub param_rand_factor: f32,
    pub left_frames: usize,
    pub right_frames: usize,
    pub amin: f32,
    pub ref_value: f32,
    pub top_db: f32,
    pub use_energy: bool,
    pub raw_energy: bool,
    pub htk_compat: bool,
    pub energy_floor: f32,
    pub use_log_fbank: bool,
    pub use_power: bool,
}

impl Default for FbankOptions {
    fn default() -> Self {
        let frame_opts = FrameOptions {
            dither: 0.0,
            preemph_coeff: 0.0,
            remove_dc_offset: false,
            window_type: "hamming_periodic".to_string(),
            round_to_power_of_two: false,
            snip_edges: false,
            ..Default::default()
        };

        Self {
            frame_opts,
            n_mels: 40,
            f_min: 0.0,
            f_max: None,
            n_fft: 400,
            filter_shape: "triangular".to_string(),
            deltas: false,
            context: false,
            requires_grad: false,
            param_change_factor: 1.0,
            param_rand_factor: 0.0,
            left_frames: 5,
            right_frames: 5,
            amin: 1e-10,
            ref_value: 1.0,
            top_db: 80.0,
            use_energy: false,
            raw_energy: true,
            htk_compat: false,
            energy_floor: 0.0,
            use_log_fbank: true,
            use_power: true,
        }
    }
}

pub struct FbankComputer {
    pub opts: FbankOptions,
    rfft: Rfft,
    fbank_matrix: Vec<f32>,
    n_stft: usize,
    log_energy_floor: f32,
}

impl FbankComputer {
    pub fn new(opts: FbankOptions) -> Result<Self, String> {
        let n_fft = opts.n_fft;
        if n_fft == 0 {
            return Err("n_fft must be greater than zero".to_string());
        }
        if opts.frame_opts.window_size() > n_fft {
            return Err("SpeechBrain Fbank requires win_length <= n_fft".to_string());
        }
        let rfft = Rfft::new(n_fft, false);
        let (fbank_matrix, n_stft) = create_speechbrain_fbank_matrix(&opts)?;

        let log_energy_floor = if opts.energy_floor > 0.0 {
            opts.energy_floor.ln()
        } else {
            -1e10
        };

        Ok(Self {
            opts,
            rfft,
            fbank_matrix,
            n_stft,
            log_energy_floor,
        })
    }

    pub fn dim(&self) -> usize {
        let mut dim = self.base_dim();
        if self.opts.deltas {
            dim *= 3;
        }
        if self.opts.context {
            dim *= self.opts.left_frames + self.opts.right_frames + 1;
        }
        dim
    }

    fn base_dim(&self) -> usize {
        self.opts.n_mels + usize::from(self.opts.use_energy)
    }

    pub fn compute(
        &mut self,
        mut signal_raw_log_energy: f32,
        _vtln_warp: f32,
        signal_frame: &mut [f32],
        feature: &mut [f32],
    ) {
        assert!(
            feature.len() >= self.base_dim(),
            "feature buffer is smaller than the base fbank dimension"
        );
        assert!(
            signal_frame.len() >= self.opts.n_fft,
            "signal frame must have at least n_fft samples"
        );

        if self.opts.use_energy && !self.opts.raw_energy {
            let energy = inner_product(signal_frame, signal_frame);
            signal_raw_log_energy = log_energy(energy);
        }

        self.rfft.compute(signal_frame);
        compute_power_spectrum_inplace(signal_frame);

        if !self.opts.use_power {
            for x in signal_frame.iter_mut().take(self.n_stft) {
                *x = x.sqrt();
            }
        }

        let mel_offset = if self.opts.use_energy && !self.opts.htk_compat {
            1
        } else {
            0
        };
        self.compute_linear_fbanks(
            &signal_frame[..self.n_stft],
            &mut feature[mel_offset..mel_offset + self.opts.n_mels],
        );

        if self.opts.use_log_fbank {
            amplitude_to_db(
                &mut feature[mel_offset..mel_offset + self.opts.n_mels],
                self.opts.amin,
                self.opts.ref_value,
                Some(self.opts.top_db),
                if self.opts.use_power { 10.0 } else { 20.0 },
            );
        }

        if self.opts.use_energy {
            if self.opts.energy_floor > 0.0 && signal_raw_log_energy < self.log_energy_floor {
                signal_raw_log_energy = self.log_energy_floor;
            }
            let energy_index = if self.opts.htk_compat {
                self.opts.n_mels
            } else {
                0
            };
            feature[energy_index] = signal_raw_log_energy;
        }
    }

    pub fn compute_waveform(&mut self, waveform: &[f32]) -> Vec<Vec<f32>> {
        let win_length = self.opts.frame_opts.window_size();
        let hop_length = self.opts.frame_opts.window_shift();
        let pad = self.opts.n_fft / 2;
        let num_frames = if hop_length == 0 {
            0
        } else {
            waveform.len() / hop_length + 1
        };
        let window = Window::new(&self.opts.frame_opts).expect("invalid fbank window");
        let mut frame = vec![0.0; self.opts.n_fft];
        let mut feats = Vec::with_capacity(num_frames);
        let window_offset = (self.opts.n_fft - win_length) / 2;
        let mel_offset = if self.opts.use_energy && !self.opts.htk_compat {
            1
        } else {
            0
        };

        for frame_index in 0..num_frames {
            frame.fill(0.0);
            let start = frame_index as isize * hop_length as isize - pad as isize;
            let mut raw_energy = 0.0;
            for (i, &window_value) in window.data.iter().enumerate().take(win_length) {
                let fft_index = window_offset + i;
                let sample_index = start + fft_index as isize;
                if sample_index >= 0 {
                    if let Some(sample) = waveform.get(sample_index as usize) {
                        raw_energy += sample * sample;
                        frame[fft_index] = *sample * window_value;
                    }
                }
            }

            let mut feat = vec![0.0; self.base_dim()];
            self.compute_spectrum(
                &mut frame,
                &mut feat[mel_offset..mel_offset + self.opts.n_mels],
            );
            if self.opts.use_log_fbank {
                amplitude_to_db(
                    &mut feat[mel_offset..mel_offset + self.opts.n_mels],
                    self.opts.amin,
                    self.opts.ref_value,
                    None,
                    if self.opts.use_power { 10.0 } else { 20.0 },
                );
            }
            if self.opts.use_energy {
                let mut raw_log_energy = log_energy(raw_energy);
                if self.opts.energy_floor > 0.0 && raw_log_energy < self.log_energy_floor {
                    raw_log_energy = self.log_energy_floor;
                }
                let energy_index = if self.opts.htk_compat {
                    self.opts.n_mels
                } else {
                    0
                };
                feat[energy_index] = raw_log_energy;
            }
            feats.push(feat);
        }

        if self.opts.use_log_fbank && !feats.is_empty() {
            apply_top_db(&mut feats, mel_offset, self.opts.n_mels, self.opts.top_db);
        }
        if self.opts.deltas {
            append_deltas(&mut feats, self.base_dim());
        }
        if self.opts.context {
            feats = apply_context(feats, self.opts.left_frames, self.opts.right_frames);
        }

        feats
    }

    fn compute_linear_fbanks(&self, spectrum: &[f32], out: &mut [f32]) {
        debug_assert_eq!(spectrum.len(), self.n_stft);
        debug_assert_eq!(out.len(), self.opts.n_mels);

        for (mel, out_value) in out.iter_mut().enumerate().take(self.opts.n_mels) {
            let mut sum = 0.0;
            for (freq_bin, value) in spectrum.iter().enumerate().take(self.n_stft) {
                sum += value * self.fbank_matrix[freq_bin * self.opts.n_mels + mel];
            }
            *out_value = sum;
        }
    }

    fn compute_spectrum(&mut self, signal_frame: &mut [f32], out: &mut [f32]) {
        self.rfft.compute(signal_frame);
        compute_power_spectrum_inplace(signal_frame);

        if !self.opts.use_power {
            for x in signal_frame.iter_mut().take(self.n_stft) {
                *x = x.sqrt();
            }
        }

        self.compute_linear_fbanks(&signal_frame[..self.n_stft], out);
    }
}

fn create_speechbrain_fbank_matrix(opts: &FbankOptions) -> Result<(Vec<f32>, usize), String> {
    let sample_rate = opts.frame_opts.samp_freq;
    let f_max = opts.f_max.unwrap_or(sample_rate / 2.0);
    if opts.f_min >= f_max {
        return Err(format!("Require f_min: {} < f_max: {}", opts.f_min, f_max));
    }
    if opts.n_mels == 0 {
        return Err("n_mels must be greater than zero".to_string());
    }
    if opts.filter_shape != "triangular"
        && opts.filter_shape != "rectangular"
        && opts.filter_shape != "gaussian"
    {
        return Err("filter_shape must be 'triangular', 'rectangular', or 'gaussian'".to_string());
    }

    let n_stft = opts.n_fft / 2 + 1;
    let mel_min = to_mel(opts.f_min);
    let mel_max = to_mel(f_max);
    let mut hz = Vec::with_capacity(opts.n_mels + 2);
    for i in 0..opts.n_mels + 2 {
        let mel = mel_min + (mel_max - mel_min) * i as f32 / (opts.n_mels + 1) as f32;
        hz.push(to_hz(mel));
    }

    let mut matrix = vec![0.0; n_stft * opts.n_mels];
    let denom = (n_stft - 1).max(1) as f32;
    for freq_bin in 0..n_stft {
        let freq = (sample_rate / 2.0) * freq_bin as f32 / denom;
        for mel in 0..opts.n_mels {
            let center = hz[mel + 1];
            let band = hz[mel + 1] - hz[mel];
            let weight = match opts.filter_shape.as_str() {
                "triangular" => {
                    let slope = (freq - center) / band;
                    0.0_f32.max((slope + 1.0).min(-slope + 1.0))
                }
                "rectangular" => {
                    if freq >= center - band && freq <= center + band {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => (-0.5 * ((freq - center) / (band / 2.0)).powi(2)).exp(),
            };
            matrix[freq_bin * opts.n_mels + mel] = weight;
        }
    }

    Ok((matrix, n_stft))
}

fn to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn amplitude_to_db(x: &mut [f32], amin: f32, ref_value: f32, top_db: Option<f32>, multiplier: f32) {
    let db_multiplier = multiplier * amin.max(ref_value).log10();
    let mut max_db = f32::NEG_INFINITY;
    for v in x.iter_mut() {
        *v = multiplier * v.max(amin).log10() - db_multiplier;
        max_db = max_db.max(*v);
    }
    if let Some(top_db) = top_db {
        let floor = max_db - top_db;
        for v in x.iter_mut() {
            *v = v.max(floor);
        }
    }
}

fn apply_top_db(feats: &mut [Vec<f32>], offset: usize, len: usize, top_db: f32) {
    let max_db = feats
        .iter()
        .flat_map(|frame| frame[offset..offset + len].iter())
        .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
    let floor = max_db - top_db;
    for frame in feats {
        for v in &mut frame[offset..offset + len] {
            *v = v.max(floor);
        }
    }
}

fn append_deltas(feats: &mut [Vec<f32>], input_size: usize) {
    let delta1 = compute_deltas(feats, input_size);
    let delta2 = compute_deltas(&delta1, input_size);
    for (i, frame) in feats.iter_mut().enumerate() {
        frame.extend_from_slice(&delta1[i]);
        frame.extend_from_slice(&delta2[i]);
    }
}

fn compute_deltas(feats: &[Vec<f32>], input_size: usize) -> Vec<Vec<f32>> {
    let n = 2isize;
    let denom = 10.0;
    let mut out = vec![vec![0.0; input_size]; feats.len()];
    for (t, out_frame) in out.iter_mut().enumerate().take(feats.len()) {
        for (c, out_value) in out_frame.iter_mut().enumerate().take(input_size) {
            let mut sum = 0.0;
            for offset in -n..=n {
                let src = (t as isize + offset).clamp(0, feats.len() as isize - 1) as usize;
                sum += offset as f32 * feats[src][c];
            }
            *out_value = sum / denom;
        }
    }
    out
}

fn apply_context(feats: Vec<Vec<f32>>, left_frames: usize, right_frames: usize) -> Vec<Vec<f32>> {
    if feats.is_empty() {
        return feats;
    }
    let frame_dim = feats[0].len();
    let context_len = left_frames + right_frames + 1;
    let mut out = vec![vec![0.0; frame_dim * context_len]; feats.len()];
    for (t, out_frame) in out.iter_mut().enumerate().take(feats.len()) {
        for ctx in 0..context_len {
            let src = (t as isize + ctx as isize - left_frames as isize)
                .clamp(0, feats.len() as isize - 1) as usize;
            let dst_offset = ctx * frame_dim;
            out_frame[dst_offset..dst_offset + frame_dim].copy_from_slice(&feats[src]);
        }
    }
    out
}
