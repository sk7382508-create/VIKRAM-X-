# VIKRAM-X

**Adaptive Noise Cancellation for Defence Communication**  
SIH Problem Statement: SIH26052  
Organization: DRDO — Department of Defence R&D  
Team: TEAM ZENITH

## Overview

VIKRAM-X is a software feasibility prototype for adaptive suppression of noisy speech recordings. It analyzes audio in short frames, detects impulsive events, processes continuous interference and transients through separate signal-processing paths, and fuses their outputs.

The current application processes prerecorded WAV files through a Streamlit interface. Its supplied battlefield-style samples use synthetic helicopter, engine, wind, and impulse noise. They are not field recordings.

## Implemented pipeline

```text
Noisy WAV
   ↓
Mono conversion and resampling to 16 kHz
   ↓
20 ms frames with 10 ms hop
   ↓
Acoustic feature extraction and threshold-based Traffic Director
   ├── Continuous interference → spectral suppression path
   └── Impulsive event          → fast DSP transient limiter
                    ↓
           Time-varying adaptive fusion
                    ↓
          Reconstructed enhanced WAV
```

### Processing modules

- **Feature extraction:** RMS, crest factor, kurtosis, spectral flux, spectral centroid, zero-crossing rate, and sub-band energy.
- **Traffic Director:** deterministic thresholds classify frames as continuous noise, speech-dominant, or impulsive. The current router is not a trained classifier.
- **Continuous-noise path:** temporal-percentile spectral subtraction with a gain floor. This is a signal-processing baseline, not a trained neural model or DCCRN implementation.
- **Impulsive path:** envelope limiting and targeted attenuation around detected transient frames.
- **Fusion:** sample-level smoothing of the time-varying weight blends the continuous and transient paths.
- **Reconstruction and evaluation:** WAV export and measured signal telemetry. SI-SNR and STOI are calculated only when a matching clean reference is available.

## Demonstration scenarios

The repository includes generated 16 kHz WAV scenarios:

| Scenario | Contents |
| --- | --- |
| Battlefield mixed, −5 dB target SNR | Speech, continuous synthetic helicopter/engine interference, impulses near 5 s and 8 s |
| Battlefield mixed, −10 dB target SNR | Higher-interference stress scenario |
| Battlefield mixed, 0 dB target SNR | Moderate-interference scenario |
| Helicopter dominant | Speech and continuous helicopter interference |
| Impulsive events | Speech and synthetic transient events |

The Streamlit interface displays before/after audio, routing timeline, waveform and spectrogram plots, fusion telemetry, and objective measurements. Custom WAV uploads are supported. A clean, time-aligned reference WAV enables reference-based SI-SNR and STOI; without one, the interface reports RMS, peak-level changes, and clipping percentages.

## Measured demonstration result

One local run of the bundled battlefield mixed sample produced the following reference-based measurements:

| Metric | Noisy input | Enhanced output | Change |
| --- | ---: | ---: | ---: |
| SI-SNR | −5.43 dB | −3.79 dB | +1.64 dB |
| STOI | 0.709 | 0.709 | 0.000 |
| PESQ | Not available | Not available | Not measured |

This is a single result on a synthetic demo sample. It does not establish performance on real battlefield recordings or generalize to other speakers and noise conditions. PESQ was unavailable in the tested Windows environment. Runtime latency and embedded performance have not been measured.

The SIH26052 problem statement lists targets of SNR greater than 15 dB, STOI greater than 0.85, PESQ greater than 2.5, and real-time operation. The current prototype has not demonstrated those targets.

## Run locally

Requirements: Python 3.10 or later and the packages listed in `requirements.txt`.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit serves the application at `http://localhost:8501` by default.

## Audio generation

Pre-generated scenarios are included under `data/generated/`. The generation script uses `data/speech/clean_speech.wav` as its reference input:

```bash
python generate_samples.py
```

The clean speech file is required; the script does not download a replacement. The source and redistribution license for the supplied speech reference are not documented in this repository.

## Project structure

```text
app.py                  Streamlit interface and pipeline orchestration
generate_samples.py     Synthetic demo-scenario generation
requirements.txt        Python dependencies
data/
  generated/             Mixed and isolated demo WAV files
  speech/                Clean speech reference
  helicopter/            Synthetic helicopter component
  impulse/               Synthetic impulse component
outputs/                  Processed and sample output files
src/
  audio.py                Audio loading, resampling, and synthesis
  preprocessing.py        STFT/iSTFT and framing
  features.py             Frame-level acoustic features
  traffic_director.py     Impulse detection and routing
  ai_path.py              Continuous-noise spectral suppression
  impulse_path.py         Fast transient suppression
  fusion.py               Adaptive path fusion
  reconstruction.py       Output WAV reconstruction
  metrics.py              Reference-based and signal-level metrics
```

## Scope and limitations

- The input is prerecorded audio; live radio and microphone streaming are not implemented.
- The continuous-noise processor is a deterministic spectral baseline; no neural checkpoint is included.
- Multi-microphone beamforming, spatial filtering, and embedded hardware deployment are not implemented.
- Tests use controlled synthetic mixtures. Robustness to real operational recordings remains unverified.
- The prototype is a software feasibility demonstration, not a field-tested defence system.

## References

- Hu et al., “DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement,” *Interspeech 2020*. DCCRN is a possible future enhancement direction and is not used by the current prototype.
- Westhausen and Meyer, “Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression,” *Interspeech 2020*.
- Valin, “A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement,” *IEEE MMSP 2018*.

## Team

**TEAM ZENITH** — Smart India Hackathon 2026  
**VIKRAM-X — Adaptive Noise Cancellation for Defence Communication**