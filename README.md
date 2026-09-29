# 🎙️ VIKRAM-X

### AI/ML-Enabled Adaptive Noise Cancellation for Defence Communication

> **Adaptive speech enhancement for high-noise communication environments — designed to detect different interference conditions and dynamically select appropriate suppression strategies while preserving speech intelligibility.**

**Problem Statement:** SIH26052 — *AI/ML-enabled adaptive noise cancellation (ANC) system that effectively suppresses stationary, non-stationary, and impulsive defence noises while maintaining high speech intelligibility and real-time performance on embedded hardware.*
**Organization:** DRDO — Department of Defence R&D
**Category:** Hardware | **Theme:** Miscellaneous
**Team:** TEAM ZENITH


## 📑 Table of Contents

- [Why This Is Different](#-why-this-is-different)
- [Project Overview](#-project-overview)
- [Target Performance](#-target-performance-per-sih26052-problem-statement)
- [Demo](#-demo)
- [Demo vs. Full Implementation](#️-demo-vs-full-implementation)
- [Key Features](#-key-features)
- [Proposed System Architecture](#-proposed-system-architecture)
- [Core Modules](#-core-modules)
- [Validation](#-validation)
- [Performance Evaluation](#-performance-evaluation)
- [Current Status](#-current-status)
- [R&D Roadmap](#-research--development-roadmap)
- [Limitations](#️-limitations-of-the-current-demo)
- [Demo Evidence](#-demo-evidence)
- [Dataset & Audio Provenance](#-dataset--audio-provenance)
- [References](#-references)
- [Reproducibility](#-reproducibility)
- [Repository Structure](#-repository-structure)

---

---


## 🎯 Why This Is Different

Most adaptive ANC approaches either apply one suppression strategy to every
noise condition, or lean entirely on a single large neural model. VIKRAM-X
makes three deliberate engineering choices instead:

- **Threshold-based routing, not a learned classifier.** The Traffic Director uses interpretable signal features (kurtosis, crest factor, spectral flux) rather than an ML classifier. This is a speed and auditability choice, not a shortcut — routing decisions can be logged and inspected frame-by-frame during development, and the reaction path for impulsive events stays free of neural inference latency.
- **Physically-grounded synthetic data, not arbitrary noise.** Real battlefield audio (close-proximity gunfire, blast, rotor noise) isn't publicly available and can't be safely collected by a student team. Our synthetic noise generation is derived from published acoustic models — a two-component ballistic model for gunshot, the Friedlander waveform for blast overpressure, a blade-passage-frequency harmonic model for rotor noise — rather than filtered white noise. This keeps the training data physically plausible instead of just "noisy-sounding," and the domain gap to real recordings is stated explicitly, not hidden.
- **Fusion, not a hard switch.** Mixed conditions (rotor noise with a gunshot mid-transmission) are common in the real scenario this targets. A hard AI/DSP switch forces a wrong all-or-nothing choice on exactly this case; the time-varying λ(t) fusion controller blends both paths based on transient confidence and estimated SNR instead.

---

## 📌 Project Overview

Defence radio communication routinely degrades in the presence of two
acoustically distinct interference types: continuous, non-stationary
background noise (rotor, engine, wind) and sudden high-amplitude
transients (gunfire, blast, artillery). These behave differently in time
and frequency, and a single fixed suppression strategy handles one
condition at the cost of the other — an AI model tuned for continuous
noise reacts too slowly for a transient; a fast limiter tuned for
transients degrades continuous speech unnecessarily. Classical approaches
named in the problem statement (spectral subtraction, Wiener filtering,
LMS-based ANC) assume stationary noise and struggle exactly here.

VIKRAM-X is built around detecting which condition is present, frame by
frame, and routing accordingly:

```text
DETECT → CLASSIFY → SELECT → SUPPRESS → PRESERVE

```

---

## 🎯 Target Performance (per SIH26052 problem statement)

These are the official targets stated in the problem statement, not
claims about the current prototype:

| Metric  | Target                                     |
| ------- | ------------------------------------------ |
| SNR     | > 15 dB                                    |
| STOI    | > 0.85                                     |
| PESQ    | > 2.5                                      |
| Latency | Real-time, suitable for live communication |

The current feasibility demo is evaluated against these same metrics on
our own held-out test set (see Validation and Performance Evaluation
below) — no number is presented as achieved unless measured.

---

# 🎬 Demo

## What the current demo demonstrates

The current prototype is a **software feasibility demonstration** of the
core VIKRAM-X architecture, not the complete system described in the
problem statement.

It uses a controlled battlefield-style audio scenario containing:

```text
Human Speech + Helicopter / Engine Noise + Sudden Impulsive Noise

```

The input is intentionally made difficult to understand so the effect of
adaptive suppression can be demonstrated clearly.

### Demo pipeline

```text
Noisy Speech → Preprocessing → Feature Extraction → Traffic Director
                                                          │
                                        ┌─────────────────┴─────────────────┐
                                        ▼                                   ▼
                                  Continuous Noise                  Impulsive Event
                                        │                                   │
                                        ▼                                   ▼
                                 AI Enhancement                       Fast DSP
                                        │                                   │
                                        └─────────────────┬─────────────────┘
                                                           ▼
                                                        Fusion
                                                           ▼
                                                         iSTFT
                                                           ▼
                                                   Enhanced Speech

```

### Demo scenarios

1. **Continuous Noise** — Speech + Helicopter → AI Enhancement Path → Enhanced Speech
2. **Impulsive Noise** — Speech + Sudden Impulse → Fast DSP Path → Suppressed Output
3. **Mixed Battlefield Scenario (primary demo)** — Speech + Helicopter + Sudden Impulsive Events, routed adaptively through both paths and fused. This is the scenario that actually demonstrates the central claim: **the processing strategy changes according to the detected interference condition**, not a single fixed transform applied throughout.

---

# ⚠️ Demo vs. Full Implementation

The current demonstration should **not be interpreted as the complete
VIKRAM-X system**, and does not yet claim the embedded/edge deployment,
multi-microphone beamforming, or hardware integration the problem
statement's expected solution describes. It was intentionally scoped as
a **rapid feasibility prototype** validating the core software
architecture within the available development time.

### Current demo covers

Controlled noisy-audio generation · Preprocessing · STFT-based analysis ·
Acoustic feature extraction · Noise-condition detection · Adaptive path
selection · Continuous-noise enhancement · Impulsive-noise suppression ·
AI/DSP fusion · Audio reconstruction · Before/after comparison · Objective
measurements

### Intended full implementation (per problem statement's expected solution)

```text
Multi-Microphone Input → Spatial Filtering / Beamforming
        ↓
Adaptive Noise Classification
        ↓
Continuous-Noise AI Enhancement + Impulsive-Noise DSP Suppression
        ↓
Adaptive Fusion → Residual Adaptive Cancellation (LMS)
        ↓
Low-Latency Reconstruction → Real-Time Communication Output
        ↓
Edge deployment (quantization / ONNX / TensorRT) on target hardware

```

**The current demo is a feasibility demonstration, not a claim that the
complete system — including real-time embedded deployment — has already
been implemented.**

---

# 🧠 Key Features

### 1. Adaptive Traffic Director

Analyses RMS energy, crest factor, kurtosis, spectral flux, and spectral
centroid to determine the current noise condition per frame.

### 2. Continuous-Noise Enhancement

Continuous and non-stationary background noise (helicopter, engine,
vehicle, wind) is routed to an AI-based speech-enhancement path. The
long-term architecture investigates DCCRN-class complex-domain models,
matching the problem statement's call for phase-preserving,
complex-domain processing. Any substitute pretrained model used in the
current demo is explicitly identified, never presented as DCCRN.

### 3. Fast Impulsive-Noise Suppression

Sudden high-energy transients are detected via time/frequency-domain
characteristics and routed through a low-latency DSP suppression path,
targeting reaction well under the audible onset of a gunshot/blast event
(2-5 ms), without unnecessarily suppressing surrounding speech.

### 4. Adaptive Fusion

The AI and DSP paths are combined through a time-varying fusion
controller:

$$
S\_{out}(t) = \lambda(t) , S\_{AI}(t) + (1-\lambda(t)) , S\_{DSP}(t), \quad 0 \leq \lambda(t) \leq 1
$$

λ(t) shifts toward the DSP path as transient confidence rises and is
scaled conservatively at low estimated SNR — logged per frame so routing
behaviour can be inspected and visualised directly.

### 5. Speech Preservation

The objective is not simply maximizing noise reduction:

```text
NOISE SUPPRESSION + SPEECH PRESERVATION + LOW-LATENCY RESPONSE

```

False suppression of speech (misrouting a speech-dominant frame as noise)
is treated as the worst failure mode, per the risk register below.

---

# 🏗️ Proposed System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                         VIKRAM-X                              │
├──────────────────────────────────────────────────────────────┤
│  Microphone / Audio Input                                     │
│            ↓                                                  │
│  Preprocessing + Synchronization                               │
│            ↓                                                  │
│  STFT + Acoustic Feature Extraction                             │
│            ↓                                                  │
│      Traffic Director                                          │
│       ┌─────┴──────┐                                          │
│       ▼            ▼                                          │
│ Continuous      Impulsive                                      │
│       ↓            ↓                                          │
│ AI Enhancement   Fast DSP                                      │
│       └─────┬──────┘                                          │
│             ▼                                                  │
│      Adaptive Fusion λ(t)                                      │
│             ▼                                                  │
│      Residual Adaptive Processing (LMS)                        │
│             ▼                                                  │
│           iSTFT / Output                                       │
│             ▼                                                  │
│      Enhanced Communication                                    │
└──────────────────────────────────────────────────────────────┘

```

---

# 🧩 Core Modules

| # | Module                           | Responsibility                                                                                                                                                       |
| - | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Audio Input & Mixture Generation | Controlled noisy scenarios at defined SNRs from clean speech + noise sources. Values are test conditions, not claimed real-world performance.                        |
| 2 | Preprocessing                    | Sample-rate standardization, mono conversion, normalization, framing, STFT                                                                                           |
| 3 | Feature Extraction               | RMS, crest factor, kurtosis, spectral flux, spectral centroid, zero-crossing rate                                                                                    |
| 4 | Traffic Director                 | Deterministic threshold logic → SPEECH DOMINANT / CONTINUOUS NOISE / IMPULSIVE EVENT. A learned classifier is a later-stage option if thresholds prove insufficient. |
| 5 | AI Enhancement Path              | Continuous/non-stationary noise; DCCRN-class architecture under investigation; substitute models explicitly labelled in the demo                                     |
| 6 | Fast DSP Path                    | Transient detection → event identification → controlled attenuation/limiting → speech preservation                                                                   |
| 7 | Fusion Controller                | Time-varying λ(t) blend of AI and DSP outputs                                                                                                                        |
| 8 | Reconstruction                   | iSTFT back to time-domain waveform                                                                                                                                   |

---

# 📊 Validation

| Area                      | Current Demo     |
| ------------------------- | ---------------- |
| Audio preprocessing       | Implemented      |
| Noise mixture generation  | Implemented      |
| Feature extraction        | Implemented      |
| Noise-condition detection | Implemented      |
| Adaptive routing          | Implemented      |
| Continuous-noise path     | Prototype        |
| Impulsive-noise path      | Prototype        |
| Fusion                    | Prototype        |
| Audio reconstruction      | Implemented      |
| Before/after comparison   | Available        |
| Objective metrics         | Where applicable |

**Any numerical performance values shown in this repository or
demonstration represent measured experimental results only.** No target,
simulated, or illustrative number is presented as achieved real-world
performance.

---

# 📈 Performance Evaluation

### Results

| System / condition | SNR | STOI | PESQ | Latency |
| --- | ---: | ---: | ---: | ---: |
| Bundled Battlefield Mixed — raw | **−5.43 dB SI-SNR** | **0.709** | TBD | — |
| Bundled Battlefield Mixed — VIKRAM-X demo | **−3.79 dB SI-SNR** | **0.709*** | TBD | TBD |
| RNNoise baseline | TBD | TBD | TBD | TBD |
| DTLN baseline | TBD | TBD | TBD | TBD |
| VIKRAM-X — AI path only | TBD | TBD | TBD | TBD |
| VIKRAM-X — AI + impulsive path | TBD | TBD | TBD | TBD |
| VIKRAM-X — full system (+ fusion, LMS) | TBD | TBD | TBD | TBD |
| **SIH26052 target** | **> 15 dB SNR** | **> 0.85** | **> 2.5** | **Real-time** |

\* The bundled demo README reports STOI = 0.709 for the before/after evaluation;
the displayed value should be labelled precisely in the UI/report once the exact
metric computation convention is finalized. fileciteturn1file0L106-L115

The −5.43 → −3.79 dB result is a **single documented local demo measurement**,
not a generalized performance claim. Do not present it as meeting the SIH target.
All future benchmark rows should include the test configuration and measurement
methodology.

Future evaluation will additionally consider SI-SDR, speech distortion,
CPU/memory/throughput, and robustness across speakers, accents, noise
types, and unseen (OOD) noise conditions.

---

# 🚧 Current Status

### Completed / Demonstration Scope

```text
Audio Input → Feature Analysis → Noise Detection → Adaptive Routing → Noise Suppression → Fusion → Enhanced Output

```

An end-to-end proof that these components operate as a unified pipeline.

### Under Development

Advanced neural enhancement (DCCRN training/validation) · Multi-microphone
spatial filtering / beamforming · Real-time streaming · Adaptive residual
cancellation · Edge deployment (ONNX/TensorRT, quantization) · Hardware
integration (per problem statement's embedded/edge deployment
requirement) · Robustness and larger-scale dataset evaluation

---

# 🔬 Research & Development Roadmap

```text
Phase 1 — Software Feasibility:    Controlled Audio → Detection → Adaptive Routing → Suppression → Fusion
Phase 2 — Advanced Signal Proc.:   Multi-Mic Input → Spatial Filtering / Beamforming → Adaptive Cancellation
Phase 3 — Advanced AI:             Larger Dataset → Neural Classification → Robust Enhancement → OOD Testing
Phase 4 — Real-Time System:        Live Microphones → Low-Latency Processing → Real-Time Output
Phase 5 — Edge / Hardware:         Optimized Models → Edge Deployment → Embedded Pipeline → Hardware Validation

```

---

# ⚠️ Limitations of the Current Demo

- Controlled audio mixtures, not a complete real-world battlefield dataset
- Primarily software-based; real-time hardware processing not yet established
- Multi-microphone spatial processing is future development
- Neural enhancement path requires further training/validation for deployment conditions
- Robustness against unseen, highly variable noise conditions needs further experimentation
- Real-world communication-system validation remains future work
- Numerical results from the demo should not be interpreted as final field performance

These are **future engineering and validation work, not missing
documentation.**

---

# 💡 Why This Approach?

> **Different noise conditions require different suppression strategies.**

A continuous rotor or engine signal and a sudden impulsive event have
different temporal and spectral characteristics. Rather than forcing one
processing method to handle every condition, noise is routed to whichever
path is suited to its behaviour, then recombined through fusion rather
than a hard switch — this adaptive architecture is the basis for the
broader VIKRAM-X research direction.

---

# 📚 Related Work

- Hu et al. (2020). *DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement.* Interspeech 2020 — continuous-noise enhancement backbone under investigation for the AI path.
- Westhausen & Meyer (2020). *DTLN.* Interspeech 2020 — real-time enhancement baseline.
- Valin (2018). *RNNoise: A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement.* IEEE MMSP — cheap-DSP-first baseline and design precedent for two-speed processing.
- Zhu et al. (2025). *ASL FxLMS + CNN-GRU.* Sensors — convex fusion of an adaptive filter and a neural branch, closest published analogue to the λ(t) fusion controller here.

---

# 🧪 Reproducibility

```
pip install -r requirements.txt
python src/mix_dataset.py --out_dir data --n_speech 40 --n_noise_per_class 8
python app.py

```

The objective is to allow the feasibility demonstration to be reproduced
independently before progressing toward real-time and hardware
implementations.

---

# 📁 Repository Structure

```text
vikram-x/
├── src/
│   ├── audio.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── traffic_director.py
│   ├── ai_path.py
│   ├── impulse_path.py
│   ├── fusion.py
│   ├── reconstruction.py
│   └── metrics.py
├── simulation/
│   └── synth_noise.py
├── data/
│   ├── speech/
│   ├── noise/
│   └── generated/
├── models/
├── evaluation/
├── results/
│   ├── plots/
│   └── tables/
├── app.py
├── requirements.txt
├── README.md
└── LICENSE

```

---


# 🎧 Demo Evidence

The README should be judged together with the **actual audio evidence**. The repository
contains a before/after demonstration for the bundled battlefield-mixed scenario.

### Recommended primary evidence

| Evidence | Configuration | What it demonstrates |
| --- | --- | --- |
| Raw audio | Battlefield Mixed, approximately −5 dB target SNR | Severity of the starting condition |
| Enhanced audio | Same input processed by VIKRAM-X | Audible noise suppression and speech preservation |
| Routing timeline | Same run | Frame-level adaptive path selection |
| Spectrogram / waveform | Same run | Time-frequency effect of suppression |
| Objective metrics | Same clean-reference pairing | Quantitative change in speech quality |

**Important:** the audio clips and plots are evidence from the software feasibility
demo. They are not evidence of field performance on real operational recordings.

The bundled demo README records one local measurement on the −5 dB battlefield sample:
SI-SNR changed from **−5.43 dB to −3.79 dB**, while STOI was **0.709** before/after
processing. This is reported as a modest experimental result, not as achievement of
the SIH target. fileciteturn1file0L106-L115

> **Before/After Audio:** Add the repository's raw and enhanced WAV/MP3 clips here
> (or embed them through the GitHub-supported media/link format). Keep the clips
> generated from the same input so the comparison is reproducible.

### Measurement discipline

For every future performance number, record:

- test scenario and target SNR;
- clean-reference availability;
- metric definition and software/library used;
- hardware and operating environment when latency is measured;
- number of runs and aggregation method;
- whether the result is a mean, median, percentile, or single run.

Do **not** report unsupported decimal-precision latency or hardware-performance
numbers as fixed achievements. A measured distribution with the test setup is more
credible than a single highly precise number.

---

# 📚 Dataset & Audio Provenance

The current feasibility demo uses **synthetic/demo battlefield audio**, not real
battlefield recordings. The demo README explicitly identifies the supplied material
as synthetic and says it should not be presented as real battlefield audio.
fileciteturn1file0L29-L40

| Asset | Current use | Provenance / licensing status |
| --- | --- | --- |
| Clean speech reference | Reference signal and scenario generation | Repository-provided asset; verify source license before redistribution |
| Synthetic helicopter / engine noise | Continuous-noise demo scenarios | Generated for the prototype |
| Synthetic impulsive events | Impulsive/gunfire-like demo scenarios | Generated for the prototype |
| Generated mixed scenarios | End-to-end feasibility testing | Derived from the above assets |

**Release rule:** do not add a license or commercial-use claim for any external
speech/noise source until its exact source, license, attribution requirement, and
redistribution terms have been verified.

For the final SIH repository, a source-level licensing table should be added once
the actual external dataset/source list is fixed:

| Source / Dataset | Asset used | License | Commercial use | Attribution required | Link |
| --- | --- | --- | --- | --- | --- |
| *To be verified* | Speech / noise | *Verify* | *Verify* | *Verify* | *Add source* |

This avoids the credibility problem of copying a polished licensing table without
being able to substantiate each row.

---

# 📖 References

The project references established speech-enhancement and hybrid DSP literature.
These references describe related methods; they are **not claims that those models
are already deployed in the current demo**.

```bibtex
@inproceedings{hu2020dccrn,
  title     = {DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement},
  author    = {Hu, Yanxin and Liu, Yun and Lv, Shubo and Xing, Mengtao and Zhang, Shimin and Fu, Yihui and Wu, Jian and Zhang, Bihong and Xie, Lei},
  booktitle = {Interspeech 2020},
  pages     = {2472--2476},
  year      = {2020},
  doi       = {10.21437/Interspeech.2020-2537}
}

@inproceedings{westhausen2020dtln,
  title     = {Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression},
  author    = {Westhausen, Nils L. and Meyer, Bernd T.},
  booktitle = {Interspeech 2020},
  pages     = {2477--2481},
  year      = {2020},
  doi       = {10.21437/Interspeech.2020-2631}
}

@inproceedings{valin2018rnnoise,
  title     = {A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement},
  author    = {Valin, Jean-Marc},
  booktitle = {IEEE 20th International Workshop on Multimedia Signal Processing (MMSP)},
  year      = {2018},
  doi       = {10.1109/MMSP.2018.8547084}
}
```

DCCRN is the longer-term phase-aware neural enhancement direction under
investigation; DTLN and RNNoise are relevant real-time speech-enhancement
references/baselines. The current demo itself uses a deterministic spectral
suppression baseline on the continuous-noise path, as stated in the demo README.
fileciteturn1file0L82-L93

---

# 🎯 Final Note

The current VIKRAM-X demonstration is **not presented as the finished
system** described in SIH26052's expected solution. It is a deliberately
scoped **feasibility prototype** built to demonstrate that the proposed
adaptive ANC architecture can be implemented as an end-to-end software
pipeline, evaluated against the same SNR/STOI/PESQ targets the problem
statement defines.

**noise suppression → speech preservation → latency → robustness → real-time operation → edge deployment → hardware validation**

---

## Team

**TEAM ZENITH** — Smart India Hackathon 2026
**VIKRAM-X — Adaptive Noise Cancellation for Defence Communication**