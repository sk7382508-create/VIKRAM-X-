# VIKRAM-X

**Adaptive battlefield communication audio enhancement — SIH proof of concept**

VIKRAM-X is a Streamlit demo that analyzes a noisy WAV recording, detects short impulsive events, routes the audio through continuous-noise and transient-suppression paths, and blends their outputs. The interface presents the input and enhanced audio, routing timeline, signal plots, and reference-based quality metrics.

> This is a research/demo prototype for prerecorded audio. It is not a field-tested defence system, and its continuous-noise processor is a signal-processing baseline rather than a trained neural speech-enhancement model.

## Processing flow

```text
Noisy WAV
   │
   ├── Feature extraction ── Traffic Director ── frame-level routing
   │                                                │
   ├── Continuous-noise spectral suppression ──────┤
   ├── Impulsive-event DSP limiter ─────────────────┤
   │                                                ▼
   └────────────────────────────────────────── Adaptive fusion
                                                    │
                                              Enhanced WAV
```

- **Continuous interference:** a spectral-mask baseline suppresses persistent rotor, engine, and wind noise.
- **Impulsive events:** a fast DSP path attenuates detected transients, such as the demo's gunfire-like impulses.
- **Adaptive fusion:** a time-varying weight favors the continuous-noise path during steady interference and the DSP path during detected impulses.
- **Evaluation:** when a clean reference is available, the app computes SI-SNR and STOI before and after processing. These are measured results, not generated claims.

## Demo scenarios

Choose a scenario from the Streamlit sidebar:

| Scenario | Description |
| --- | --- |
| Battlefield mixed | Speech with synthetic continuous helicopter/engine noise and impulses at approximately 5 s and 8 s. Includes −5 dB, −10 dB, and 0 dB target SNR options. |
| Helicopter dominant | Speech with continuous helicopter noise and no impulses. |
| Impulsive gunfire | Speech with several synthetic impulsive events. |
| Custom WAV | Upload a WAV recording for offline processing. Mono 16 kHz audio is recommended; other sample rates are converted. |

The supplied audio is synthetic/demo material. It should not be presented as a real battlefield recording.

## Run locally

Requirements: Python 3.10 or newer and the packages in `requirements.txt`.

### Windows PowerShell

```powershell
cd path\to\vikram-x-demo
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS or Linux

```bash
cd path/to/vikram-x-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local address, usually <http://localhost:8501>.

## Using the demo

1. Select **MODE 3: Battlefield Mixed** for the main showcase.
2. Leave the default **Extreme (−5 dB SNR)** setting for the primary demonstration.
3. Play the raw recording, then select **EXECUTE VIKRAM-X DUAL-PATH ENHANCEMENT**.
4. Compare the before/after players and inspect the routing timeline, spectrograms, fusion-weight plot, and objective metrics.
5. Use the download button to save the enhanced WAV.

For custom recordings, select **Custom Upload (.WAV)** in the sidebar. Reference-based metrics are only available for the bundled demo/reference pairing; they are omitted for custom uploads.

## Project layout

```text
app.py                  Streamlit user interface and processing orchestration
src/audio.py            WAV loading, resampling, and synthetic battlefield audio
src/features.py         Frame-level temporal and spectral features
src/traffic_director.py Impulse detection and path routing
src/ai_path.py          Continuous-noise spectral suppression baseline
src/impulse_path.py     Transient DSP suppression
src/fusion.py           Adaptive output fusion
src/reconstruction.py   Safe WAV output
src/metrics.py          SI-SNR, STOI, and optional PESQ evaluation
data/                   Demo audio, clean speech reference, and generated scenarios
outputs/                Enhanced audio and demo outputs
```

## Regenerating the demo audio

The repository includes the generated WAV scenarios. To regenerate them, keep `data/speech/clean_speech.wav` in place and run:

```bash
python generate_samples.py
```

The clean speech reference is required to generate the scenarios and calculate objective metrics. The generator does not download a replacement reference automatically.

## Evaluation note

Results depend on the recording and suppression settings. On the bundled −5 dB battlefield sample, a local run measured an SI-SNR change from **−5.43 dB** to **−3.79 dB** and STOI of **0.709** before and after enhancement. This is a modest improvement, not complete noise removal. PESQ may be unavailable depending on the platform's native runtime support.

## Limitations and roadmap

- Current continuous-noise enhancement is a deterministic spectral-processing baseline. No DCCRN or other trained neural checkpoint is included.
- The demo processes prerecorded/uploaded files; it does not provide live radio, microphone-array, beamforming, or embedded-device processing.
- Synthetic scenario performance does not establish performance on real operational recordings.
- A trained speech-enhancement model, real noise datasets, and deployment-specific latency and safety validation would be separate future work.
