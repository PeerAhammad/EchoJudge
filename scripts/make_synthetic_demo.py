"""Writes SYNTHETIC test tones (harmonic bursts, not speech) to data/synthetic_demo/ for smoke tests only.
Never use these as dataset evidence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import soundfile as sf
from synth import SR, baseline, synth_speech

out = Path(__file__).resolve().parent.parent / "data" / "synthetic_demo"; out.mkdir(parents=True, exist_ok=True)
sf.write(out / "reference.wav", baseline(seed=1)[0], SR)
sf.write(out / "participant_rushed_middle.wav", synth_speech([(18, 4.5), (36, 7.2), (36, 7.2), (18, 4.5)], [0.5, 0.5, 0.5], seed=3)[0], SR)
sf.write(out / "participant_flat_pitch.wav", baseline(seed=6, pitch_st=0.3)[0], SR)
print("wrote", out)
