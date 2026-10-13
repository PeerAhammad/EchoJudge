# Recording guide (about 1 hour)

Read each passage in `data/passages/p1.txt`, `p2.txt`, `p3.txt` (about 30-40 s each) in a quiet room, same microphone/phone
position for everything. Save into `data/raw/` with EXACTLY these names (extension .ogg, .wav or .mp3):

`<speaker>_<passage>_<type>.<ext>`   speaker = s1 (you), s2 (second person);  passage = p1, p2, p3

| type    | what to do                                                                     | label created            |
|---------|--------------------------------------------------------------------------------|--------------------------|
| `ref`   | clear, natural, careful read (the reference)                                   | none (reference)         |
| `clean` | a SECOND natural read, no tricks. Do not try to match the first read.         | none (false-positive control) |
| `rushed`| read the whole passage clearly faster, about 1.4-1.6x your normal speed        | rushed_pacing            |
| `slow`  | read the whole passage clearly slower, about 0.6-0.7x your normal speed        | slow_pacing              |
| `pauses`| normal speed, but stop silently for 1-2 s after every comma/sentence end       | excessive_pauses         |
| `flat`  | normal speed, read like a robot: as little pitch movement as you can           | monotone_delivery        |

**Speaker s1 (you), all three passages:** ref, clean, rushed, slow, pauses, flat  ->  18 files.
**Speaker s2 (a friend/family member, with their consent), passage p3 only:** `s2_p3_clean`, `s2_p3_rushed`, `s2_p3_flat`
(they are compared against YOUR `s1_p3_ref`, which tests different voices). If s2 is not available, skip them.

Rules: speak the full passage every time (same words); one take per file (do not edit); keep the file as the phone recorded it.
Passage p3 is held out for TESTING: do not tune anything on it. Optional: if a flaw covers only part of a read, note it in
`data/annotations.json` (see docs/dataset.md); otherwise labels cover the whole speech.
