**What the floor says about the paper's own models (C6).** At the paper's context, on the same 4095-byte windows:

* **Text.** Every *single* released self-play learner, up to the largest (24.4M), predicts DCLM text **worse than an untrained byte counter**. Only ensembles of up to 10 independently trained seeds get below it. Averaging seeds is worth over a bit per byte here, which says the single models are badly calibrated on text, not that they lack a unigram model.
* **DNA.** Every single released learner is worse than the counter at every size; the uniform-prior ensemble (no adaptation at all) also beats it.
* **Images, speech, melody.** The released learners from 1M up clearly beat the counter: real, beyond-counting structure transfers here.

This does not contradict the paper's scaling claim (bits/byte does fall with compute), but it changes what "transfer" means per modality, and it is the reference every number in Part 10 is read against.
