**What the floor says about the paper's own models (C6).** At the paper's context, on the same 4095-byte windows:

* **Text.** Every *single* released self-play learner, up to the largest (24.4M), predicts DCLM text **worse than an untrained byte counter**. Only ensembles of up to 10 independently trained seeds get below it.
* **DNA.** Every single released learner is worse than the counter at every size; ensembles, and even the uniform-prior ensemble (no adaptation at all), beat it.
* **Images, speech, melody.** The released learners from 1M up clearly beat the counter: real, beyond-counting structure transfers here.

This does not contradict the paper's scaling claim (bits/byte does fall with compute), but it changes what "transfer" means per modality. Why do ensembles help so much? The next cells separate the two possible reasons: single models being over-confident (fixable by one temperature) versus seeds genuinely knowing different things.
