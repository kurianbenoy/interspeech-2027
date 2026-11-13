# interspeech-2027

## About paper idea

I wouldn't write a pure position paper for the main Interspeech track. Interspeech papers are usually short and reviewers expect experiments. A better plan is an **evidence-backed position paper**: state a clear claim and test it on VAANI Malayalam. Check the 2027 call for papers in case there's a special session or workshop that takes position papers.

## The position

> **A single pooled WER hides who Malayalam ASR fails. Text-normalization choices can make those failures look smaller, and can even change which model looks best.**

Each paper you found supports part of this:
- **LAHAJA:** looking at Hindi results by accent and region showed worse performance for South and North-East Indian speakers, especially on named entities and technical terms.
- **AfriBED / the African bias review:** fair comparisons need speech that is comparable across speaker groups.
- **Manohar & Pillai (EMNLP 2024):** normalization in Whisper, MMS and other models artificially improved scores for Indic scripts.
- **VAANI / the noise dataset:** regional, spontaneous, noisy speech is the real-world setting.

No one is better placed than the creator of whisper-normalizer to make the normalization part of this argument.

## Experiments, in priority order

**E0 – Check the data before anything else (1–2 weeks)**
- Count transcribed hours by district, speaker and gender, and count how often the noise tags and `{English}` words appear.
- Set a rule in advance for which groups are large enough to report, e.g., at least 5 speakers and 30 minutes per district. Otherwise the conclusions will rest on one or two speakers.

**E1 – Build the evaluation set**
- Keep each speaker in only one of train or test, and balance the test set across districts.
- If the same images were shown to speakers in different districts, use those utterances as a closer-matched comparison. This is the AfriBED idea in a cheaper form.

**E2 – Test several existing ASR models without fine-tuning**
- For example: Whisper large-v3, MMS, IndicConformer and the VAANI-tuned model.
- Report both word error rate (WER) and character error rate (CER). CER matters a lot for an agglutinative language like Malayalam.

**E3 – Show how much normalization changes the results (your core contribution)**
- Score every model under 3–4 normalizers:
  1. raw text
  2. Whisper's basic normalizer
  3. a Malayalam-aware normalizer (chillu letters, ZWJ/ZWNJ characters, punctuation)
  4. different handling of noise tags and English words
- Check whether model rankings change (Kendall's τ) and whether the gaps between districts shrink or grow under each normalizer.
- Headline finding to look for: "normalizer X hides a Y-point gap between districts."

**E4 – Measure the gaps properly**
- Error rate per district, with confidence intervals from resampling whole speakers rather than single utterances.
- A regression with a separate baseline for each speaker: CER ~ district + gender + noise + duration + English-word rate. This separates real regional effects from differences in noise or English mixing.
- Report the gap between the best and worst groups, and the worst group's score, not just the averages.

**E5 – Error analysis by native speakers**
- Have native speakers label about 300 errors: named entities, English loanwords, word-joining (sandhi) or word-boundary errors, dialect words, and noise.
- This adds the language-expert grounding that Manohar & Pillai call for.

**E6 – Optional fix if time allows**
- Fine-tune one model on the same number of hours, sampled randomly versus balanced across districts, and see whether the gap shrinks.
- This turns the paper from "here is a problem" into "here is a problem and a cheap fix."

## Scope

If time is short, E0–E4 is enough for a 4-page paper. E5 makes it stronger and E6 makes it stand out. A possible title: *"Lost in Pooling and Normalization: Auditing Regional Disparities in Malayalam ASR on VAANI."*

Submission deadlines have usually been in early March, so you have roughly 3–4 months; confirm the date on the official site. One more thing: since the EMNLP paper is by Malayalam researchers working on this exact problem, it may be worth contacting them about collaborating rather than only citing them.
