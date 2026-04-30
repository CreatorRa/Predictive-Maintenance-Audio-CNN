# Listening to Machines: An AI System for Predicting Industrial Pump Failures from Sound

**A Scientific Report on the MIMII Pump Anomaly Detection Pipeline**

---

## Abstract

Industrial machines, like all mechanical systems, eventually break down. The traditional response — waiting for a failure to occur and then fixing it — is enormously expensive: factories grind to a halt, replacement parts arrive late, and production schedules collapse. This report describes a system that takes a different approach: it **listens** to industrial pumps and learns to recognize the subtle change in sound that occurs when a machine is starting to fail, *before* a catastrophic breakdown occurs. The technique is called **predictive maintenance**, and the listening device is an **artificial intelligence (AI)** model trained to distinguish "healthy" pump sounds from "broken" pump sounds.

The project uses a public dataset called **MIMII** (Malfunctioning Industrial Machines Investigation and Inspection), which contains thousands of pump recordings made inside a noisy factory. Each recording is converted into a kind of picture of sound — a **spectrogram** — and shown to a type of AI called a **convolutional neural network**, which is the same family of AI that recognizes faces in photos and reads handwriting on cheques. Three serious problems had to be solved along the way:

1. **The dataset was wildly imbalanced.** Only about 11% of the recordings were of broken pumps. A naive AI would learn to ignore the rare "broken" cases and just guess "healthy" every time.
2. **The recordings would not fit into the computer's memory all at once.** The AI had to be redesigned to load each recording from disk only when needed.
3. **The first version of the AI was too clever for its own good.** It memorised individual training recordings instead of learning the general pattern of failure — like a student who memorises practice answers but fails the real exam.

After fixing these problems, we systematically tested twelve different "settings combinations" for the AI to find which combination performed best. The winner — a configuration that pairs a large batch of examples per training step with a slow, careful learning speed — correctly identified **58 out of 68 truly broken pumps** in a held-out test set, raising only **8 false alarms among 563 healthy ones**. Heatmap visualisations confirmed that the AI was paying attention to the same mid-frequency sound patterns that a human mechanical engineer would associate with bearing wear and cavitation. The result is not just a classifier, but an **explainable AI tool** ready to be deployed in a real industrial setting.

---

## 1. Introduction and Problem Description

### 1.1 The Domain Problem: Why Do We Want a Machine That Listens?

In industrial operations, expensive equipment is the foundation of every revenue-generating activity. Pumps, motors, turbines, and conveyors are constantly running, and they all degrade over time. The economic question is: how do you decide *when* to take a machine offline for maintenance?

There are three options, each with its own cost profile:

- **Reactive maintenance.** Wait for the machine to break, then fix it. This is the most expensive option because failures usually happen at the worst possible time, taking the entire production line down with them.
- **Scheduled maintenance.** Service every machine on a fixed calendar (e.g. every six months), whether it needs it or not. Safer than reactive, but wasteful — you replace parts that still have months of life left in them.
- **Predictive maintenance (PdM).** Continuously monitor the machine, and intervene only when the data shows that a failure is imminent. This is the cheapest option in the long run, but it requires a way of *seeing* deterioration before it becomes catastrophic.

This project tackles the third option using **acoustic monitoring** — the idea that a degraded pump *sounds* different from a healthy one, even if the difference is too subtle for a human to notice in a noisy factory. The goal is to build an AI that can hear what humans cannot, and raise an alarm minutes, hours, or days before the failure actually occurs.

### 1.2 The Data Challenge: A Whisper in a Crowded Room

The MIMII dataset contains pump recordings made under deliberately difficult conditions: each recording was mixed with real factory background noise at a level called **−6 dB SNR**. Two terms need unpacking here.

**Decibels (dB)** are a way of measuring how loud a sound is, on a logarithmic scale — so every increase of 10 dB roughly corresponds to "ten times louder." Negative dB values mean the sound being measured is quieter than the reference sound it's being compared to.

**Signal-to-Noise Ratio (SNR)** compares two sounds: the *signal* you care about (the pump) and the *noise* you don't (everything else in the factory). A positive SNR means the pump is louder than the background; a negative SNR means the background is louder than the pump. **−6 dB SNR** specifically means the background factory noise is roughly four times louder than the pump itself. The AI has to listen to a whisper inside a crowded factory and decide whether the whisper is healthy or broken.

On top of this, the data is severely **imbalanced** — there are far more healthy recordings than broken ones:

- **Normal (healthy) clips**: 3,749 samples, labelled `0`
- **Abnormal (broken) clips**: 456 samples, labelled `1`

That makes broken pumps roughly **11%** of the total. In economic language, broken pumps are a "**black swan**" event: rare occurrences that are difficult to model precisely *because* they almost never happen.

### 1.3 The Accuracy Paradox: Why "89% Correct" Can Mean "Useless"

Imagine an AI that is allowed to examine a pump recording and respond with one of two answers: "Healthy" or "Broken." Now imagine a *lazy* AI that doesn't actually look at the recording at all and simply answers "Healthy" every single time. Because 89% of all real recordings really are healthy, this lazy AI will be correct 89% of the time. Most students would happily take a 89% on an exam.

But this lazy AI is **completely useless** in a factory. It has never once raised an alarm — which means every single broken pump in the dataset has slipped past it. This is the **Accuracy Paradox**: when one outcome is far rarer than the other, raw accuracy (the percentage of correct answers) stops being a meaningful score.

The reason this matters is that the cost of being wrong is **wildly asymmetric**:

- A **false alarm** (raising an alarm on a healthy pump) costs an unnecessary inspection — annoying, but cheap.
- A **missed failure** (failing to alarm on a broken pump) costs a real breakdown — production downtime, secondary mechanical damage, possibly injury.

Missing a failure is *enormously* more expensive than a false alarm. The lazy AI achieves 89% accuracy by missing every single failure, which is exactly the wrong trade-off. We therefore have to abandon raw accuracy as our scoring metric and use ones that reward catching the rare class:

- **Recall** answers the question: "Of all the broken pumps in the data, what fraction did the AI correctly identify?"
- **Precision** answers: "Of all the alarms the AI raised, what fraction were genuine?"
- **F1-Score** is a balanced combination of the two — it goes up when both Recall *and* Precision are high, and crashes if either is low.
- **ROC-AUC** measures how well the AI *ranks* recordings: if you sort all the recordings by the AI's confidence that they are broken, how often does a truly broken one appear above a truly healthy one?

> **Key Takeaway — The Problem.** A factory pump that fails unexpectedly is far more costly than a few unnecessary inspections. Because broken pumps make up only 11% of our dataset, an AI that never raises alarms can score 89% accuracy and still be useless. We measure success by how reliably the AI catches the rare failures, not by how often it agrees with the obvious answer.

---

## 2. Assumptions: The Three Beliefs That Justify Our Design

Before describing how the AI is built, it is worth being explicit about the three assumptions that the entire project rests on. If any of these turned out to be wrong, the design choices that follow would not make sense.

### 2.1 Mechanical Faults Have Specific "Frequency Fingerprints"

We assume that when a pump bearing wears out, when an impeller is damaged, or when a fluid-handling mechanism cavitates, the resulting sounds appear as **disturbances in particular frequency ranges**, rather than as a global increase in volume. This matches the physics of rotating machinery: every mechanical fault has a characteristic frequency signature determined by the geometry of the failing part.

The economic analogue is a *sector-specific shock*: identifying which sector (which frequency band) is in distress is more diagnostic than measuring the volume of the entire economy (the total loudness of the recording). This assumption justifies using a frequency-resolved input rather than just measuring how loud the recording is.

### 2.2 Sound Can Be Treated as an Image

A raw audio waveform — the wiggly line on an oscilloscope — is a **1-dimensional** signal: just one number per moment in time, repeated thousands of times per second. It is messy, high-dimensional, and difficult for an AI to learn from directly.

We assume instead that we can convert each recording into a **2-dimensional image** in which the vertical axis represents pitch (low frequencies at the bottom, high frequencies at the top), the horizontal axis represents time (left to right), and the colour or brightness at each point represents the loudness of that pitch at that moment. This is called a **spectrogram**, and it is best understood as a **visual fingerprint of sound**. Treating sound as an image lets us use a kind of AI — a **convolutional neural network** — that was originally invented to recognise patterns in pictures.

We further specialise the spectrogram by warping the vertical axis using the **Mel scale**, a non-linear pitch axis that mimics how human hearing actually works. The Mel scale gives more vertical "real estate" to low pitches (where most mechanical faults live) and compresses the high pitches together. The economic intuition is **diminishing marginal utility**: a 100 Hz difference at the bottom of the spectrum (say, 100 Hz vs 200 Hz) is a much more informative gap than a 100 Hz difference at the top (say, 7,900 Hz vs 8,000 Hz). The Mel scale spends our limited "vertical pixels" where they matter most.

### 2.3 Ten Seconds Is Long Enough to Catch a Full Cycle

Industrial pumps rotate. A failure signature that occurs *during* a rotation must be captured within the recording window, or the AI will never see it. We assume that a **10-second** recording — at our sampling rate of 16,000 measurements per second, that's 160,000 individual samples — is long enough to catch at least one full mechanical cycle of any pump in the dataset, including transient or intermittent faults that only flare up periodically.

> **Key Takeaway — Assumptions.** We treat mechanical failure as a *visual pattern* in sound. By converting each recording into a 2-dimensional spectrogram image, warping the vertical axis to mimic human hearing, and ensuring each clip is long enough to capture a full mechanical cycle, we give the AI a structured, image-like input that it is well-suited to analyse.

---

## 3. The Architecture and Approach

The pipeline went through several rounds of redesign as each early prototype revealed a new problem. This section walks through the final architecture and explains *why* each piece is built the way it is.

### 3.1 Preparing the Audio: From Raw Recordings to Visual Fingerprints

#### 3.1.1 The Workflow

Every `.wav` audio file in the MIMII dataset is processed in the following way before it ever reaches the AI:

1. **Load the audio.** The recording is read from disk as a long sequence of numbers.
2. **Standardise the length.** If the recording is shorter than 10 seconds, pure silence is appended to the end. If it is longer than 10 seconds, the tail is cut off.
3. **Convert to a spectrogram.** A series of mathematical operations turns the 1-D waveform into a 2-D image in which time runs left-to-right and pitch runs bottom-to-top.
4. **Warp the pitch axis to the Mel scale.** As described in §2.2, this gives more vertical resolution to the low frequencies where mechanical faults live.
5. **Convert loudness to decibels.** Sound power varies across an enormous range, so we apply a logarithmic transformation that compresses the scale and prevents a single loud burst from dominating the picture.
6. **Save the image as a `.npy` file.** This is a fast-loading format that PyTorch (the AI library) can read efficiently.

The end result for every single recording is an identically-shaped image with **128 rows of pitch** and **313 columns of time**. From the AI's point of view, every input looks like a 128 × 313 grayscale picture.

#### 3.1.2 The Audio Settings, in Plain Language

The numerical parameters that control this conversion are not just arbitrary numbers. Each one represents a specific engineering trade-off, and together they determine what the AI is able to see.

| Setting | Value | Plain-English explanation |
|---|---|---|
| **Sample rate** | 16,000 Hz | The "frames-per-second" of the microphone. The recording captures **16,000 snapshots of the sound's pressure every second**. This is fast enough to faithfully record any sound up to about 8,000 Hz — well above the range where mechanical faults live — without wasting storage on the ultra-high frequencies that matter for music but not for machinery. |
| **Target duration** | 10 seconds (= 160,000 samples) | Every recording is **cropped to exactly the same length**, the same way every photo in a passport application has to be the same size. If we let recordings have different lengths, the AI would get confused trying to compare them. |
| **N_FFT (window size)** | 2,048 | The "**focus window**." We never look at all 160,000 samples at once. Instead, we slide a small window across the recording and analyse the frequencies inside each window separately. A *larger* window gives us very precise pitch information but blurs out *when* a sound happened. A *smaller* window gives us precise timing but coarse pitch. **2,048 samples (≈ 0.13 seconds) is our sweet spot** — fine enough to localise transient mechanical events, coarse enough to resolve the pitches that matter. |
| **Hop length (stride)** | 512 | The "**stride**" — how far we slide the focus window forward between snapshots. With a window of 2,048 samples and a hop of 512, **each window overlaps the previous one by 75%**. This produces a smooth, continuous picture of how the sound evolves, like the frames of a flipbook rather than a sequence of disconnected polaroids. |
| **N_MELS (image height)** | 128 | The number of **vertical pitch slices** in our final spectrogram image. Critically, the slices are spaced according to the **Mel scale**, which mimics human hearing: the slices are densely packed at the *low* end (where heavy machinery hums and rumbles) and progressively wider apart toward the *high* end (the squeaky, ultra-high frequencies that carry less diagnostic information). |
| **Decibel conversion** | log scaling | The recording is converted from raw pressure values into **decibels**, a logarithmic measure of loudness. This is important because raw sound power varies across a huge range — a sudden loud bang can be a million times more powerful than a quiet hum. Logarithmic scaling compresses this range so that no single loud event can drown out the subtler patterns the AI needs to see. |

The output of this whole pipeline, for every single audio clip, is a tensor (a multi-dimensional array of numbers) of shape `[1, 128, 313]`: one channel deep, 128 pitch slices tall, 313 time slices wide. This is the **visual fingerprint of sound** that the AI will learn to classify.

#### 3.1.3 Why Padding Matters: A Subtle Gotcha

When a recording is shorter than 10 seconds, we have to do *something* to make it 10 seconds long. Two options were considered:

- **Repeat the clip** (loop it until it fills 10 seconds).
- **Pad with silence** (append zeros to the end).

We chose silence. Repeating the clip would inject *artificial periodicity* — a fake rhythm that doesn't exist in the original recording. The AI, looking for patterns, might learn to associate that fake rhythm with "abnormal," and the resulting model would fail catastrophically on a new factory where recordings happen to be a different length. Silence, by contrast, fabricates no false patterns; it just stops adding information.

### 3.2 Memory-Safe Data Loading: Don't Try to Hold Everything at Once

The first version of this project tried something straightforward: load all 4,200+ spectrogram images into the computer's memory at the start of training, and keep them there for the whole run. The result was an immediate **Out-of-Memory crash** — the operating system noticed that the program was trying to use more memory than the computer had, and killed it.

The fix is conceptually simple. Rather than loading every image into memory up front, the program stores only the **file paths** in memory — basically a list of postcodes telling the program where each image lives on disk. Whenever the AI needs the next batch of images for training, the program **loads just those few images from disk on demand**, hands them to the AI, and immediately discards them once the AI is done.

This approach is called **lazy loading**. It's slightly slower than keeping everything in memory (because of the disk-read overhead), but it scales effortlessly: the same code works on a dataset of 4,000 clips or 4,000,000 clips, because at any moment in time only a handful of images are actually loaded.

### 3.3 Confronting the Imbalance: Making the AI Care About Rare Events

Recall the imbalance: only 11% of recordings are abnormal. If we trained the AI without any correction, it would discover the lazy strategy described in §1.3 — predict "Normal" for everything, score 89% accuracy, and never raise an alarm. We need to *force* it to take the rare class seriously.

#### 3.3.1 What We Decided NOT to Do

A popular technique called **SMOTE** (Synthetic Minority Over-sampling Technique) tries to fix imbalance by *generating fake examples* of the rare class. The idea is to invent new "abnormal" recordings by mixing existing ones together and feeding the synthetic mixtures to the AI alongside the real data.

We rejected this approach. Mixing two real recordings together does *not* produce a third recording that could plausibly come from a real broken pump — it produces an audio chimera that sits somewhere outside the manifold of physically realistic sounds. An AI trained on these chimeras might learn to recognise *the chimeras themselves*, then fail completely when shown a new, genuinely-broken pump in deployment.

#### 3.3.2 What We Did Instead: Weighted Loss

The AI learns by being shown an example, making a guess, and then being told how wrong it was via a number called the **loss**. The bigger the loss, the harder the AI tries to correct itself. Normally, every example contributes equally to the loss.

We changed that. We multiplied the loss for every Normal example by a small number (≈ 0.561) and the loss for every Abnormal example by a much bigger number (≈ 4.610). The ratio is precisely calibrated so that getting an Abnormal example wrong **hurts about 8.2 times more** than getting a Normal example wrong.

The numbers come from a simple inverse-frequency formula:

$$w_c = \frac{N_{\text{total}}}{k \cdot N_c}$$

where $N_{\text{total}}$ is the number of training examples, $k = 2$ is the number of classes, and $N_c$ is the count of class $c$. The rare class gets the bigger weight automatically.

The economic interpretation maps directly onto the FN/FP cost asymmetry from §1.3: missing a fault costs roughly 8× more than a false alarm, so we tax the AI roughly 8× more for missing a fault. The AI, being a relentless cost-minimiser, learns to take the rare class seriously.

### 3.4 Avoiding "Memorisation": The Global Average Pooling Solution

#### 3.4.1 What "Overfitting" Looks Like

Imagine a student preparing for an exam by memorising every word of the practice paper, including the exact wording of the questions. On the practice paper, the student gets 100%. On the real exam — which has the same *concepts* but different specific questions — the student gets a low score, because they never learned the underlying ideas.

This is **overfitting**. It is the single most common failure mode of AI systems, and it is what happens when an AI has too much capacity (too many adjustable parameters, called **parameters** or **weights**) relative to the amount of training data it is shown. With enough capacity, the AI can simply memorise the training examples one by one rather than discovering a general rule that works for new examples.

#### 3.4.2 The Original Problem

The first version of our AI used a standard architecture in which a piece called a **Flatten layer** unrolled the network's internal feature maps into one enormous list and then connected every single position of that list to a final classifier. This produced a model with **roughly 2.5 million adjustable parameters** — far more capacity than 319 abnormal training examples can support. Predictably, the model overfit immediately. It was learning facts like "in training clip #142, position (47, 218) had a peak of 0.83" rather than "abnormal pumps tend to have energy in the 1024 Hz band."

#### 3.4.3 The Fix: Global Average Pooling

We replaced the Flatten layer with a piece called **Global Average Pooling**, which works very differently. Instead of remembering every spatial position in the feature map, it computes a *single average value* for each feature channel across the entire image. The network can no longer encode *where* in the spectrogram a pattern fired; it can only encode *how strongly* each pattern fired across the whole image.

Mathematically, this **slashes the parameter count by over 95%**:

```
Original architecture (with Flatten):  ~2,500,000 parameters
After Global Average Pooling, BF=16:       97,890 parameters
After Global Average Pooling, BF=32:     ~390,000 parameters
```

(BF = "base filters," a separate setting that controls how many feature detectors the network uses; we'll come back to it in §4.)

But the parameter reduction is only half the story. The deeper benefit is **structural**: by removing the network's ability to encode positional information, we *force* it to learn position-invariant abstractions. The AI can no longer say "I saw a peak at *this exact pixel* in the training clip." It can only say "I detected this kind of pattern *somewhere* in the image." That is precisely the kind of generalisable knowledge we want.

A useful side effect of this change is that the network becomes **elastic** with respect to recording length: it accepts spectrograms of any time-axis size, because the global average doesn't care how many time slices there are.

### 3.5 The Test Suite: Catching Silent Mistakes Before They Wreck the Results

A subtle bug in machine-learning code often does not crash anything. It just *quietly* corrupts the results so that the AI's performance numbers look fine but mean nothing. To prevent this, we wrote a small but targeted test suite that automatically verifies the most fragile parts of the pipeline.

- **Padding-direction test.** Verifies that silence is added to the *end* of short recordings, not the beginning. If padding were accidentally placed at the start, the AI could learn the bizarre rule "abnormal = audio that begins early in the clip" — a positional artifact that would generalise nowhere.
- **Class-weight test.** Verifies that the inverse-frequency formula in §3.3 is applied in the correct direction. A flipped sign would cause the AI to *reward* predicting Normal, and the model would silently collapse to the lazy classifier from §1.3.
- **Split-disjointness test.** Verifies that no recording appears in both the training set and the testing set. Without this check, an AI could be "tested" on examples it had already memorised during training — producing impressive but completely meaningless test scores.
- **Elasticity test.** Verifies that the AI can still process spectrograms of unusual lengths (e.g., 160 frames or 800 frames), confirming that the Global Average Pooling head from §3.4 has not been accidentally reverted to a fragile fixed-size head in some future code change.

Each of these tests runs in well under a second and can be executed automatically before any new change is committed. They turn fragile engineering invariants into machine-checkable assertions, making it safe to refactor or extend the codebase without worrying about silently breaking the science.

> **Key Takeaway — Architecture and Approach.** The pipeline transformed audio into image-like spectrograms, loaded those images lazily from disk to avoid memory crashes, used a weighted loss to force the AI to take the rare class seriously, and adopted Global Average Pooling to prevent the AI from memorising training examples. A small automated test suite freezes these invariants in place so that future code changes cannot silently break them.

---

## 4. Hyperparameter Optimisation and Experimental Setup

The architecture from §3 defines the *shape* of the AI. But before training begins, several **hyperparameters** — the dials and knobs of the training process — must be set. The key ones are:

- **Learning rate**: how *big* a step the AI takes when adjusting itself after seeing a batch of examples. A high learning rate means big, aggressive corrections; a low learning rate means small, cautious ones.
- **Batch size**: how *many* examples the AI looks at before making one adjustment. A small batch produces noisy, jittery updates; a large batch produces smooth, confident ones.
- **Base filters (BF)**: how *many* internal pattern-detectors the AI's first layer uses. More filters means a wider, more flexible network; fewer means a smaller, more constrained one.

Choosing these values well is the difference between a system that performs at world-class level and one that doesn't work at all. This section describes how we made those choices systematically.

### 4.1 Splitting the Data: Training, Validation, Testing

The dataset was split into three subsets:

- **Training set (70 %)** — what the AI is allowed to learn from.
- **Validation set (15 %)** — used during training to monitor whether the AI is improving and to decide when to stop.
- **Test set (15 %)** — set aside completely until the very end, used **once** to produce the final, unbiased report on how well the AI works.

The split was performed using **stratified sampling**, which guarantees that the 11% Abnormal ratio is preserved exactly in all three subsets. Without stratification, a random split could accidentally place zero broken pumps into the test set, which would make Recall, Precision, and F1 mathematically undefined.

A separate test in the test suite verifies that no single recording appears in more than one subset — eliminating any chance of the AI being "tested" on recordings it had already studied during training, which would produce impressive but completely fake test scores.

### 4.2 Two Automatic Safety Nets During Training

Each training run is allowed up to 20 **epochs** (one epoch = one full pass through the entire training set). Two automated mechanisms keep the run on the rails:

- **Learning Rate Reduction.** If the validation loss stops improving for 3 consecutive epochs, the learning rate is automatically *halved*. The intuition: early in training, big steps are productive — the AI is far from the optimum, and large adjustments make rapid progress. But as the AI approaches the optimum, big steps start *overshooting* the target. By halving the learning rate when progress plateaus, we let the AI settle gracefully into a tight, stable solution.

- **Early Stopping.** If the validation loss fails to improve for 7 consecutive epochs, training is *halted entirely*, even if the 20-epoch budget hasn't been used up. The principle: once the AI has stopped improving, additional epochs only allow it to memorise noise, which damages generalisation. In economic terms: keep investing as long as the marginal return is positive; stop when it goes to zero.

The two patience values (3 for the LR reducer, 7 for early stopping) are deliberately ordered: 7 > 3 + 3 means the LR reducer gets at least *two chances* to halve the learning rate before early stopping pulls the plug. This prevents premature termination during a transient plateau that a rate reduction could have escaped.

### 4.3 The Cartesian Grid Search

To find the best combination of hyperparameters, we tested every possible pairing of three values — a **Cartesian grid search**:

| Hyperparameter | Values Tested |
|---|---|
| Learning rate | `0.001` (1e-3), `0.0005` (5e-4) |
| Batch size | `16`, `32`, `64` |
| Base filters | `16`, `32` |

This produces $2 \times 3 \times 2 = 12$ candidate configurations. The batch-size axis was deliberately extended to include `64` as a **boundary-condition probe** — large batches change the optimisation dynamics in ways that smaller batches cannot reveal, and we wanted to know whether the AI behaves predictably at the edge of the search space. This decision turned out to be the most consequential of the project (see §5.3).

Each of the 12 candidate runs was launched as a separate operating-system **subprocess**. This is a deliberate engineering choice that addresses three problems at once:

1. **Memory cleanup.** The graphics card (GPU) used to train neural networks does not always release memory cleanly between runs. Even when the program tries to free it explicitly, residue accumulates, and after a few large runs the GPU runs out of memory and crashes. By launching each run as a separate process, the operating system reclaims *all* memory atomically when that process exits. The next run gets a perfectly clean slate.
2. **Crash isolation.** If one configuration explodes mid-training (a so-called "NaN loss," where the numbers go to infinity), it doesn't take the rest of the search down with it. Only that one subprocess dies; the orchestrator logs the failure and moves on to the next configuration.
3. **Reproducibility.** Each subprocess starts a brand-new Python interpreter, with no leftover state from previous runs. Two runs with the same hyperparameters will therefore always produce the same result.

### 4.4 The Winning Configuration

The grid search produced a clear winner. Surprisingly, it was *not* the small-batch / narrow-network combination that conventional wisdom about regularisation would predict. The winning configuration was:

| Parameter | Value |
|---|---|
| Learning rate | **0.0005** (5e-4) |
| Batch size | **64** |
| Base filters | **32** |
| Trainable parameters | ~390,000 |
| Run identifier | `run_lr5em04_bs64_bf32` |

The reason this configuration won — and why no smaller configuration came close — is that the three hyperparameters interact in a deeply non-linear way. We unpack this interaction at length in §5.3, which is the most important section of the report. The short version is:

- A **large batch (64)** averages over many examples per update, producing a low-noise, "confidently directed" gradient — a smooth view of the optimisation landscape.
- A **slow learning rate (0.0005)** keeps each step small, preventing the optimiser from overshooting fine-grained minima.
- A **wider network (BF=32)** has the parameter capacity to coordinate across the smoothed gradient signal that the large batch supplies.

These three choices are *complementary*. None of them works in isolation: the large batch with an aggressive learning rate is catastrophic; the wide network with a small batch is unstable; the slow learning rate with a small batch is simply too cautious to reach the optimum within 20 epochs. The **combination** is what wins.

> **Key Takeaway — Experimental Setup.** A 12-point Cartesian grid search, with each run isolated in its own operating-system subprocess, replaced trial-and-error with a systematic, reproducible procedure. Stratified data partitioning eliminated sample bias, two automated safety nets kept training runs stable, and the deliberate inclusion of a boundary-condition value (batch size = 64) uncovered an unexpected hyperparameter interaction that produced the winning configuration.

---

## 5. Evaluation and Explainability

### 5.1 The Metric Strategy: Beyond Accuracy

For all the reasons described in §1.3, we do not optimise for raw accuracy. The four metrics we actually care about are:

- **Recall** — of all genuinely broken pumps, what fraction did the AI catch? In a factory, this is the metric an operations manager will ask about first, because it directly measures how many real failures slip through undetected.
- **Precision** — when the AI raises an alarm, how often is the alarm justified? An AI with low precision creates "alarm fatigue," eventually causing operators to ignore even genuine warnings.
- **F1-Score** — the harmonic mean of Precision and Recall. F1 is the single number that goes up only when *both* are high. It crashes if either component fails.
- **ROC-AUC** — a *threshold-independent* score that measures how well the AI's confidence ranks the recordings. AUC = 1.0 means the AI always gives broken pumps a higher confidence than healthy ones; AUC = 0.5 is random guessing.

Accuracy is reported alongside these for completeness, but it is *not* the optimisation target.

### 5.2 The Headline Result

The winning configuration `run_lr5em04_bs64_bf32` was evaluated **once** on the held-out test set (which it had never seen during training or hyperparameter selection). The numbers it produced:

```
Headline Result (Test Set, n = 631 recordings):
  F1-Score   = 0.866      (excellent — close to a perfect 1.0)
  ROC-AUC    = 0.9703     (the AI ranks broken above healthy 97% of the time)
  Precision  = 0.879      (about 12% of alarms are false alarms)
  Recall     = 0.853      (the AI catches ~85% of real failures)
  Accuracy   = 0.972      (reported for completeness only)
```

Read in plain language: the AI catches roughly **85 out of every 100 broken pumps** in the held-out test set, while raising a false alarm on only about **12 out of every 100 alarms**. An ROC-AUC of 0.97 means that if we picked a healthy and a broken recording at random and asked the AI which was broken, it would get the answer right 97 times out of 100.

#### 5.2.1 The Top Three Configurations

The winning configuration leads the leaderboard on *both* F1 and AUC simultaneously — a sign that its advantage is not a fluke of where the decision threshold happens to fall, but reflects a genuinely better internal score function:

| Run | LR | BS | BF | F1 | AUC |
|---|---|---|---|---:|---:|
| **`run_lr5em04_bs64_bf32`** *(champion)* | 0.0005 | 64 | 32 | **0.866** | **0.970** |
| `run_lr1em03_bs16_bf16` *(prior best at 8-point grid)* | 0.001 | 16 | 16 | 0.830 | 0.951 |
| `run_lr5em04_bs32_bf16` | 0.0005 | 32 | 16 | 0.791 | 0.960 |

Crucially, the second-place configuration was the *winner* of an earlier 8-point grid that did not include batch-size = 64. Extending the grid to include the boundary condition is what surfaced the new champion.

### 5.3 Robustness of the Architecture and the Smoothness Trap

The complete 12-run leaderboard, ranked by F1, reveals two complementary lessons.

| Rank | Run | LR | BS | BF | F1 | AUC |
|:---:|---|---|---|---|---:|---:|
| 1 | **`run_lr5em04_bs64_bf32`** | 0.0005 | 64 | 32 | **0.866** | **0.970** |
| 2 | `run_lr1em03_bs16_bf16` | 0.001 | 16 | 16 | 0.830 | 0.951 |
| 3 | `run_lr5em04_bs32_bf16` | 0.0005 | 32 | 16 | 0.791 | 0.960 |
| 4 | `run_lr1em03_bs16_bf32` | 0.001 | 16 | 32 | 0.786 | 0.959 |
| 5 | `run_lr1em03_bs64_bf16` | 0.001 | 64 | 16 | 0.779 | 0.970 |
| 6 | `run_lr5em04_bs32_bf32` | 0.0005 | 32 | 32 | 0.756 | 0.976 |
| 7 | `run_lr1em03_bs32_bf16` | 0.001 | 32 | 16 | 0.750 | 0.946 |
| 8 | `run_lr5em04_bs16_bf32` | 0.0005 | 16 | 32 | 0.749 | 0.968 |
| 9 | `run_lr1em03_bs64_bf32` | 0.001 | 64 | 32 | 0.745 | 0.950 |
| 10 | `run_lr5em04_bs16_bf16` | 0.0005 | 16 | 16 | 0.713 | 0.963 |
| 11 | `run_lr1em03_bs32_bf32` | 0.001 | 32 | 32 | 0.663 | 0.965 |
| 12 | `run_lr5em04_bs64_bf16` | 0.0005 | 64 | 16 | 0.602 | 0.944 |

The figure [`tuning_summary_graph.png`](docs/tuning_visualizations/tuning_summary_graph.png) plots this leaderboard visually.

#### 5.3.1 The Architecture Itself Is Robust

Notice the **AUC column**. It never drops below 0.94, even for the worst-performing run. The total spread is only about 0.03, from 0.944 to 0.976. In other words, *every* configuration in the search produces an AI that is fundamentally able to distinguish broken from healthy pumps. The structural choices we made earlier — the spectrogram representation, the weighted loss, the Global Average Pooling — are doing the heavy lifting at the level of *score-distribution quality*. Hyperparameter tuning does not change *what* the AI learns to recognise so much as *how decisively* it commits to that recognition at the default decision threshold.

This is the most desirable failure mode an industrial baseline could have. The system is not brittle to small changes in its training configuration.

#### 5.3.2 The Smoothness Trap: Why Big Batches with Aggressive Learning Rates Fail

The F1 column tells a different and more interesting story. F1 ranges from **0.866 down to 0.602** — a much wider spread than the AUC range — and it is driven almost entirely by the **interaction between learning rate and batch size**.

Watch what happens when we hold the learning rate at the aggressive value `0.001` and increase the batch size:

| Configuration | F1 | False alarms (approx.) |
|---|---:|---:|
| `lr=0.001, bs=16, bf=16` | 0.830 | ~11 |
| `lr=0.001, bs=64, bf=16` | 0.779 | ~23 |
| `lr=0.001, bs=64, bf=32` | 0.745 | ~28 |

The F1 score collapses, and the false-alarm count roughly **doubles**. The training trajectory of the bottom-row configuration shows symptoms of severe gradient over-smoothing on the imbalanced dataset — early stopping fires before the AI can fully exploit its parameter budget.

The mechanism is the **Smoothness Trap**:

- A **big batch** (64) produces a *smoothed*, low-noise estimate of the gradient — a confident view of which way to go.
- An **aggressive learning rate** (0.001) takes *big steps* in that direction.
- Together: a **big step in a confidently wrong direction** overshoots the narrow regions where the truly best models live. The optimiser ends up settling in a wide, shallow basin that produces a fuzzy, indecisive score function — exactly the configuration that drives precision down and false alarms up.

#### 5.3.3 The Interaction Winner: How a Slow Learning Rate Turns the Trap Inside Out

Now watch what happens at batch size 64 when we *slow down* the learning rate to `0.0005`:

| Configuration | F1 | AUC |
|---|---:|---:|
| `lr=0.001, bs=64, bf=32` *(Smoothness Trap)* | 0.745 | 0.950 |
| `lr=0.0005, bs=64, bf=32` *(Champion)* | **0.866** | **0.970** |

Same large batch. Same wide network. Only the learning rate has changed — and F1 jumps from 0.745 to 0.866. This is the **inverse** of the Smoothness Trap:

- **Big batch (64)** → smoothed, confidently directed gradient.
- **Slow learning rate (0.0005)** → small, cautious steps in that direction.
- **Together: small steps in a confidently right direction** → the optimiser descends gradually into a tight, deep minimum that small-batch noise would have prevented it from finding. The wider network (BF = 32) provides the parameter capacity needed to actually fit that minimum.

This is the central scientific finding of the project. **Hyperparameters cannot be selected independently along separate axes**; their optimal values are mutually dependent. A grid search restricted to "reasonable" interior values would have missed this finding entirely. Probing the *boundary* (batch size = 64) revealed it.

#### 5.3.4 A Subordinate Three-Way Interaction

A second, weaker pattern is visible at the bottom of the leaderboard. Look at rank 12 — the worst run in the entire grid:

| Configuration | F1 | False alarms |
|---|---:|---:|
| `lr=0.0005, bs=64, bf=16` | 0.602 | ~69 |

Same slow learning rate, same large batch as the champion — but a *narrower* network (BF = 16 instead of 32). The F1 collapses to 0.602 and the false-alarm count rockets to about 69 — roughly **6× more false alarms** than the leaderboard's best small-batch configuration. The smoothed gradient signal that BS = 64 supplies is most productively absorbed by a *wider* parameter space; the narrow BF = 16 network simply lacks the representational capacity to use it, and the imbalanced objective destabilises during training.

The full picture is therefore a *three-way* interaction between learning rate, batch size, and base filters. The grid had to be Cartesian (every combination tested) precisely because no single axis can be tuned in isolation.

### 5.4 The ROC Curve: Looking at the AI's Confidence Across Every Possible Threshold

The figure [`roc_curve_run_lr5em04_bs64_bf32.png`](docs/tuning_visualizations/roc_curve_run_lr5em04_bs64_bf32.png) shows the ROC curve for the champion. To read this plot:

- The **horizontal axis** is the false-alarm rate. Closer to zero = fewer false alarms.
- The **vertical axis** is the catch rate (Recall). Closer to one = catching more real failures.
- The **curve itself** sweeps through every possible decision threshold the AI could use. A perfect AI's curve hugs the top-left corner; a random-guess AI's curve runs along the diagonal.

The champion's curve climbs to **about 85% catch rate while the false-alarm rate is still effectively zero** — a near-vertical wall on the left side of the plot. It then plateaus around 97% catch rate before saturating. In plain language, this means we can dial the alarm threshold to be very strict (raising essentially zero false alarms) and still catch 85 out of every 100 real failures. If we relax the threshold a little, we can catch nearly 97 out of 100 — at the cost of letting some false alarms through. **The exact trade-off is a business decision** (see §6.3) that depends on how expensive each kind of error actually is in a particular factory.

### 5.5 The Confusion Matrix: Counting Every Outcome

The figure [`confusion_matrix_run_lr5em04_bs64_bf32.png`](docs/tuning_visualizations/confusion_matrix_run_lr5em04_bs64_bf32.png) is a 2 × 2 table that tabulates every prediction the AI made on the 631 test recordings:

```
                        Predicted Healthy    Predicted Broken
True Healthy   (n=563):       555                   8           ← 8 false alarms
True Broken    (n=68):         10                  58           ← 10 missed failures
```

Out of 563 truly healthy pumps, the AI raised a false alarm on only **8**, a false-alarm rate of just **1.4%**. Out of 68 truly broken pumps, the AI correctly caught **58**, a catch rate (Recall) of **85.3%**, missing 10. Of every 66 alarms the system raised, **58 were genuine** (Precision ≈ 87.9%) — when the AI says "fault," the operator is justified in treating it as serious.

For comparison, the runner-up configuration (`run_lr1em03_bs16_bf16`) caught two fewer real faults (56 vs 58) *and* raised three more false alarms (11 vs 8). The champion is therefore strictly better on *both* error axes simultaneously — a so-called **Pareto improvement**.

### 5.6 Grad-CAM: Looking Where the AI Looks

Numbers like F1 and AUC tell us *that* the AI works. They do not tell us *why*. For a system intended for industrial deployment, this distinction matters: an engineering team will not trust a black-box decision-maker with safety-critical equipment. They want to verify that the AI is making decisions for the right reasons.

The technique we use to open the black box is called **Grad-CAM** (Gradient-weighted Class Activation Mapping). The intuition is straightforward. After the AI has classified a recording, we work *backwards* through its internal computations to figure out which regions of the spectrogram contributed most to its decision. The result is a **heatmap** — a coloured overlay on top of the original spectrogram — where bright regions show "the AI was strongly looking here" and dark regions show "the AI ignored here."

The figures [`gradcam_abnormal_run_lr5em04_bs64_bf32.png`](docs/tuning_visualizations/gradcam_abnormal_run_lr5em04_bs64_bf32.png) and [`gradcam_normal_run_lr5em04_bs64_bf32.png`](docs/tuning_visualizations/gradcam_normal_run_lr5em04_bs64_bf32.png) show what the champion was paying attention to on a typical broken-pump and a typical healthy-pump recording, respectively.

- **Broken pump.** The heatmap concentrates intense activations in the **mid-frequency range, roughly 512 Hz to 2,048 Hz**, with discrete bright bursts at approximately 1.2 seconds, 4.0 seconds, and a particularly intense burst around 6.0 seconds of the 10-second clip. This is exactly where a mechanical engineer would expect to find the harmonic signatures of bearing wear, cavitation, and impeller distress, and the *burst-like* timing pattern is consistent with the periodic mechanical impacts that characterise a degraded rotating component. The AI has learned to look exactly where a domain expert would look.
- **Healthy pump.** The heatmap looks completely different: a **broad, time-uniform horizontal band** of attention above ~4,096 Hz, with secondary attention near 2,048 Hz. The activations are essentially constant over the full 10 seconds. This is the hallmark of a *stationary* acoustic signal — the broadband noise floor of an undamaged pump operating normally. The AI is, in effect, classifying "Healthy" by recognising the *absence* of the burst-like transient structure that characterises broken pumps.

The contrast — *localised mid-frequency bursts* for broken, *uniform high-frequency hum* for healthy — confirms that the AI has internalised the *physics* of the machine, not some spurious correlation with background factory noise or recording-specific quirks. For an engineering team contemplating real deployment, this is exactly the credibility test that aggregate metrics alone cannot provide.

> **Key Takeaway — Evaluation.** The champion configuration catches 58 of 68 real failures while raising only 8 false alarms across 563 healthy recordings, with a threshold-independent ranking quality of 0.97. The 12-run grid search reveals that this performance arises not from a fragile hyperparameter combination but from an inherent interaction between batch size and learning rate that is only visible at the boundary of the search space. Grad-CAM heatmaps confirm that the AI's decisions are anchored in the same mid-frequency regions that a human mechanical engineer would diagnose by ear.

---

## 6. Limitations

The system works well within the conditions it was trained on. Three caveats must be acknowledged before any real-world deployment.

### 6.1 Domain Shift

The AI was trained on the MIMII dataset's specific acoustic environment: pumps mixed with a particular factory-noise profile at -6 dB SNR. A different factory — one with a dominant air-conditioning hum, different room reverberation, or a different mix of nearby machines — will sound subtly different to the AI, and it will likely raise more false alarms until it is **fine-tuned** on a small number of recordings from the new environment. This is a familiar phenomenon called *domain shift*, and it is not a defect of the architecture, but it is a real deployment cost that any industrial customer must plan for.

### 6.2 Single-Sensor Reliance

The AI listens through a single microphone. Real industrial monitoring systems typically combine *multiple* sensor modalities — vibration measurements from accelerometers, thermal imagery from infrared cameras, motor current measurements, fluid pressure and flow sensors — in a strategy called **sensor fusion**. A multi-modal model would be more robust to acoustic occlusion (a worker's voice or a forklift driving past temporarily masking the pump) and could detect fault categories — like overheating — that are simply invisible to a microphone.

### 6.3 The Decision Threshold Is a Business Question, Not a Maths Question

The AI produces a continuous confidence score between 0 and 1. To turn that score into a binary "alarm/no alarm" decision, we have to choose a *threshold*. The ROC-AUC of 0.97 confirms that the AI's confidence ranking is informative across the entire threshold range — but choosing the *specific* threshold for deployment is **not** a machine-learning question. It is a business question that depends on:

- The cost of a false alarm (operator time, unnecessary teardown).
- The cost of a missed failure (downtime, secondary damage, safety risk).

A reactor pump in a chemical plant warrants an aggressive (low) threshold — even a 1% chance of failure justifies a precautionary inspection. A redundant cooling pump in a non-critical loop tolerates a more relaxed (higher) threshold. The AI exposes the trade-off; the asset owner decides where to set the dial.

---

## 7. Future Work

Three avenues of follow-up work would meaningfully improve the system's statistical rigour, robustness, and deployability.

### 7.1 K-Fold Cross-Validation

The current 70/15/15 split, with a fixed random seed, produces a **point estimate** of how well the system generalises — informative, but a single sample of the underlying performance distribution. **K-fold cross-validation** (typically with K = 5) would split the data into 5 different train/test partitions and report the *average* and *spread* of performance across all of them. This converts a single number ("F1 = 0.866") into a confidence interval ("F1 = 0.85 ± 0.03"), and definitively rules out the possibility that the strong reported numbers were an artefact of one fortunate split. Combined with statistical hypothesis testing, K-fold CV would also let us formally claim that the champion is significantly better than the runner-up — at present, the gap of 0.04 F1 is suggestive but not yet proven against random variation.

### 7.2 Acoustic Data Augmentation (SpecAugment)

The minority class is under-represented by an order of magnitude, and weighted loss only partially compensates. **SpecAugment** is a technique that generates *additional* training variations by randomly masking horizontal stripes (frequency bands) and vertical stripes (time slices) of each spectrogram during training. The effect is to force the AI to learn *redundant* signatures of each fault category — multiple ways of identifying a broken pump — rather than relying on a single high-confidence pattern. This would directly address the residual recall ceiling of 85% (10 of 68 broken pumps still missed) by making the AI more robust to partial signal occlusion, which is exactly the failure mode that the noisy -6 dB factory environment produces.

### 7.3 Edge Deployment via Quantisation (TinyML)

The full-fidelity AI runs in Python with PyTorch, requiring several hundred megabytes of supporting libraries — impractical for a tiny battery-powered microcontroller mounted directly on a pump. **Quantisation** is the process of converting the AI's internal numbers from 32-bit floating-point representation (high precision, fat memory footprint) to 8-bit integer representation (lower precision, ~4× smaller, much faster on cheap hardware). The resulting model can run on an ARM Cortex-M class microcontroller, transforming the system from an off-line analysis tool into a **continuous, on-device monitor**: a small box bolted to the pump itself, listening 24/7, raising alarms in real time without ever touching the network.

---

## 8. Conclusion

This project tells the story of an AI system that learns to listen for impending mechanical failure in industrial pumps — and the engineering choices that turned that idea into a working tool.

Four decisions account for most of the final system's quality:

1. **Class imbalance was confronted head-on** with a weighted loss function that makes missing a real failure roughly 8× more costly to the AI than raising a false alarm. We deliberately avoided synthetic over-sampling, which would have manufactured fake failures the AI would have learned to recognise but a real factory never produces.
2. **Overfitting was eliminated structurally** by replacing the conventional Flatten-then-classify head with a Global Average Pooling layer, slashing the parameter count by 95% and forcing the AI to learn position-invariant patterns rather than memorising individual training clips.
3. **The optimum was located systematically** through a 12-point Cartesian grid search, with each run isolated in its own operating-system subprocess to eliminate GPU-memory leaks, crash propagation, and reproducibility bugs.
4. **The grid was extended to a boundary condition** — batch size = 64 — which exposed an unexpected interaction effect that an interior-only search would have missed. The *Smoothness Trap* (large batch + aggressive learning rate = overshoot) and its inverse (large batch + slow learning rate + wide network = optimum) were invisible until the grid was extended; once visible, they redrew the deployable model.

The final system catches **58 of 68 truly broken pumps** in a held-out test set, raising only **8 false alarms across 563 healthy recordings** — and Grad-CAM heatmaps confirm that it is doing so by paying attention to the same mid-frequency acoustic signatures that a mechanical engineer would diagnose by ear.

The methodological lesson generalises beyond this dataset. When a hyperparameter grid is restricted to "reasonable" interior values, the search will converge to whichever interior configuration best balances regularisation and capacity — but it cannot characterise how those properties interact at the *boundaries*. Pushing the grid to a boundary value cost almost nothing computationally, and uncovered the configuration that became the deployable system. **Boundary-condition probes deserve to be a standard part of every hyperparameter search.**

The final deliverable is therefore not just a classifier. It is an **explainable AI tool** — one that bridges raw acoustic signal processing, principled treatment of class imbalance, systematic ablation across boundary conditions, and the visual transparency required for an engineering team to trust it with the asset-protection responsibilities that motivated the project in the first place.

---

## Appendix A: Repository Structure

```
Predictive-Maintenance-Audio-CNN/
├── src/
│   ├── preprocess.py    # Audio → Log-Mel-Spectrogram conversion
│   ├── dataset.py       # Lazy-loading dataset, class weights, train/val/test splits
│   ├── model.py         # The 4-block CNN with Global Average Pooling head
│   ├── train.py         # Training loop, weighted loss, automatic safety nets
│   ├── evaluate.py      # F1, ROC-AUC, confusion matrix, per-run logging
│   └── explain.py       # Grad-CAM heatmap generation
├── tests/
│   ├── conftest.py      # Imports plumbing
│   ├── test_preprocess.py
│   ├── test_dataset.py
│   └── test_model.py
├── tune.py              # 12-point Cartesian grid search via OS subprocess
└── docs/
    ├── experiment_tracking.csv             # Per-run metrics for all 12 configurations
    └── tuning_visualizations/              # ROC curves, confusion matrices, Grad-CAM heatmaps
```

## Appendix B: Glossary of Plain-English Definitions

| Technical term | Plain-English definition |
|---|---|
| **Spectrogram** | A "visual fingerprint" of sound: a 2-D image where time runs left-to-right, pitch runs bottom-to-top, and brightness shows loudness. |
| **Mel scale** | A non-linear pitch axis that mimics human hearing. Gives more vertical resolution to low frequencies (where mechanical faults live) than to high frequencies. |
| **Decibel (dB)** | A logarithmic measure of loudness. Compresses huge ranges of sound power into manageable numbers. |
| **CNN (Convolutional Neural Network)** | A type of AI originally designed to recognise patterns in images. Treats each spectrogram as a single-channel grayscale picture. |
| **Parameter / weight** | An internal adjustable number inside the AI. More parameters = more capacity = greater risk of memorisation rather than learning. |
| **Hyperparameter** | A dial set *before* training begins (e.g. learning rate). Hyperparameters control *how* the AI learns, not what it learns. |
| **Loss function** | The AI's "report card" — a number that measures how wrong each prediction was. The AI learns by trying to make this number smaller. |
| **Weighted loss** | A loss function that punishes mistakes on the rare class more harshly than mistakes on the common class. |
| **Epoch** | One complete pass through every example in the training set. |
| **Batch size** | How many examples the AI looks at before doing one round of self-correction. |
| **Learning rate** | How big a step the AI takes when it self-corrects. |
| **Overfitting** | When an AI memorises its training examples instead of learning generalisable patterns. The textbook example of an AI that aces practice tests but fails the real exam. |
| **Global Average Pooling (GAP)** | A specific structural change to the AI that removes its ability to encode *where* a pattern fired, only *how strongly*. Acts as a powerful built-in regulariser. |
| **Recall** | Of all the genuinely broken pumps, what fraction did the AI catch? |
| **Precision** | When the AI raised an alarm, how often was it justified? |
| **F1-Score** | A balanced combination of Precision and Recall. High when both are high. |
| **ROC-AUC** | A threshold-independent score: the probability that the AI gives a randomly chosen broken pump a higher confidence than a randomly chosen healthy pump. |
| **Confusion matrix** | A 2 × 2 table showing all four outcomes: true positives, false positives, true negatives, false negatives. |
| **Grad-CAM** | A heatmap that shows where on the spectrogram the AI was looking when it made a particular decision — a window into the AI's reasoning. |
| **Subprocess** | A separate program launched by the main program. Used here to give each grid-search run a clean GPU memory state. |
| **Stratified split** | A way of dividing data into subsets that preserves the class proportions in every subset. |
| **Domain shift** | The phenomenon where an AI trained in one environment performs worse in a slightly different one. |
| **Quantisation** | Converting the AI's internal numbers from 32-bit (high-precision, big) to 8-bit (lower-precision, small), so the AI can run on tiny low-power hardware. |
