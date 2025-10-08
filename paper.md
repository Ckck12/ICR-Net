Are Deepfake Detectors Robust to Temporal Corruption?
Ben Trovato∗
G.K.M. Tobin∗
trovato@corporation.com
webmaster@marysville-ohio.com
Institute for Clarity in Documentation
Dublin, Ohio, USA
Abstract
Deepfake video detection aims to classify video authenticity by identifying facial forgeries. Although existing studies have achieved
promising results by shifting from frame-based to video-based methods, focusing on evaluating spatial robustness, leaving temporal
robustness underexplored. In real-world scenarios, web-streaming
videos can be corrupted by network disruptions such as packet loss,
bit error, or compression, leading to temporal inconsistencies. However, current evaluation protocols and benchmarks lack addressing
these temporal corruptions. To close this gap, we introduce DFTCB (DeepFake Temporal Corruption Benchmark), a benchmark
for temporal robustness built on FaceForensics++ and DFDC spanning diverse corruption types and severity levels. Our evaluation
on these benchmarks reveals a critical fragility in existing methods.
Furthermore, we propose ICR-Net, which estimates frame reliability
and selectively restores corrupted frames. Trained with contrastive
learning on clean-corrupted pairs, it learns corruption-invariant
and class-separable features. Our method not only achieves stateof-the-art (SOTA) robustness performance but also demonstrates
strong cross-dataset generalization under temporal corruptions.
We believe that this study will serve as a cornerstone for temporal
robustness in deepfake detection.
CCS Concepts
• Security and privacy → Web application security.
Keywords
Deepfake Video, Temporal Corruption, Model Robustness
ACM Reference Format:
Ben Trovato and G.K.M. Tobin. 2025. Are Deepfake Detectors Robust to
Temporal Corruption?. In Proceedings of The Web Conference (WWW). ACM,
New York, NY, USA, 12 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Deepfake video detection is an important task for ensuring digital
media integrity, which aim is to distinguish AI-generated forgeries
from authentic videos. Deepfake content targets human faces, creating realistic forgeries used for criminal impersonation and targeted
disinformation. Therefore, developing a robust deepfake detection
method [44] has become a critical research imperative. While existing methods have achieved remarkable performance advancements,
∗Both authors contributed equally to this research.
WWW, April 13–17, 2026, Dubai, UAE
2025. ACM ISBN 978-x-xxxx-xxxx-x/YY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnn
Packet Loss Motion Blur Bit Error
H.265 CRF H.265 ABR
Black Frame
H.264 CRF H.264 ABR
Temporal Corruption Types
Web streaming
unstable Detector failed
Figure 1: Examples of deepfake videos with temporal corruption types commonly encountered in web-streamed content.
their robustness has been evaluated exclusively against spatial corruptions [14, 28, 38, 49], leaving their research gap to temporal
corruptions as a significant and largely unexplored vulnerability.
This vulnerability becomes especially critical in real-world applications like live-streaming [31], where video streams are often
degraded by network issues and video signal processing such as
packet loss, bit errors, or compression. Such temporal corruptions
are particularly damaging because they attack on two fronts: they
introduce jarring, unnatural transitions between frames while also
degrading the spatial artifacts within them.
Existing deepfake detectors are categorized as frame-based [5, 22,
41] and video-based [11, 38] methods. Frame-based methods operate independently per frame and primarily exploit spatial artifacts
such as blending boundaries and color and texture inconsistencies. However, temporal corruptions distort or erase these forgery
cues, weakening the model’s capability to distinguish deepfake artifacts [19]. Moreover, aggregating frame scores by a simple mean
assumes frame independence, so when temporal corruptions occur, the video level result is effectively dictated by the model’s
spatial information on individual frames. When more than half of
the frames are corrupted, corrupted frames dominate the videolevel prediction, causing the model to respond more sensitively
to corruption artifacts than actual deepfake artifacts. This makes
frame-based methods vulnerable to temporal corruption, leading to
severe performance degradation in temporal corrupted scenarios.
WWW, April 13–17, 2026, Dubai, UAE Trovato et al.
On the other hand, video-based methods typically split a video
into clips and jointly encode multiple frames, thereby explicitly
modeling temporal dynamics alongside spatial appearance. They exploit temporal inconsistencies in facial alignment caused by frameby-frame manipulation. A primary goal of these models is to improve the generalization of deepfake detection by focusing on temporal information. To address the tendency of video-based encoders
to overfit to spatial information, several approaches utilize 3D-CNN
encoders with a filter kernel size of 𝐾 × 1 × 1 ( 𝐾𝑡 × 𝐾𝑤 × 𝐾ℎ), framing temporal information as the change in value at a specific pixel
location over time [38, 49]. However, these approaches implicitly
assume smooth motion and uniform frame reliability in each video.
When some frames are corrupted, the model finds it difficult to
distinguish whether changes in pixel values are rooted in manipulation or corruption. Also, temporal corruptions introduce abrupt
discontinuities and noise that models misread as manipulation dynamics, causing false positives on real videos. Nevertheless, there
is no mechanism to separate corruption artifacts from forgery patterns and to correct corrupted frames. This highlights the general
gap of temporal robustness in existing methods and highlights the
need for a formal benchmark to evaluate these vulnerabilities. To
systematically investigate this vulnerability, we address our study
around three research questions: RQ 1. How robust are existing
deepfake detectors across temporal corruption? RQ 2. How does
temporal corruption affect a model’s generalization capability on
unseen data? RQ 3. How can temporal corruption-robust deepfake
detectors be designed?
To answer these questions, we introduce the DeepFake Temporal Corruption Benchmark (DF-TCB), a public benchmark
built upon standard datasets on FaceForensics++ [33] for training
and DFDC [7] for testing (FF++-C and DFDC-C, respctively). DFTCB includes a suite of eight realistic temporal corruption types
ranging from video compression to transmission errors, each applied across three levels of severity to enable a comprehensive and
fine-grained analysis. Furthermore, we propose ICR-Net, a robust
training method designed to defend against temporal corruptions.
We observe that existing methods confuse corruption artifacts with
forgery artifacts, which inflates false positives on real videos. ICRNet makes the model reliability-aware and correction-capable: it
predicts a per-frame integrity score and applies a residual correction
only when integrity is low, while blending original and corrected
features based on the predicted integrity so that the forensic cue is
preserved and corruption is suppressed. Training uses paired clean
and pre-generated corrupted clips. We adopt pairwise contrastive
learning so that clean-corrupted views encourage corruption invariance while real–fake supervision preserves class separability. This
directly addresses the corruption-forgery confusion, reducing false
positives on corrupted real videos and improving robustness across
corruption types, severities, and datasets. Our extensive experiments on DF-TCB show that contemporary frame- and video-based
detectors suffer drops under temporal corruptions, with corruption
patterns playing a decisive role. ICR-Net achieves strong robustness
and preserves clean-set accuracy while improving cross-dataset
generalization.
Our main contributions are summarized as follows:1
1Our code and dataset will be released here:
• We provide the first systematic analysis of the temporal robustness of deepfake detectors, revealing the critical impact
of corruption patterns.
• We introduce DF-TCB, a public benchmark built on FaceForensics++ and DFDC with 8 temporal corruption types
and 3 severities and standardized protocols for fine-grained
robustness evaluation.
• We propose ICR-Net, a novel framework that learns corruptionrobust deepfake detection through three key designs: (i) temporal prediction for integrity assessment, (ii) frame selective correction, and (iii) corruption-invariant, class-separable
contrastive learning.
2 Related Work
2.1 Frame-Based Deepfake Detection
Early approaches treat deepfake detection as an image-level classification task. These methods exploit spatial artifacts such as local
inconsistencies [10, 30, 42], global texture differences [24], and
multi-scale anomalies [36]. Frequency-based methods exploit spectral artifacts introduced during up-sampling. For example, grid-like
structures in the frequency domain have been used for detection [8].
F3-Net [32] and FDFL [20] combine RGB with frequency-aware features to capture complementary traces. SPSL [23] leverages spatial
images and phase spectra to highlight up-sampling artifacts, improving transferability. SRM [25] focuses on high-frequency noise,
while SFDG [37] jointly models spatial-frequency relations through
dynamic graph learning. Data augmentation further enhances detector by exposing models to harder training examples. Multi-att [47]
introduces attention-guided augmentation to reveal subtle artifacts.
Face X-ray [21] emphasizes blending boundaries for boundaryaware learning. PCL [48] builds on this by generating synthetic
images and applying pair-wise self-consistency learning to detect
internal inconsistencies. Recent work also focuses on higher-quality
forgeries and greater diversity. SBI [34] produces self-blended faces
by swapping within the same identity. LAA-Net [28] synthesizes
pseudo-deepfakes through self-supervised learning to improve generalization. While these methods perform well on still images, their
reliance on static spatial cues limits robustness in videos where
temporal consistency is crucial.
2.2 Video-Based Deepfake Detection
Video-based detection methods extend frame-level methods by modeling temporal dynamics. Early work adapts network architectures
or training strategies to combine spatial and temporal information.
AltFreezing [38] alternates freezing spatial and temporal weights to
balance their contributions. TALL [39] transforms video clips into
thumbnail layouts, converting each frame to grid boxes for imagelevel detectors to exploit temporal structures. Other approaches fuse
domains more directly, either with joint 2D CNN modeling [11]
or dual-branch designs that separately learn and then combine
spatial and temporal cues [12]. Some methods focus on temporal
consistency. FTCN [49] applies a temporal convolutional network
with minimal spatial reliance, identifying artifacts from coherence
violations across frames. Region-specific models also appear. For
example, LipForensics [14] targets the lip region to expose irregularities in mouth movements. RealForensics [13] incorporates audio
Are Deepfake Detectors Robust to Temporal Corruption? WWW, April 13–17, 2026, Dubai, UAE
Distributed Corrupted Frames
Clean Frames Corrupted Frames Clean Frames
Deepfakes - Bit Error - Severity L3
(a) Type and Severity of Corruption Consecutive Corrupted Frames
Corrupted Frames
16 Frame Distributed Corruption
Corrupted Frames Corrupted Frames
Clean Frames Corrupted Frames Clean Frames
16 Frame Corruption
Clean Frames Corrupted Frames Clean Frames
8 Frame Corruption
Clean Frames Corrupted Frames Clean Frames
24 Frame Corruption
(b)
(c)
Clean Frames
Clean Frames Clean Frames
Real - Bit Error – Severity L3
Clean Frames
FaceSwap – H.265 CRF - Severity L1
Real – H.265 CRF–Severity L1
Corrupted Frames
Clean Frames
Clean Frames
Corrupted Frames
Corrupted Frames
(6 types x 8 types x 3 types) = 144 types
Figure 2: Overview of the proposed temporal corruption benchmark. (a) We define eight types of temporal corruptions
that commonly occur in real-world web scenarios, each with three severity levels. (b) We further investigate scenarios with
consecutive corrupted frames, and (c) scenarios where corrupted frames are distributed across the video.
cues and applied self-supervised learning to improve feature representations. Transformer-based approaches have also been applied
to video-level modeling, where a Vision Transformer [2] captures
long-range spatio-temporal dependencies [46].
2.3 Robustness of Deepfake Detection
The robustness of deepfake detectors against common corruptions
has been studied extensively in the past few years. FaceForensics++ [33] provides benchmark datasets compressed with the H.264
CRF codec at 23 and 40 levels, enabling evaluation under different
compression intensities. DeeperForensics-1.0 [17] introduces distortions of varying intensity to the test set, allowing systematic analysis under corrupted inputs. Subsequent studies [3, 9, 28, 40] have
consistently examine model robustness against image degradations
such as saturation, contrast changes, Gaussian noise, Gaussian blur,
and pixelation. ISTVT [46] further assesses robustness under JPEG
compression, downscaling, and pixel dropout. These evaluation
protocols have become standard in deepfake detection field. Videobased detectors have emerged to capture temporal dependencies
neglected by frame-based methods. However, their robustness evaluations still largely focus on frame-level corruptions [14, 28, 38, 49].
In real-world video processing, corruptions can occur during acquisition, encoding, streaming, or platform-level processing. These
include transient frame drops, compression artifacts, or brief bit
errors, collectively referred to as temporal corruptions. Prior
studies in video classification [43] and action recognition [45] show
that such disruptions can severely degrade model performance.
Deepfake videos circulating on the web are particularly vulnerable
to these brief temporal corruptions.
3 DF-TCB Benchmark Creation
3.1 Benchmark Setup
Our research is motivated by the scenario of streaming quality
degradation that can occur in real-world video processing environments. It is common for a video stream to temporally suffer from
artifacts like blocking or packet loss due to network instability on
social media platforms or during video calls, only to return to a
clean state seconds later. Therefore, our study focuses on partial corruption patterns as the baseline and treats whole-frame corruption
as ablation.
Let an original video be 𝑉 = {𝐼𝑡 }
𝑇
𝑡=1 with its corresponding label
𝑦 ∈ {0 → real, 1 → fake}, where 𝐼𝑡 denotes the frame at timestep 𝑡.
Given a corruption operator𝑐(·;𝑙) with severity 𝑙 and an corrupted
index set 𝑚 ⊂ {1, . . . ,𝑇 }, we define a corrupted video 𝑉˜ = {
˜
𝐼𝑡 }
𝑇
𝑡=1
as:
˜
𝐼𝑡 =
(
𝑐(𝐼𝑡
;𝑙), 𝑡 ∈ 𝑚,
𝐼𝑡
, otherwise.
(1)
We consider replacement patterns on a 32-frame clip: consecutive, where |𝑚| ∈ {8, 16, 24} contiguous frames are replaced by
WWW, April 13–17, 2026, Dubai, UAE Trovato et al.
a window centered in the clip. Unless otherwise stated, inputs are
standardized to 32 frames sampled with a fixed stride of 8 and
resized to 256×256.
3.2 Temporal Corruptions in Videos
We consider eight realistic corruptions that arise across the video
pipeline, ranging from camera acquisition through video processing to transmission. We denote the set of corruption types by
𝑐 = {Black Frame, Motion Blur, Packet Loss, Bit Error, H.264 CRF,
H.264 ABR, H.265 CRF, H.265 ABR} and severity levels by𝑙 = {1, 2, 3}.
The description of each corruption type, along with the configuration of the three severity levels, is as follows:
• Black Frame [4]: Represents transient signal loss that can occur
from capture glitches or network dropouts. This is simulated by
replacing selected video frames entirely with black pixels.
• Motion Blur [16]: Occurs from rapid camera or subject motions. This is simulated by temporally averaging frames within
a sliding window, where window sizes of 3, 7, and 13.
• Packet Loss [43]: Happens on imperfect channels when data
packets are dropped, propagating decoding errors to subsequent
frames. This is simulated by streaming video over a virtual link
with packet drop rates of 1%, 3%, and 5%.
• Bit Error [18]: An error in the transport stream that results in
visual artifacts like blocking and smearing. This is simulated
using a bitstream filter to flip bits at frequencies of one per
100,000, 30,000, and 10,000.
• H.264 CRF [26]: A compression strategy where a higher Constant Rate Factor (CRF) increases quantization, producing stronger
artifacts. This is simulated by re-encoding videos with the libx2642
codec using CRF values of 23, 37, and 51.
• H.264 ABR [35]: A compression strategy where a lower Average
Bitrate (ABR) yields coarser textures and ringing artifacts. This
is simulated by re-encoding videos at 1/2, 1/8, and 1/32 of the
original’s bitrate.
• H.265 CRF [1]: The H.265/HEVC equivalent of H.264 CRF,
where a higher CRF intensifies quantization artifacts. This is
simulated by re-encoding with CRF values of 27, 39, and 51.
• H.265 ABR [27]: The H.265/HEVC equivalent of H.264 ABR,
where a reduced target bitrate increases degradation. This is
simulated by re-encoding at 1/2, 1/8, and 1/32 of the original’s
bitrate.
3.3 DeepFake Temporal Corruption Benchmark
We build DF-TCB using two standard deepfake datasets, FaceForensics++ (FF++) [33] and DeepFake Detection Challenge (DFDC) [7].
Following the official split protocol of FF++, we apply temporal corruptions to generate the corrupted version, FF++-C, which is used
to evaluate intra-dataset robustness. Similary, we apply the same
set of temporal corruptions to the DFDC test set, creating DFDC-C,
which serves as the benchmark for cross-dataset robustness.
Our base setting is 16-frame consecutive replacement, which
is generated for every corruption type at all severities. Exceptionally, for black frames, severity levels 1, 2, and 3 correspond to the
corruption of 8, 16, and 24 frames respectively, which cover consecutive variants. In addition, we include a small set of targeted
2https://www.videolan.org/developers/x264.html Accuracy (%)
20
80
60
40
100
Clean Video (FF++) Corrupted Video (FF++-C)
Effort#
-22.48
AltFreezing*
-14.93
CORE#
-12.23
STIL*
-19.02
FTCN*
-38.72
SRM#
-22.28
SPSL#
-12.03
F3-Net#
-14.09
FFD#
-12.53
FFD#
F3-Net#
SPSL#
SRM#
FTCN*
CORE#
Effort#
STIL*
AltFreezing*
Accuracy (%)
20
80
60
40
100
Clean Video (DFDC) Corrupted Video (DFDC-C)
-31.19
-24.38 -18.94 -23.77 -21.07 -16.25 -23.21 -24.02 -20.68
Corrputed Frames
Figure 3: Performance degradation of detectors under temporal corruptions. Top: models trained on FF++ tested on
FF++ and FF++-C. Bottom: models trained on FF++ tested on
DFDC and DFDC-C. * and # indicate video- and frame-based
detectors. Results are averaged over eight corruption types.
variants to probe pattern/extent sensitivity. 16-frame distributed
variants are created for black frame and motion blur across the
three severities. Packet loss are produced with a controlled Mininet
topology and UDP transport3
. For codec corruptions (H.264 CRF,
H.265 CRF, H264 ABR, H265 ABR), bit error, and motion blur, we
adopt FFmpeg4
for generation. Lastly, black frames are inserted by
replacing the selected temporal indices with zero-valued frames.
3.4 Preliminary Robustness Evaluation
Evaluation Protocol and Metrics. We first evaluate the temporal robustness of current deepfake detectors using the proposed
DF-TCB benchmark. We consider 9 detectors, including classical
methods, FFD [6], F3-Net [32], SPSL [23], SRM [15], FTCN [49],
STIL [11], CORE [29], AltFreezing [38], and the latest state-of-theart model, Effort [41]. All models are trained on clean FF++. We then
test the models on two settings: (i) intra-dataset, using clean FF++
and corrupted FF++-C, and (ii) cross-dataset, using clean DFDC
and corrupted DFDC-C. Video-level Accuracy (ACC) is reported
as the primary metric. For each corruption type and severity level,
we provide detailed per-suite results. We also report corruptionwise and severity-wise averages to capture both fine-grained and
aggregated robustness trends.
Evaluation Analysis. Fig. 3 (top) shows intra-dataset evaluation
results. Since the models are evaluated on clean frames, all detectors achieve near-perfect accuracy. However, as the severity
3https://mininet.org/
4https://ffmpeg.org/
Are Deepfake Detectors Robust to Temporal Corruption? WWW, April 13–17, 2026, Dubai, UAE
of the eight temporal corruptions increases, performance consistently drops across all detectors. Among frame-based detectors,
Effort [41], the most recent SOTA model, exhibits a 22.48% average performance drop under temporal corruption, the most significant observed vulnerability. We attribute Effort’s performance
drop to its parameter-constrained design, reflecting a robustnessgeneralization trade-off. Effort uses Singular Value Decomposition
(SVD) to freeze high-ranked components, preserving pre-trained
knowledge and enhancing spatial generalization. However, this
limits the model’s adaptive capacity. Temporal corruptions require
learning higher-level invariance, which typically involves adjusting the full parameter space. By restricting updates to only the
low-ranked components, Effort loses expressive power to counteract dynamic signal degradation. The enforced rigidity for spatial
generalization thus causes temporal vulnerability, explaining its
pronounced susceptibility among frame-based detectors.
Intriguingly, among all video-based detectors, FTCN [49] exhibits
the most severe performance degradation even under mild temporal
corruptions (severity level 1). Specifically, compared to its accuracy
on clean frames 99.65%, FTCN suffers an average drop of 38.72%
across eight types of temporal corruptions. We attribute this vulnerability to FTCN’s architecture. It replaces spatial convolutions
with 1 × 1 kernels while using large temporal kernels, emphasizing
temporal dynamics over spatial consistency. This design works well
for clean, stable videos but makes the model highly reliant on continuous temporal cues. When these cues are disrupted by temporal
corruptions, feature representations collapse, causing a sharp drop
in performance. The results suggest that strong temporal modeling
without sufficient spatial grounding increases sensitivity to realworld temporal corruptions. Robust deepfake detection requires
balanced spatio-temporal representations.
Figure 3 (bottom) presents cross-dataset evaluation results. Models trained on clean FF++ result in a range of 64.76% to 84.15%
accuracy when tested on clean DFDC. Performance differences are
modest between detectors. However, introducing temporal corruptions causes a sharp, universal drop. Even at severity level 1, all
detectors fall to approximately 50% accuracy. This indicates that
temporal corruption can reduce generalization performance to near
random guessing. The results highlight a serious risk: in tasks where
generalization to unseen domains is critical, temporal corruptions
can severely compromise detector reliability. A detailed report of
the results in Fig. 3 is provided in Tables 5 and 6 in our Appendix.
These tables report detector performance for all eight corruption
types and across severity levels from 1 to 3.
4 ICR-Net: Robust Deepfake Detection under
Temporal Corruptions
4.1 Motivation
Existing deepfake detectors often address the robustness problem
through spatial augmentations that utilize pixel-level transformations to input data. However, temporal corruptions are often stochastic and inconsistent, exhibiting patterns that can mimic manipulation artifacts, causing models to misinterpret benign distortions
as malicious forgeries. To address this challenge, we propose a
framework that explicitly trains the model to differentiate between
corruption and manipulation by learning from paired clean and
temporally corrupted versions of the same video. Our framework
operates from three perspectives. It begins by assessing the temporal integrity of frame features to distinguish reliable signals from
corruption. Based on this, a selective correction mechanism then refines only the unreliable features, preserving forensic cues. Finally,
it learns a corruption-invariant and class-separable representation
by pulling clean and corrupted views of the same video closer, while
pushing them apart from views of other videos. This integrated
approach maximizes robustness and generalization.
4.2 Problem Setup
Given a video clip 𝑉 ∈ R
𝑇 ×𝐻 ×𝑊 ×3
and its label 𝑦 ∈ {0, 1}, where
𝑦 = 0 denotes a real video and 𝑦 = 1 denotes a fake video, we
define two variants of each clip: a clean version and a corrupted
version. Accordingly, 𝑉REAL ∈ {𝑉R_clean, 𝑉R_corrupted} and 𝑉FAKE ∈
{𝑉F_clean, 𝑉F_corrupted} represent the real and fake pairs we use in
the training together. We forward these pairs of real and fake samples along with their corruptions to a Spatial Encoder 𝐸𝑆 , as illustrated in Step 1 of Fig. 4. The encoder produces clean and corrupted embeddings for both real and fake inputs: 𝑆R_clean, 𝑆R_corrupted,
𝑆F_clean, and 𝑆F_corrupted. We then group frame-level embeddings 𝑆 ∈
{𝑆R_corrupted, 𝑆R_clean, 𝑆F_corrupted, 𝑆F_clean}, where 𝑆 = [𝑠1, . . . , 𝑠𝑡 ] ∈
R
𝑇 ×𝐷 . These grouped embeddings are forwarded to the subsequent
processing modules.
4.3 Temporal Integrity and Correction
After obtaining frame embeddings 𝑆 = [𝑠1, . . . , 𝑠𝑡 ] from the spatial
encoder, our framework forwards them to the Temporal Integrity
and Correction module. As shown in Step 2 of Fig. 4, this module
is composed of two complementary branches designed to assess
feature reliability and selectively correct corruptions.
4.3.1 Temporal Prediction and Integrity Assessment. A GRU is employed to model temporal consistency. At each step 𝑡, it predicts
the next embedding 𝑠ˆ𝑡 using the hidden state ℎ𝑡−1 before observing
the actual embedding 𝑠𝑡
:
ℎ𝑡 = GRU(𝑠𝑡
, ℎ𝑡−1; 𝜃𝑊 ), 𝑠ˆ𝑡 = 𝑔(ℎ𝑡−1). (2)
The prediction error, 𝑒𝑡 = 𝑠𝑡 −𝑠ˆ𝑡
, is then converted into a integrity
score 𝛼𝑡 ∈ (0, 1], where a score close to 1 indicates high temporal
consistency (i.e., the frame is likely clean).
𝛼𝑡 = exp(−𝜆𝛼 ∥𝑒𝑡 ∥2), (3)
where 𝜆𝛼 is a scaling constant. The GRU is trained with a prediction
loss defined as:
Lpred =
1
𝑇 − 1
∑︁𝑇
𝑡=2
∥𝑒𝑡 ∥
2
2
. (4)
4.3.2 Residual Prediction. In parallel to the integrity assessment, a
1D-CNN acts as a Residual Predictor. This module, composed of 1D
convolutional layers, is designed to capture local temporal patterns,
such as flickers or abrupt changes between adjacent frames. Its
role is to process the entire embedding sequence 𝑆 to estimate a
residual 𝑟𝑡 for each frame. Conceptually, the residual 𝑟𝑡 represents
the vector required to shift a corrupted feature embedding back
towards its original, clean manifold. The network learns to produce
WWW, April 13–17, 2026, Dubai, UAE Trovato et al.
Pull
(+)
Training data
𝑽𝑹_𝒄𝒍𝒆𝒂𝒏
𝑽𝑹_𝒄𝒐𝒓𝒓𝒖𝒑𝒕
𝑽𝑭_𝒄𝒍𝒆𝒂𝒏
𝑽𝑭_𝒄𝒐𝒓𝒓𝒖𝒑𝒕
Spatial Encoder
𝐸𝑆
𝑆𝑅_𝑐𝑜𝑟𝑟𝑢𝑝𝑡
Residual Prediction
GRU
1D-CNN
𝑆መ𝑅_𝑐𝑜𝑟𝑟𝑢𝑝𝑡
𝑆𝑅_𝑐𝑙𝑒𝑎𝑛
𝑆𝐹_𝑐𝑙𝑒𝑎𝑛
𝑆𝐹_𝑐𝑜𝑟𝑟𝑢𝑝𝑡
𝑆𝑅_𝑐𝑜𝑟𝑟𝑢𝑝𝑡 𝑆𝐹_𝑐𝑜𝑟𝑟𝑢𝑝𝑡
𝑆መ𝐹_𝑐𝑜𝑟𝑟𝑢𝑝𝑡
𝐑𝐄𝐀𝐋
𝐅𝐀𝐊𝐄
Temporal Prediction and Integrity Assessment
𝑠ǁ𝑡 = 𝑟𝑡 ∗ 1 − 𝛼𝑡 + 𝑠𝑡
𝑠1
𝑠𝑡
𝑆1:𝑡−1
Prediction
ℎ0:𝑡−1 𝑆𝑅_𝑐𝑙𝑒𝑎𝑛 𝑆𝐹_𝑐𝑙𝑒𝑎𝑛
𝑆መ𝑅_𝑐𝑙𝑒𝑎𝑛 𝑆መ𝐹_𝑐𝑙𝑒𝑎𝑛
𝛼𝑡 = 𝑒
(−ℷ||𝑠𝑡 − 𝑠ෝ𝑡
||)
𝑠1
𝑠𝑡
Selective Correction
𝑆ሚ𝑅_𝑐𝑜𝑟𝑟𝑢𝑝𝑡
𝑆ሚ𝑅_𝑐𝑙𝑒𝑎𝑛
𝑆ሚ𝐹_𝑐𝑙𝑒𝑎𝑛
𝑆ሚ𝐹_𝑐𝑜𝑟𝑟𝑢𝑝𝑡
Pull Classifier
(+)
Push (-)
𝐑𝐄𝐀𝐋
𝐅𝐀𝐊𝐄
ℒcls
𝑆1:𝑡
ℒcon
ℒreg
𝑠Ƹ1
𝑠Ƹ𝑡
𝑠1
𝑠𝑡
ResNet
Frame Embeddings
ℒpred
1
0
2
3
4
5
𝑡
6
Step 1 Step 2
Step 3
Step 4
𝛼𝑡
𝛼1
𝑟 1− 𝛼 𝑆 𝑆ሚ
∗
𝑆ሚ
+
Figure 4: Overview of the proposed ICR-Net framework for temporal robustness. Step 1: The method first extracts frame
embeddings from clean and corrupted videos. Step 2: It then assesses temporal integrity for each frames, which selectively
corrects corrupted representation. Step 3: Clean and corrupted embeddings are aligned via contrastive learning to produce
corruption-invariant, class-separable representations. Step 4: Finally, classification determines video authenticity.
this corrective vector via end-to-end training, where the final classification loss guides the predictor to generate residuals. This branch
is trained with the following regularization loss:
Lreg =
1
𝑇
∑︁𝑇
𝑡=1
𝛼𝑡
· ∥𝑟𝑡 ∥2. (5)
This loss penalizes the magnitude of 𝑟𝑡 for frames with a high
integrity score 𝛼𝑡
. This encourages the predictor to only activate to
produce significant residuals when it detects strong local evidence
of corruption.
4.3.3 Selective Correction. The final corrected feature, 𝑠˜𝑡
, is produced by combining the outputs of the previous two stages. We use
an integrity fusion mechanism that applies the predicted residual
𝑟𝑡 based on the integrity score 𝛼𝑡
:
𝑠˜𝑡 = 𝑟𝑡
· (1 − 𝛼𝑡 ) + 𝑠𝑡
. (6)
This selective correction ensures that the correction is applied
strongly to frames with low integrity scores, while high-integrity
frames are preserved by minimizing changes.
4.4 Contrastive Alignment
After effective integrity score assessment and correction, we utilize
contrastive learning to learn corruption-invariant, class-separable
representations, as shown in Step 3 of Fig. 4 that better align corrupted and non-corrupted features. This approach reduces the distance between corrupted features and their clean counterparts, enabling better understanding of temporal correspondence between
both views. By pushing similar clean features in both real and fake
videos toward their corresponding corrupted versions, we achieve
better alignment and understanding of the temporal dynamics. This
objective further solidifies the discrimination of real and fake samples by learning corruption-invariant representations that maintain
clear class separation between real and fake samples:
Lcon = −
1
4𝑁
∑︁4𝑁
𝑖=1
log
exp(sim(𝑧𝑖
, 𝑧+
𝑖
)/𝜏)
exp(sim(𝑧𝑖
, 𝑧+
𝑖
)/𝜏) + Í
𝑧𝑛 ∈𝑁 (𝑖) exp(sim(𝑧𝑖
, 𝑧𝑛)/𝜏)
, (7)
where the sum is over the 4𝑁 total embeddings in a batch of 𝑁
quartets. For each anchor embedding 𝑧𝑖
, its corresponding positive
pair 𝑧
+
𝑖
is the other view from the same source video that shares the
same label (e.g., if 𝑧𝑖
is real-clean, 𝑧
+
𝑖
is real-corrupt). The set 𝑁 (𝑖)
contains its two negative pairs the views from the same source that
have a different label (e.g., for a real-clean anchor, 𝑁 (𝑖) consists of
the fake-clean and fake-corrupt embeddings). sim(·, ·) is the cosine
similarity function and 𝜏 is a temperature parameter.
Are Deepfake Detectors Robust to Temporal Corruption? WWW, April 13–17, 2026, Dubai, UAE
Table 1: Intra-dataset temporal corruption robustness of deepfake detectors (severity L3). Each model is trained on clean FF++
combined with one of the eight corrupted FF++-C sets, and evaluated on the corresponding corruption-specific FF++-C test set.
Model Clean
Frame
Corruption Type
Black Frame Motion Blur Packet Loss Bit Error H.264 CRF H.264 ABR H.265 CRF H.265 ABR
Frame-based
FFD [6] 98.14 85.89 91.23 90.98 90.83 91.80 88.70 91.44 90.63
F3-Net [32] 98.10 85.84 88.16 90.22 90.47 90.47 88.33 89.21 92.05
SPSL [23] 98.29 84.76 91.24 86.94 91.84 85.64 85.31 91.89 91.81
SRM [15] 97.87 86.11 88.27 90.85 92.09 91.45 91.17 90.04 90.21
CORE [29] 98.27 85.23 85.14 90.18 91.43 86.22 87.93 89.10 90.66
Effort [41] 94.40 84.36 85.52 86.75 87.26 85.80 85.61 85.52 86.67
Video-based
FTCN [49] 87.62 86.37 82.64 82.97 82.65 83.48 83.69 83.69 83.54
STIL [11] 97.35 86.00 91.31 86.83 89.58 88.35 89.15 92.16 89.28
AltFreezing [38] 97.21 91.54 90.71 85.73 90.77 89.40 94.58 92.55 90.47
ICR-Net (ours) 97.86 94.92 94.88 96.03 96.67 96.55 92.93 97.50 95.83
Table 2: Cross-dataset temporal corruption robustness of deepfake detectors (severity L3). Each model is trained on clean FF++
combined with one of the eight corrupted FF++-C sets, and evaluated on the corresponding corruption-specific DFDC-C test set.
Model Clean
Frame
Corruption Type
Black Frame Motion Blur Packet Loss Bit Error H.264 CRF H.264 ABR H.265 CRF H.265 ABR
Frame-based
FFD [6] 61.41 50.32 50.12 50.49 50.22 50.38 50.19 51.02 49.73
F3-Net [32] 72.02 49.93 49.77 50.01 49.92 50.13 49.68 49.33 49.59
SPSL [23] 72.15 50.41 50.18 50.25 49.97 50.03 49.83 50.31 50.28
SRM [15] 69.91 50.11 50.34 49.95 50.21 49.87 49.72 49.54 49.61
CORE [29] 71.28 50.23 49.91 50.17 50.33 49.88 49.52 49.78 49.44
Effort [41] 79.85 55.13 54.88 54.02 54.71 54.96 53.87 55.43 53.69
Video-based
FTCN [49] 61.21 53.11 52.89 53.39 54.12 53.02 52.07 53.34 53.55
STIL [11] 67.71 49.15 49.98 48.50 50.05 48.61 48.01 50.24 49.82
AltFreezing [38] 72.95 51.82 50.73 51.23 50.15 51.35 49.22 57.11 51.47
ICR-Net (ours) 78.56 58.14 59.23 59.51 58.98 59.12 57.98 59.88 58.33
4.5 Classification
After refining the embedding in all stages, we input {
˜𝑆𝑅_clean,
˜𝑆𝐹 _clean}
and {
˜𝑆𝑅_corrupt,
˜𝑆𝐹 _corrupt} through a linear classifier in Step 4. This
method help in classifying real and fake by discriminating based
on embeddings obtained from Step 3. The backward pass from the
classifier help modules in all stages assess, correct, and align based
on the classification objective. A solid classification between real
and fake samples and their corruption helps the encoder 𝐸𝑆 discriminate real and fake samples effectively. The classification objective
can be defined as:
Lcls = −
1
𝑁
∑︁𝑁
𝑖=1
h
𝑦𝑖
log𝑦ˆ𝑖 + (1 − 𝑦𝑖) log(1 − 𝑦ˆ𝑖)
i
, (8)
where 𝑦𝑖 ∈ {0, 1} is the ground-truth label and 𝑦ˆ𝑖 = 𝜎(𝑊𝑐𝑠˜𝑖 + 𝑏𝑐 ) is
the predicted probability. Here, 𝑠˜𝑖 represents the clip-level feature
vector for the 𝑖-th sample. This loss encourages the network to
learn discriminative representations that effectively separate real
samples from fake ones.
The entire ICR-Net framework is trained end-to-end by minimizing a final objective function L, which is a weighted sum of
the four previously defined loss components: the classification loss,
the prediction loss, the selective correction regularization, and the
contrastive loss. The overall objective is formulated as:
L = 𝜆1Lpred + 𝜆2Lreg + 𝜆3Lcon + 𝜆4Lcls, (9)
where, 𝜆1, 𝜆2, 𝜆3, and 𝜆4 are hyperparameters to 1.0, 0.01, 0.5, and
1.0, respectively.
5 Experiment
5.1 Experimental Results
Evaluation Protocol. While our primary benchmark investigates
model robustness against unseen corruptions (clean training, corrupted testing), a main step involves enhancing robustness directly.
Inspired by [43], we investigate this by training models with awareness of the specific corruptions encountered. We aim to improve
evaluation accuracy specifically on corrupted videos within a supervised learning setting. We augment the training data for each
video in the dataset by applying a single corruption type at a
fixed severity level of 3, in addition to the clean data. All other
experimental settings, including baseline models, evaluation metrics, and test datasets, adhere to our DF-TCB benchmark protocols.
This consistency ensures a rigorous and direct comparison with
our initial robustness analysis. We employ this identical corruptionaware training protocol to establish the enhanced robustness of our
proposed ICR-Net. Experiments on the cross corruption robustness
of ICR-Net are provided in Table 7 in the Appendix.
Intra-Domain Analysis. Table 1 reports intra-dataset evaluation results. On average, all baseline detectors exhibit a substantial improvement in robustness under corrupted conditions when
WWW, April 13–17, 2026, Dubai, UAE Trovato et al.
trained with clean and corrupted data, compared to training with
only clean samples (as shown in the preliminary experiment in
Sec. 3.4), yet the trade-off between robustness and clean data accuracy differs markedly between frame-based and video-based approaches. Nevertheless, even with corruption-augmented training,
none of the baselines achieve robustness on temporally corrupted
inputs comparable to their clean data performance, primarily due
to the absence of a tailored training strategy explicitly designed to
handle temporal degradation. In contrast, our proposed approach
demonstrates the strongest robustness against nearly all types of
temporal corruption while maintaining clean data performance
comparable to other SOTA baselines as show in Tab. 1. This improvement stems from the integration of our data-driven integrity
score, which effectively quantifies and reflects temporal degradation within the video representation, and our corruption-invariant
and class-separable contrastive objective, which enforces consistent
forensic representations regardless of corruption severity. Together,
these components enable the model to better disentangle forensic
cues and corruption cues.
Cross-Domain Analysis. Table. 2 presents the cross-dataset
evaluation results, where models are trained on the clean FF++ and
corrupted FF++-C datasets and evaluated on the DFDC-C test set.
Most detectors still experience performance degradation on unseen
datasets, even after being trained with corrupted data. In particular,
while baseline methods trained with augmented FF++-C data exhibit reduced accuracy on clean frames (compared to their vanilla
performance in the preliminary experiment in Sec. 3.4), they also fail
to generalize effectively to corrupted DFDC-C samples. In contrast,
ICR-Net maintains accuracy and robustness, achieving competitive
performance on clean frames and surpassing all other baselines
across every corruption type on the unseen DFDC-C test set. Its
training strategy enables the model to learn corruption-invariant
representations, mitigating the impact of temporal degradations
while encouraging the network to mine more subtle forensic cues,
thereby improving generalization across datasets. By estimating
frame reliability and selectively correcting corrupted frames, ICRNet consistently outperforms all baselines in both intra- and crossdataset evaluations. These results demonstrate that our approach
substantially enhances temporal corruption robustness.
5.2 Ablation of Proposed Loss Functions
We perform ablation studies to measure the proposed loss functions
in ICR-Net in Tab. 3. The baseline (Row 1) is trained with only the
classification loss Lcls to evaluate the gain of each proposed module. We first analyze the contribution of each loss term individually.
Introducing the Lpred alone (Row 2) improves clean-set accuracy
to 96.50% by enforcing temporal consistency. However, without a
correction mechanism, the robustness gain is limited. Using only
the Lreg (Row 3) is detrimental, causing performance to drop. This
is because, without Lpred enforcing a meaningful prediction, the
model can trivially minimize Lreg by forcing the integrity score
𝛼𝑡 → 0, leading to a gating collapse where correction is always
applied. The Lcon alone (Row 4) yields a minor improvement, as
view alignment is insufficient to suppress temporal noise on its own.
The synergistic effects become clear when combining the losses.
Pairing Lpred with Lreg (Row 5) stabilizes the selective correction
Table 3: Ablation study of loss components in ICR-Net on
clean FF++ and averaged corrupted FF++-C.
Lpred Lreg Lcon Clean Corrupted
× × × 91.57 86.38
✓ × × 96.50 88.38
× ✓ × 82.54 77.61
× × ✓ 95.31 85.20
✓ ✓ × 97.33 89.96
× ✓ ✓ 84.33 78.14
✓ × ✓ 96.55 91.15
✓ ✓ ✓ 97.86 95.67
Table 4: Performance comparison of deepfake detectors under distributed corruption with varying frame lengths on
corrupted FF++-C.
Model Black Frame Motion Blur
8f 16f 24f 8f 16f 24f
Frame-based
FFD [6] 97.62 83.33 83.33 98.49 91.34 90.33
F3-Net [32] 97.38 83.33 83.33 97.21 91.84 90.25
SPSL [23] 98.93 83.33 83.33 98.33 91.17 91.72
SRM [15] 96.07 92.74 16.67 97.06 88.78 87.35
CORE [29] 97.30 96.67 83.33 98.58 85.16 88.15
Effort [41] 83.26 84.76 83.33 95.36 83.93 86.90
Video-based
FTCN [49] 85.86 83.33 83.33 99.65 78.33 81.23
STIL [11] 97.62 62.50 23.93 95.71 92.27 91.28
AltFreezing [38] 96.43 91.51 88.87 96.90 94.88 90.60
ICR-Net (ours) 98.21 97.12 91.53 97.61 95.48 91.17
mechanism, improving corrupted-set accuracy. The combination
of Lpred and Lcon (Row 7) proves even more effective, boosting
robustness by using temporal prediction to identify noise and contrastive alignment to enforce feature invariance. Finally, the full
model incorporating all three losses (Row 8) achieves the highest
performance on both clean 97.86% and corrupted 95.67% data.
5.3 Impact of Corruption Conditions
We further investigate the impact of the temporally distributed
pattern of corruption and the total number of corrupted frames. For
this distributed setting, corruption is applied in non-overlapping
4-frame blocks that are randomly scattered throughout the clip. We
evaluate this pattern with a varying total of 8, 16, and 24 corrupted
frames to measure the impact of the corruption amount. The results
are presented in Tab. 4, existing methods suffer a significant performance drop as the number of corrupted frames increases, and our
proposed ICR-Net maintains high robustness. This highlights its
superior ability to handle varying amounts of scattered temporal
disruptions, particularly in the more challenging scenarios.
6 Conclusion
This paper addresses the critical vulnerability of deepfake detectors
to temporal corruptions. We introduce DF-TCB, the first systematic benchmark for this problem, and propose ICR-Net, a novel
framework that learns to identify and correct these corruptions.
Our experiments demonstrate that while existing SOTA detectors
Are Deepfake Detectors Robust to Temporal Corruption? WWW, April 13–17, 2026, Dubai, UAE
are fragile against such degradations, ICR-Net achieves superior robustness and generalization. By providing both a robust evaluation
standard and an effective method, this work lays the cornerstone for
developing the next generation of deepfake detectors for real-world
deployment.
References
[1] T. T. Alfaqheri et al. 2020. Low delay error resilience algorithm for H.265/HEVC
video. Journal of Real-Time Image Processing (2020). Springer.
[2] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, and
Cordelia Schmid. 2021. Vivit: A video vision transformer. In Proceedings of the
IEEE/CVF international conference on computer vision. 6836–6846.
[3] Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, and Xiaokang
Yang. 2022. End-to-end reconstruction-classification learning for face forgery
detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition. 4113–4122.
[4] Sungwoo Choi, Moonsik Lee, Byunghee Jung, Kiok Ahn, Byungyong Ryu, Jaemyun Kim, and Oksam Chae. 2015. Automated content restoration system for
file-based broadcasting environments. SMPTE Motion Imaging Journal 124, 8
(2015), 39–46.
[5] Xinjie Cui, Yuezun Li, Ao Luo, Jiaran Zhou, and Junyu Dong. 2025. Forensics
Adapter: Adapting CLIP for Generalizable Face Forgery Detection. In Proceedings
of the Computer Vision and Pattern Recognition Conference (CVPR). 19207–19217.
[6] Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, and Anil K Jain. 2020. On the
detection of digital face manipulation. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern recognition. 5781–5790.
[7] Brian Dolhansky, Joanna Bitton, Ben Pflaum, Jikuo Lu, Russ Howes, Menglin
Wang, and Cristian Canton Ferrer. 2020. The deepfake detection challenge (dfdc)
dataset. arXiv preprint arXiv:2006.07397 (2020).
[8] Chengdong Dong, Ajay Kumar, and Eryun Liu. 2022. Think twice before detecting
GAN-generated fake images from their spectral domain imprints. In Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition. 7865–7874.
[9] Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Ting Zhang, Weiming Zhang, Nenghai Yu, Dong Chen, Fang Wen, and Baining Guo. 2022. Protecting celebrities from
deepfake with identity consistency transformer. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 9468–9478.
[10] Jianwei Fei, Yunshu Dai, Peipeng Yu, Tianrun Shen, Zhihua Xia, and Jian Weng.
2022. Learning second order local anomaly for general face forgery detection. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
20270–20280.
[11] Zhihao Gu, Yang Chen, Taiping Yao, Shouhong Ding, Jilin Li, Feiyue Huang, and
Lizhuang Ma. 2021. Spatiotemporal inconsistency learning for deepfake video
detection. In Proceedings of the 29th ACM international conference on multimedia.
3473–3481.
[12] Zhihao Gu, Yang Chen, Taiping Yao, Shouhong Ding, Jilin Li, and Lizhuang Ma.
2022. Delving into the local: Dynamic inconsistency learning for deepfake video
detection. In Proceedings of the AAAI conference on artificial intelligence, Vol. 36.
744–752.
[13] Alexandros Haliassos, Rodrigo Mira, Stavros Petridis, and Maja Pantic. 2022.
Leveraging real talking faces via self-supervision for robust forgery detection. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
14950–14962.
[14] Alexandros Haliassos, Konstantinos Vougioukas, Stavros Petridis, and Maja Pantic. 2021. Lips don’t lie: A generalisable and robust approach to face forgery
detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition. 5039–5049.
[15] Bing Han, Xiaoguang Han, Hua Zhang, Jingzhi Li, and Xiaochun Cao. 2021.
Fighting Fake News: Two Stream Network for Deepfake Detection via Learnable
SRM. IEEE Transactions on Biometrics, Behavior, and Identity Science 3, 3 (2021),
320–331. https://doi.org/10.1109/TBIOM.2021.3065735
[16] Dan Hendrycks and Thomas Dietterich. 2019. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint
arXiv:1903.12261 (2019).
[17] Liming Jiang, Ren Li, Wayne Wu, Chen Qian, and Chen Change Loy. 2020.
Deeperforensics-1.0: A large-scale dataset for real-world face forgery detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition. 2889–2898.
[18] J. Korhonen et al. 2007. Bit-error Resilient Packetization for Streaming H.264/AVC.
In Proceedings of the Motion and Video Communications (MV) Workshop. https://
www.epfl.ch/labs/lts4/wp-content/uploads/2018/10/mv2007a.pdf EPFL Technical
Report.
[19] Binh M Le and Simon S Woo. 2023. Quality-agnostic deepfake detection with
intra-model collaborative learning. In Proceedings of the IEEE/CVF International
Conference on Computer Vision. 22378–22389.
[20] Jiaming Li, Hongtao Xie, Jiahong Li, Zhongyuan Wang, and Yongdong Zhang.
2021. Frequency-aware discriminative feature learning supervised by singlecenter loss for face forgery detection. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition. 6458–6467.
[21] Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang, Dong Chen, Fang Wen, and Baining Guo. 2020. Face x-ray for more general face forgery detection. In Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition. 5001–5010.
[22] Xi Li, Liuqiang Shu, and Lei Yu. 2024. Face Forgery Detection Based on multiscale anchor and frame Fusion. In 2024 7th International Conference on Computer
Information Science and Application Technology (CISAT). 910–913. https://doi.
org/10.1109/CISAT62382.2024.10695316
[23] Honggu Liu, Xiaodan Li, Wenbo Zhou, Yuefeng Chen, Yuan He, Hui Xue, Weiming
Zhang, and Nenghai Yu. 2021. Spatial-phase shallow learning: rethinking face
forgery detection in frequency domain. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition. 772–781.
[24] Zhengzhe Liu, Xiaojuan Qi, and Philip HS Torr. 2020. Global texture enhancement
for fake face detection in the wild. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition. 8060–8069.
[25] Yuchen Luo, Yong Zhang, Junchi Yan, and Wei Liu. 2021. Generalizing face
forgery detection with high-frequency features. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 16317–16326.
[26] Siwei Ma, Wen Gao, and Yan Lu. 2005. Rate-Distortion Analysis for H.264/AVC
Video Coding and its Application to Rate Control. IEEE Transactions on Circuits
and Systems for Video Technology 15, 12 (2005), 1527–1535.
[27] V. V. Menon et al. 2023. EMES: Efficient Multi-encoding Schemes for HEVC-based
Adaptive Bitrate Streaming. In Proceedings of ACM Multimedia Systems (MMSys).
[28] Dat Nguyen, Nesryne Mejri, Inder Pal Singh, Polina Kuleshova, Marcella Astrid,
Anis Kacem, Enjie Ghorbel, and Djamila Aouada. 2024. Laa-net: Localized artifact
attention network for quality-agnostic and generalizable deepfake detection. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
17395–17405.
[29] Yunsheng Ni, Depu Meng, Changqian Yu, Chengbin Quan, Dongchun Ren, and
Youjian Zhao. 2022. Core: Consistent representation learning for face forgery
detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition. 12–21.
[30] Yuval Nirkin, Lior Wolf, Yosi Keller, and Tal Hassner. 2021. Deepfake detection
based on discrepancies between faces and their context. IEEE transactions on
pattern analysis and machine intelligence 44, 10 (2021), 6111–6121.
[31] Marco Postiglione, Julian Baldwin, Natalia Denisenko, Luke Fosdick, Chongyang
Gao, Isabel Gortner, Chiara Pulice, Sarit Kraus, and V.S. Subrahmanian. 2025.
GODDS: The Global Online Deepfake Detection System. Proceedings of the
AAAI Conference on Artificial Intelligence 39, 28 (Apr. 2025), 29685–29687. https:
//doi.org/10.1609/aaai.v39i28.35367
[32] Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. 2020. Thinking in frequency: Face forgery detection by mining frequency-aware clues. In
European conference on computer vision. Springer, 86–103.
[33] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies,
and Matthias Nießner. 2019. Faceforensics++: Learning to detect manipulated
facial images. In Proceedings of the IEEE/CVF international conference on computer
vision. 1–11.
[34] Kaede Shiohara and Toshihiko Yamasaki. 2022. Detecting deepfakes with selfblended images. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition. 18720–18729.
[35] Minhao Tang, Jiangtao Wen, and Yuxing Han. 2019. A Generalized RateDistortion–𝜆 Model Based HEVC Rate Control Algorithm. arXiv preprint (2019).
arXiv:1911.00639 arXiv:1911.00639.
[36] Junke Wang, Zuxuan Wu, Wenhao Ouyang, Xintong Han, Jingjing Chen, YuGang Jiang, and Ser-Nam Li. 2022. M2tr: Multi-modal multi-scale transformers
for deepfake detection. In Proceedings of the 2022 international conference on
multimedia retrieval. 615–623.
[37] Yuan Wang, Kun Yu, Chen Chen, Xiyuan Hu, and Silong Peng. 2023. Dynamic
graph learning with content-guided spatial-frequency relation reasoning for
deepfake detection. In Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition. 7278–7287.
[38] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, and Houqiang Li.
2023. Altfreezing for more general video face forgery detection. In Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition. 4129–4138.
[39] Yuting Xu, Jian Liang, Gengyun Jia, Ziming Yang, Yanhao Zhang, and Ran He.
2023. Tall: Thumbnail layout for deepfake video detection. In Proceedings of the
IEEE/CVF international conference on computer vision. 22658–22668.
[40] Zhiyuan Yan, Yuhao Luo, Siwei Lyu, Qingshan Liu, and Baoyuan Wu. 2024.
Transcending forgery specificity with latent space augmentation for generalizable
deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition. 8984–8994.
[41] Zhiyuan Yan, Jiangming Wang, Peng Jin, Ke-Yue Zhang, Chengchun Liu, Shen
Chen, Taiping Yao, Shouhong Ding, Baoyuan Wu, and Li Yuan. 2024. Orthogonal
Subspace Decomposition for Generalizable AI-Generated Image Detection. arXiv
preprint arXiv:2411.15633 (2024).
WWW, April 13–17, 2026, Dubai, UAE Trovato et al.
[42] Ziming Yang, Jian Liang, Yuting Xu, Xiao-Yu Zhang, and Ran He. 2023. Masked relation learning for deepfake detection. IEEE Transactions on Information Forensics
and Security 18 (2023), 1696–1708.
[43] Chenyu Yi, Siyuan Yang, Haoliang Li, Yap-peng Tan, and Alex Kot. 2021. Benchmarking the robustness of spatial-temporal models against corruptions. Thirtyfifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2) (2021).
[44] Han Yue-Hua, Huang Tai-Ming, Hua Kai-Lung, and Chen Jun-Cheng. 2025. Towards More General Video-based Deepfake Detection through Facial Component
Guided Adaptation for Foundation Model. In Proceedings of the Conference on
Computer Vision and Pattern Recognition (CVPR)
.
[45] Runhao Zeng, Xiaoyong Chen, Jiaming Liang, Huisi Wu, Guangzhong Cao, and
Yong Guo. 2024. Benchmarking the robustness of temporal action detection
models against temporal corruptions. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. 18263–18274.
[46] Cairong Zhao, Chutian Wang, Guosheng Hu, Haonan Chen, Chun Liu, and Jinhui
Tang. 2023. ISTVT: interpretable spatial-temporal video transformer for deepfake
detection. IEEE Transactions on Information Forensics and Security 18 (2023),
1335–1348.
[47] Hanqing Zhao, Wenbo Zhou, Dongdong Chen, Tianyi Wei, Weiming Zhang, and
Nenghai Yu. 2021. Multi-attentional deepfake detection. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition. 2185–2194.
[48] Tianchen Zhao, Xiang Xu, Mingze Xu, Hui Ding, Yuanjun Xiong, and Wei Xia.
2021. Learning self-consistency for deepfake detection. In Proceedings of the
IEEE/CVF international conference on computer vision. 15023–15033.
[49] Yinglin Zheng, Jianmin Bao, Dong Chen, Ming Zeng, and Fang Wen. 2021. Exploring temporal coherence for more general video face forgery detection. In
Proceedings of the IEEE/CVF international conference on computer vision. 15044–
15054.
Are Deepfake Detectors Robust to Temporal Corruption? WWW, April 13–17, 2026, Dubai, UAE
A Detailed Results for Section 3
A.1 Intra-Dataset Robustness
Table 5 presents the intra-dataset temporal corruption robustness
of various deepfake detection models trained on FF++ and validated
on clean FF++ and corrupted FF++-C. A distinct decline trend is
evident across all types and severity levels of corruption (L1–L3)
Corruptions that interrupt temporal continuity or cause significant
compression result in the most substantial declines in accuracy.
The black frame corruption significantly impairs the efficacy of
frame-based detectors, causing SRM [15] to decline from 97.06% on
uncorrupted movies to 16.67% at L3. This indicates that arbitrary
blank frames can overshadow predictions when models depend on
per-frame inference. Motion blur has a more progressive decrease.
AltFreezing [38] declines from 85.57% to 60.00% at L3, whereas
frame-based models like CORE [29] and FFD [6] maintain performance over 80% even at elevated corruption levels. Packet loss and
bit errors produce similar effects, causing relatively mild but consistent frame-level deterioration. Among codec-related corruptions,
all cases lead to substantial performance drops across all models.
Interestingly, as noted in Sec. 3.4, the frame-based SOTA model
Effort [41] experiences the most severe degradation.
A.2 Cross-Dataset Robustness
Table 6 broadens the assessment to cross-dataset contexts, wherein
models trained on FF++ are evaluated on both clean DFDC and
corrupted DFDC-C. When evaluated on clean frames, performance
naturally drops due to domain gap. The highest accuracy is 84.15%,
achieved by the SOTA model Effort [41], which is considerably
lower than its FF++ counterpart (95.36%). Under corrupted conditions, the drop is more severe. At severity level 1, most models
achieve only 48 to 53% accuracy. Both video-based and frame-based
models exhibit comparable performance degradation under crossdomain corruptions, indicating that dataset-specific biases predominantly influence performance and restrict transferability. These
findings highlight the urgent necessity for domain-agnostic training and corruption-invariant feature learning to guarantee uniform
robustness across datasets.
B Robustness on Different Corruptions
Table 7 examines the resilience of the proposed ICR-NET when
trained on a singular corruption type and evaluated across all other
types. The averages of the columns indicate which training corruption generalizes more effectively across unobserved distortions.
Training with H.264 CRF attains the greatest mean accuracy of
94.53%, closely followed by motion blur at 94.10%, indicating that
augmentations retaining temporal structure enhance overall robustness. In contrast, H.264 ABR training demonstrates the least
successful generalization (mean 90.69%), suggesting that H.264 ABR
artifacts are too particular to the codec. Training with black frame
produces restricted cross-type advantages (mean 91.55%), indicating that managing missing frames does not inherently enhance resilience to other distortions. From a testing standpoint (row means),
the most challenging corruption is black frame (84.78%), whereas
H.265 CRF and bit error are the least challenging (96.07% and 95.25%,
respectively). Notably, training on one codec variation effectively
transfers to others. For example, models trained on H.264 CRF
achieve 97.50% accuracy on H.265 CRF, demonstrating significant
cross-codec synergy. In conclusion, motion blur and CRF training
are the most efficacious single-type augmentations for extensive
robustness, but ABR and frame-drop training do not generalize
across distortions.
C Fully Temporal Corrupted Benchmark
Table 8 presents the performance of all detectors on the entirely
corrupted FF++-C benchmark. In contrast to prior contexts, there is
no pristine reference available. Each sample is corrupted. In the context of motion blur, most frame-based models achieve high accuracy
(95–98% at L1 and beyond 80% at L3), hence affirming the relative
ease of handling blur distortions. Packet loss, however, introduces
greater unpredictability, with accuracies fluctuating between 33%
and 83% at L3. Bit error corruption has a milder impact, since the
majority of models maintain performance over 80% even at the
greatest severity levels. Codec corruptions reemerge as the most
formidable challenge-particularly H.264 ABR, which results in catastrophic failures in antiquated frame-based models such as SRM [15]
(16.90%) and SPSL [23] (25.00%). Nonetheless, contemporary CNNs
such as F3-Net [32] and FFD [6] exhibit significant robustness (78–
82%), affirming their exceptional spatial-temporal consistency. In
comparison, H.265-based corruptions (CRF and ABR) exhibit less
severity, with the majority of detectors maintaining high accuracies
even at L3.
D Discussion and Recommendations
Our experiments provide several key insights for temporally robust
deepfake detection. First, benchmark should include both frame
loss and compression scenarios, such as black frames and H.264
ABR corruptions, as they expose distinct failure modes. Among
single-type augmentations, H.264 CRF and motion blur achieve the
best balance between robustness and generalization, suggesting
that training with these corruptions can markedly improve performance. Second, model architecture plays a critical role. Detectors
relying heavily on high-frequency features, such as SRM [15]-style
designs, are vulnerable to compression, while temporal or deep
convolutional models are more stable. Notably, the results shown in
Tab 6 indicate that corruption resilience does not guarantee crossdomain generalization. Models trained on FF++ fail to transfer
reliably to DFDC, highlighting the need for cross-dataset training
or adaptation strategies. Finally, evaluation protocols should report
per-corruption performance, including both row-wise and columnwise summaries, to better capture intrinsic corruption challenges
and augmentation transferability. Collectively, these findings offer
practical guidance for designing more resilient detectors.
Future work should address two main challenges. First, methods
must bridge remaining gaps in cross-dataset generalization. Second,
future studies should extend beyond single synthetic corruptions to
more complex, real-world scenarios combining multiple temporal
distortions, such as re-compression and transmission errors.
WWW, April 13–17, 2026, Dubai, UAE Trovato et al.
Table 5: Intra-dataset temporal corruption robustness of existing deepfake detectors. All models are trained on clean FF++, and
evaluated on both clean FF++ and corrupted FF++-C. The table shows ACC (%) drop from clean to corrupted videos. * and #
denote the detection type of video-based and frame-based, respectively.
Model Clean
Frame
Corruption Type
Black Frame Motion Blur Packet Loss Bit Error H.264 CRF H.264 ABR H.265 CRF H.265 ABR
L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3
FFD# [6] 98.49 97.62 83.33 83.33 95.29 90.82 85.25 95.71 91.02 85.88 96.57 92.17 86.84 96.21 92.22 85.77 96.29 91.80 87.13 95.86 91.65 86.74 95.86 91.65 86.74
F3-Net# [32] 97.52 97.26 83.33 83.33 94.71 89.39 82.19 95.29 89.74 83.35 95.86 91.21 85.78 95.79 90.08 83.31 95.64 90.78 82.55 95.86 90.93 82.86 95.86 90.84 84.00
SPSL# [23] 98.33 98.57 83.33 83.33 97.57 90.31 80.33 96.29 92.21 86.47 97.07 93.66 87.71 97.00 92.88 87.38 96.93 93.17 88.04 97.36 93.32 88.30 97.36 93.13 88.84
SRM# [15] 97.06 96.19 92.98 16.67 94.79 88.93 82.76 93.79 87.43 80.06 95.86 90.58 87.05 95.43 90.22 82.61 95.50 90.21 80.46 95.00 89.30 83.36 95.00 90.18 85.26
CORE# [29] 98.57 97.38 96.79 83.33 96.50 92.32 87.81 95.64 90.92 86.47 96.64 91.86 86.07 96.21 91.18 84.82 96.21 91.45 85.89 96.29 91.74 87.94 96.43 91.53 88.35
Effort# [41] 95.36 82.26 83.69 83.33 81.93 74.64 66.14 80.14 76.27 73.99 80.71 77.86 74.00 79.71 80.21 76.79 79.29 80.50 75.79 80.21 75.07 66.43 79.79 75.86 66.50
FTCN* [49] 99.65 83.33 83.33 83.33 57.57 59.14 60.71 57.29 55.68 57.94 57.79 57.71 53.79 57.79 58.07 58.86 58.14 58.21 58.50 58.57 57.50 57.64 58.57 57.93 56.64
STIL* [11] 95.71 97.86 94.40 16.67 92.86 87.05 81.66 95.93 91.15 85.00 95.86 93.07 85.42 95.36 92.80 84.13 95.71 93.08 85.32 94.79 92.44 86.96 95.21 93.43 88.29
AltFreezing* [38] 96.90 95.56 95.48 92.98 85.57 69.64 60.00 93.78 88.28 79.81 93.86 94.07 90.91 94.14 92.64 76.86 94.50 91.50 72.43 92.64 93.50 90.93 92.21 93.00 91.79
Table 6: Cross-dataset temporal corruption robustness of existing deepfake detectors. All models are trained on clean FF++, and
tested on both clean DFDC and corrupted DFDC-C, showing ACC (%) drop from clean to corrupted videos.
Model Clean
Frame
Corruption Type
Black Frame Motion Blur Packet Loss Bit Error H.264 CRF H.264 ABR H.265 CRF H.265 ABR
L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3
FFD# [6] 64.76 48.50 48.43 48.78 48.42 48.50 48.39 48.50 48.50 48.88 48.57 48.65 48.61 48.52 48.29 49.15 48.63 48.68 48.91 48.45 48.50 49.81 48.42 48.39 48.42
F3-Net# [32] 71.44 48.17 48.18 48.29 48.14 48.21 48.24 48.22 48.18 48.25 48.32 48.44 48.16 48.24 48.11 48.55 48.47 48.34 48.47 48.01 47.95 47.86 48.16 48.01 48.24
SPSL# [23] 72.49 48.34 48.27 48.69 48.50 48.21 48.47 48.46 48.35 48.66 48.52 48.54 48.40 48.39 48.11 48.73 48.50 48.50 48.50 48.32 48.24 48.95 48.55 48.50 48.94
SRM# [15] 69.12 48.23 48.43 48.38 48.50 48.50 48.39 48.50 48.43 48.43 48.37 48.47 48.29 48.42 48.37 48.42 48.42 48.37 48.37 48.42 48.47 48.38 48.39 48.42 48.39
CORE# [29] 72.58 48.15 48.17 48.27 48.14 48.29 48.08 48.20 48.23 48.32 48.34 48.39 48.55 48.21 48.24 48.47 48.08 48.21 48.32 48.13 48.11 48.28 48.29 48.16 48.21
Effort# [41] 84.15 53.21 53.57 53.80 53.50 53.65 53.18 52.96 53.41 53.38 53.19 53.22 53.00 52.43 53.03 53.50 52.25 53.57 53.16 53.34 53.99 54.62 53.03 52.98 52.80
FTCN* [49] 73.24 52.20 52.18 51.91 51.99 51.81 51.66 52.17 52.07 51.91 52.20 52.18 52.31 52.23 52.10 51.94 52.23 52.18 51.73 52.15 52.02 51.80 52.20 52.10 52.02
STIL* [11] 67.71 48.42 48.45 47.85 48.39 48.27 48.34 48.77 48.66 48.47 48.32 48.41 47.92 48.68 48.08 48.37 48.73 48.50 47.85 49.33 49.30 49.37 49.15 49.38 48.96
AltFreezing* [38] 72.64 48.86 48.95 50.55 48.81 48.99 49.61 48.86 48.81 50.55 48.65 48.93 48.55 48.91 48.71 49.92 48.83 48.65 48.73 49.07 48.91 56.36 48.91 48.68 50.16
Table 7: Performance comparison using different corruptions in the training, measured by ACC (%) of the proposed ICR-NET
on FF++-C, with row and column averages.
Test
Train Black Frame Motion Blur Packet Loss Bit Error H.264 CRF H.264 ABR H.265 CRF H.265 ABR Mean
Black Frame 94.92 83.33 83.33 83.33 83.33 83.33 83.33 83.33 84.78
Motion Blur 91.90 94.88 93.45 92.62 94.64 96.55 92.26 92.26 93.57
Packet Loss 88.94 95.79 96.03 95.79 95.43 71.27 87.26 95.31 90.73
Bit Error 96.67 97.26 96.43 96.67 95.83 87.26 94.88 97.02 95.25
H.264 CRF 90.71 95.60 91.55 94.76 96.55 93.57 95.36 94.17 94.03
H.264 ABR 76.90 94.29 93.45 91.79 96.19 92.93 95.24 91.43 91.53
H.265 CRF 95.60 95.48 95.12 95.36 97.50 96.43 97.50 95.60 96.07
H.265 ABR 96.79 96.19 95.12 95.95 96.79 94.17 96.19 95.83 95.88
Mean 91.55 94.10 93.06 93.28 94.53 90.69 92.75 93.12 92.86
Table 8: Intra-domain temporal corruption robustness of existing deepfake detectors. All models are trained on clean FF++
and corrupted FF++-C, and evaluated on fully corrupted FF++-C. The table reports ACC (%). * and # denote video-based and
frame-based detectors, respectively.
Model Corruption Type
Motion Blur Packet Loss Bit Error H.264 CRF H.264 ABR H.265 CRF H.265 ABR
L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3 L1 L2 L3
FFD# [6] 96.55 87.78 81.10 82.65 79.45 72.02 98.10 88.93 82.47 97.62 90.60 74.29 97.62 86.79 78.45 97.62 92.86 89.24 97.62 92.86 89.24
F3-Net# [32] 96.55 87.68 80.81 84.18 84.05 82.61 97.98 88.90 82.33 97.62 91.67 73.69 97.50 88.69 82.26 97.38 92.52 89.00 97.38 92.52 89.00
SPSL# [23] 97.74 88.85 82.49 77.35 60.63 39.30 98.45 89.27 82.62 97.98 78.57 22.86 97.62 66.67 25.00 98.45 93.64 90.14 98.69 93.83 90.32
SRM# [15] 95.95 87.31 80.50 74.08 55.21 33.02 96.55 87.18 80.89 96.79 85.60 19.40 96.79 78.57 16.90 97.14 92.38 88.88 97.26 92.50 89.00
CORE# [29] 97.26 88.45 82.24 80.00 74.23 64.81 96.55 87.34 81.00 95.83 85.95 38.45 95.48 82.14 48.21 97.02 92.16 88.64 96.90 92.04 88.52
Effort# [41] 83.93 75.09 70.07 58.98 58.18 56.07 76.67 69.23 64.12 68.10 47.98 71.90 66.31 47.50 72.74 75.24 70.48 67.74 75.48 70.71 67.98
FTCN* [49] 78.33 71.16 65.49 81.92 81.99 82.06 80.12 72.51 67.40 80.00 80.60 80.12 80.00 80.60 82.74 80.00 75.20 72.00 80.00 75.20 72.00
STIL* [11] 96.43 87.22 81.24 75.38 53.84 34.74 96.90 87.83 81.41 97.62 87.62 17.50 97.50 80.00 36.31 97.62 92.76 89.62 97.86 93.00 89.86
AltFreezing* [38] 94.88 85.29 79.62 81.84 77.81 72.43 95.12 86.26 79.98 95.95 93.57 56.67 96.07 93.57 66.79 95.36 90.60 87.36 95.36 90.60 87.36
ICR-Net (ours) 95.95 94.40 93.21 82.61 83.63 81.22 96.90 91.90 86.52 97.14 92.38 86.60 91.31 88.69 84.98 97.62 96.79 92.55 97.50 96.55 95.48