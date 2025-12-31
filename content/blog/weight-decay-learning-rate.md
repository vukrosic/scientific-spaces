---
title: "Weight Decay and Learning Rate from a Moving Average Perspective"
date: "2025-12-05"
author: "Vuk Rosić"
excerpt: "Exploring the relationship between Weight Decay and Learning Rate by reinterpreting the model parameter update rule as an Exponential Moving Average (EMA)."
tags: ["Mathematics", "Deep Learning", "Optimization", "LLM"]
credit: "Translated and adapted from original work by Jianlin Su (kexue.fm)"
---

> **Note**: This article is translated and adapted from the original Chinese blog post by **Jianlin Su** at [kexue.fm](https://kexue.fm/archives/11459). We have expanded on the original derivations and added further technical context for modern LLM training.

Here is the full translation of the blog post into English, formatted in Markdown with LaTeX equations.

***

# Weight Decay and Learning Rate from the Perspective of Sliding Average

**By Su Jianlin | 2025-12-05**

Weight Decay and Learning Rate are critical components of LLM pre-training; whether they are set appropriately is one of the key factors determining the ultimate success or failure of a model. Since the introduction of **AdamW**, it has basically become a consensus to decouple Weight Decay to replace traditional L2 regularization. However, beyond this, there has been no significant theoretical progress on how to reasonably set Weight Decay and Learning Rate.

This article aims to initiate a discussion by sharing some of the author's new understandings of this problem: viewing the training process as a "sliding average memory" of the training data, and exploring how to set Weight Decay and Learning Rate to make this memory more scientific.

### Sliding Average #

The general form of Weight Decay is:

$$
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t + \lambda_t \boldsymbol{\theta}_{t-1}) \tag{1}
$$

Where $\boldsymbol{\theta}$ represents the parameters, $\boldsymbol{u}$ is the update quantity provided by the optimizer, $\lambda_t, \eta_t$ are what we refer to as Weight Decay and Learning Rate, and the entire sequences $\{\lambda_t\}$ and $\{\eta_t\}$ are referred to as the "WD Schedule" and "LR Schedule," respectively. Introducing the notation:

$$
\begin{aligned}
\boldsymbol{m}_t &= \beta_1 \boldsymbol{m}_{t-1} + (1 - \beta_1) \boldsymbol{g}_t, & \hat{\boldsymbol{m}}_t &= \boldsymbol{m}_t / (1 - \beta_1^t) \\
\boldsymbol{v}_t &= \beta_2 \boldsymbol{v}_{t-1} + (1 - \beta_2) \boldsymbol{g}_t^2, & \hat{\boldsymbol{v}}_t &= \boldsymbol{v}_t / (1 - \beta_2^t)
\end{aligned} \tag{2}
$$

Then for **SGDM**, we have $\boldsymbol{u}_t = \boldsymbol{m}_t$; for **RMSProp**, $\boldsymbol{u}_t = \boldsymbol{g}_t / (\sqrt{\boldsymbol{v}_t} + \epsilon)$; for **Adam**, it is $\boldsymbol{u}_t = \hat{\boldsymbol{m}}_t / (\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon)$; for **SignSGDM**, it is $\boldsymbol{u}_t = \text{sign}(\boldsymbol{m}_t)$; and for **Muon**, it is $\boldsymbol{u}_t = \text{msign}(\boldsymbol{m}_t)$. With the exception of SGDM, the examples listed here are all considered forms of adaptive learning rate optimizers.

Our starting point is the **Exponential Moving Average (EMA)** perspective, which is to write Weight Decay as:

$$
\boldsymbol{\theta}_t = (1 - \lambda_t \eta_t)\boldsymbol{\theta}_{t-1} - \eta_t \boldsymbol{u}_t = (1 - \lambda_t \eta_t)\boldsymbol{\theta}_{t-1} + \lambda_t \eta_t \left(-\boldsymbol{u}_t / \lambda_t \right) \tag{3}
$$

At this point, Weight Decay appears as a weighted average form of model parameters and $-\boldsymbol{u}_t/\lambda_t$. 

---
> Vuk's Note:

At first it may seem like a weighted sum, not a weighted average.

In mathematics, the defining property that turns a "weighted sum" into a "weighted average" is precisely that the coefficients add up to 1.

---

The sliding average perspective is not new; articles such as *«How to set AdamW's weight decay as you scale model and dataset size»* and *«Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training»* have already discussed it. This article, however, calculates various aspects more carefully under this perspective.

In the following sections, we mainly take Adam as an example, and finally discuss the applicability of other optimizers. The calculation process will overlap significantly with *«Asymptotic Estimation of AdamW's Weight RMS (Part 1)»* and *«(Part 2)»*, and readers can cross-reference them.

### Iterative Expansion #

For simplicity, let us first consider constant $\lambda, \eta$. Let $\beta_3 = 1 - \lambda \eta$ ; (think of $\beta_3$ as a "retention rate" - how much of the old weight we keep).

Then $\boldsymbol{\theta}_t = \beta_3 \boldsymbol{\theta}_{t-1} + (1 - \beta_3)(-\boldsymbol{u}_t / \lambda)$, which is formally consistent with $\boldsymbol{m}_t, \boldsymbol{v}_t$. Expanding the iteration directly gives:

$$
\boldsymbol{\theta}_t = \beta_3^t \boldsymbol{\theta}_0 + (1 - \beta_3) \sum_{i=1}^t \beta_3^{t-i} (-\boldsymbol{u}_i / \lambda) \tag{4}
$$

For Adam, $\boldsymbol{u}_t = \hat{\boldsymbol{m}}_t / (\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon)$. generally, at the end of training, $t$ is large enough that $\beta_1^t, \beta_2^t$ are sufficiently close to zero, so we do not need to distinguish between $\boldsymbol{m}_t$ and $\hat{\boldsymbol{m}}_t$, or $\boldsymbol{v}_t$ and $\hat{\boldsymbol{v}}_t$. Furthermore, simply setting $\epsilon=0$, we can simplify to $\boldsymbol{u}_t = \boldsymbol{m}_t / \sqrt{\boldsymbol{v}_t}$. Then we apply a classic **Mean Field Approximation**:

$$
\frac{1-\beta_3}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \boldsymbol{u}_i \underbrace{\approx}_{\text{denoted as } \bar{\boldsymbol{u}}_t} \frac{1-\beta_3}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \frac{\boldsymbol{m}_i}{\sqrt{\boldsymbol{v}_i}} \approx \frac{\bar{\boldsymbol{m}}_t}{\sqrt{\bar{\boldsymbol{v}}_t}} \tag{5}
$$

Where:
$$
\bar{\boldsymbol{m}}_t \triangleq \frac{1-\beta_3}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \boldsymbol{m}_i, \quad \bar{\boldsymbol{v}}_t \triangleq \frac{1-\beta_3}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \boldsymbol{v}_i
$$

Expanding $\boldsymbol{m}_t, \boldsymbol{v}_t$ gives $\boldsymbol{m}_t = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} \boldsymbol{g}_i$ and $\boldsymbol{v}_t = (1-\beta_2)\sum_{i=1}^t \beta_2^{t-i} \boldsymbol{g}_i^2$. Substituting these into the equation above:

$$
\bar{\boldsymbol{m}}_t = \frac{(1-\beta_3)(1-\beta_1)}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \sum_{j=1}^i \beta_1^{i-j} \boldsymbol{g}_j = \frac{(1-\beta_3)(1-\beta_1)}{(1-\beta_3^t)(\beta_3 - \beta_1)} \sum_{j=1}^t (\beta_3^{t-j+1} - \beta_1^{t-j+1}) \boldsymbol{g}_j \tag{6}
$$

$$
\bar{\boldsymbol{v}}_t = \frac{(1-\beta_3)(1-\beta_2)}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \sum_{j=1}^i \beta_2^{i-j} \boldsymbol{g}_j^2 = \frac{(1-\beta_3)(1-\beta_2)}{(1-\beta_3^t)(\beta_3 - \beta_2)} \sum_{j=1}^t (\beta_3^{t-j+1} - \beta_2^{t-j+1}) \boldsymbol{g}_j^2 \tag{7}
$$

The exchange of summation symbols is based on the identity $\sum_{i=1}^t \sum_{j=1}^i a_i b_j = \sum_{j=1}^t \sum_{i=j}^t a_i b_j$. In summary, we have:

$$
\boldsymbol{\theta}_t = \beta_3^t \boldsymbol{\theta}_0 + (1 - \beta_3^t) (-\bar{\boldsymbol{u}}_t / \lambda) \tag{8}
$$

The weight $\boldsymbol{\theta}_t$ is the training result we want, expressed as a weighted average of $\boldsymbol{\theta}_0$ and $-\bar{\boldsymbol{u}}_t / \lambda$. Here, $\boldsymbol{\theta}_0$ is the initial weight, and $\bar{\boldsymbol{u}}_t$ is data-dependent. Under the mean field approximation, it is approximately equal to $\bar{\boldsymbol{m}}_t / \sqrt{\bar{\boldsymbol{v}}_t}$, and $\bar{\boldsymbol{m}}_t$ and $\bar{\boldsymbol{v}}_t$ can be expressed as weighted sums of the gradients at each step. Taking $\bar{\boldsymbol{m}}_t$ as an example, the weight of the gradient at step $j$ is proportional to $\beta_3^{t-j+1} - \beta_1^{t-j+1}$.

### Memory Period #

We are mainly concerned with pre-training, which is characterized by being **Single-Epoch** (most data is seen only once). Therefore, one of the keys to training good results is not to forget early data. Assuming the training data has been globally shuffled, it is reasonable to assume that the data in every Batch is equally important.

Data is linearly superimposed into $\bar{\boldsymbol{m}}_t$ in the form of gradients. Assuming that the gradient at each step contains only the information of the current Batch, then for a certain Batch not to be forgotten, the coefficient $\beta_3^{t-j+1} - \beta_1^{t-j+1}$ must not be too small. Consider the function $f(s) = \beta_3^s - \beta_1^s$. It is a function that increases first and then decreases, but because $\beta_3$ is closer to 1 than $\beta_1$, the number of increasing steps is small, and in the long range, it is mostly an exponential decay.

Overall, the trend is that the coefficient gets smaller as the distance increases. To ensure the model does not forget every Batch, the coefficient at the farthest point must not be too small. Assuming the coefficient must be no less than $c \in (0, 1)$ to be remembered, when $s$ is large enough, $\beta_1^s$ tends to 0 first, so $\beta_3^s - \beta_1^s \approx \beta_3^s$. From $\beta_3^s \ge c$, we can solve for $s \le \frac{\log c}{\log \beta_3} \approx -\frac{\log c}{\lambda \eta}$. This indicates that the model can at most remember $O(1/\lambda \eta)$ steps of data; this is its **Memory Period**.

So, can we simply set $\lambda=0$ blindly to make the memory period infinite, so we don't have to worry about forgetting? Theoretically yes, but this is not a good choice. Another function of Weight Decay is to help the model forget the initialization. From equation (8), the weight of the initialization $\boldsymbol{\theta}_0$ is $\beta_3^t$. If $\beta_3$ is too large or the training steps $t$ are too small, the proportion of initialization is still high, and the model may still be in an underfitting stage.

In addition, Weight Decay is beneficial for stabilizing the "internal health" of the model. In *«Asymptotic Estimation of AdamW's Weight RMS (Part 1)»*, we derived that the asymptotic result of AdamW's Weight RMS is $\sqrt{\eta / 2\lambda}$. If $\lambda=0$, the Weight RMS will expand at a rate of $\sqrt{\eta t}$. This means that setting $\lambda=0$ directly may bring about internal abnormalities such as weight explosion.

Therefore, $\beta_3$ cannot be too small (to avoid forgetting early data) and cannot be too large (to avoid underfitting or weight explosion). A more suitable setting is to make $1/\lambda \eta$ proportional to the training steps. If it is a Multi-Epoch training scenario, consider making $1/\lambda \eta$ proportional to the training steps of a single Epoch.

### Dynamic Version #

In actual training, we apply dynamically changing LR Schedules, such as Cosine Decay, Linear Decay, WSD (Warmup-Stable-Decay), etc. Therefore, the above results for static Weight Decay and Learning Rate do not completely match practice, and we need to generalize them to the dynamic version.

Starting from equation (3), using the approximation $1 - \lambda_t \eta_t \approx e^{-\lambda_t \eta_t}$, and expanding iteratively, we get:

$$
\boldsymbol{\theta}_t = (1 - \lambda_t \eta_t)\boldsymbol{\theta}_{t-1} - \eta_t \boldsymbol{u}_t \approx e^{-\lambda_t \eta_t} \boldsymbol{\theta}_{t-1} - \eta_t \boldsymbol{u}_t = e^{-\kappa_t} \left( \boldsymbol{\theta}_0 - \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{u}_i \right) \tag{9}
$$

Where $\kappa_t = \sum_{i=1}^t \eta_i \lambda_i$. Continuing to set $z_t = \sum_{i=1}^t e^{\kappa_i} \eta_i$, we can obtain the same mean field approximation:

$$
\bar{\boldsymbol{u}}_t \triangleq \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{u}_i = \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \frac{\boldsymbol{m}_i}{\sqrt{\boldsymbol{v}_i}} \approx \frac{\bar{\boldsymbol{m}}_t}{\sqrt{\bar{\boldsymbol{v}}_t}} \tag{10}
$$

Where $\bar{\boldsymbol{m}}_t \triangleq \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{m}_i$ and $\bar{\boldsymbol{v}}_t \triangleq \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{v}_i$. Substituting the expansions of $\boldsymbol{m}_t, \boldsymbol{v}_t$:

$$
\bar{\boldsymbol{m}}_t = \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{m}_i = \frac{1-\beta_1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \sum_{j=1}^i \beta_1^{i-j} \boldsymbol{g}_j = \sum_{j=1}^t \boldsymbol{g}_j \underbrace{\frac{1-\beta_1}{z_t} \sum_{i=j}^t e^{\kappa_i} \beta_1^{i-j} \eta_i}_{\text{denoted as } \bar{\beta}_1(j,t)} \tag{11}
$$

$$
\bar{\boldsymbol{v}}_t = \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{v}_i = \frac{1-\beta_2}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \sum_{j=1}^i \beta_2^{i-j} \boldsymbol{g}_j^2 = \sum_{j=1}^t \boldsymbol{g}_j^2 \underbrace{\frac{1-\beta_2}{z_t} \sum_{i=j}^t e^{\kappa_i} \beta_2^{i-j} \eta_i}_{\text{denoted as } \bar{\beta}_2(j,t)} \tag{12}
$$

It can be seen that compared to the static Weight Decay and Learning Rate, the dynamic version does not change much in form, except that the weighting coefficients of the gradient become slightly more complex $\bar{\beta}_1(j,t)$ and $\bar{\beta}_2(j,t)$. Specifically, when $\beta_1, \beta_2 \to 0$, $\bar{\beta}_1(j,t)$ and $\bar{\beta}_2(j,t)$ simplify to:

$$
\bar{\beta}_1(j,t) = \bar{\beta}_2(j,t) = \frac{e^{\kappa_j} \eta_j}{z_t} \tag{13}
$$

### Optimal Schedule #

There are many things we can do next. The most basic one is to calculate $\bar{\beta}_1(j,t)$ and $\bar{\beta}_2(j,t)$ and estimate the memory period according to specific WD Schedules and LR Schedules. However, here we choose to do something more extreme—directly reverse-engineer an **Optimal WD Schedule and LR Schedule**.

Specifically, we assumed earlier that the data has been globally shuffled, so the data in each Batch is equally important. However, the coefficient obtained in the static version $\bar{\beta}_1(j,t) \propto \beta_3^{t-j+1} - \beta_1^{t-j+1}$ is not constant, but changes with distance, which does not fully match "each Batch is equally important." If conditions permit, we expect it to be equal to a constant. Based on this expectation, we can solve for the corresponding $\lambda_j, \eta_j$.

For simplicity, we start with $\beta_1, \beta_2 \to 0$. In this case, the expectation condition can be written as $\forall 0 \le i, j \le t, e^{\kappa_i} \eta_i / z_t = e^{\kappa_j} \eta_j / z_t$. Rearranging gives $\eta_i / \eta_j = e^{\kappa_j - \kappa_i}$. Substituting $i = j-1$, we get $\eta_{j-1} / \eta_j = e^{\kappa_j - \kappa_{j-1}} = e^{\lambda_j \eta_j}$, or written as:

$$
\frac{e^{\lambda_j \eta_j}}{\eta_j} = \frac{1}{\eta_{j-1}} \tag{14}
$$
*(Note: There is a slight typo in the original text's equation 14 arrangement, but the derivation implies $\eta_{j-1} = \eta_j e^{\lambda_j \eta_j}$)*

This provides a numerical method for solving $\lambda_j, \eta_j$: after obtaining $\eta_{j-1}$ at each step, $\lambda_j, \eta_j$ can be obtained by solving this nonlinear equation, so the entire sequence can be obtained recursively starting from $\eta_1$. If a more analytical result is desired, derivatives can be used to approximate the difference: taking the logarithm of both sides gives $\lambda_j \eta_j + \log \eta_j - \log \eta_{j-1} = 0$. Treating $\lambda_j, \eta_j$ as continuous functions $\lambda_s, \eta_s$, and $\log \eta_j - \log \eta_{j-1}$ as the derivative approximation of $\log \eta_s$, we have:

$$
\lambda_s \eta_s + \frac{\dot{\eta}_s}{\eta_s} \approx 0 \tag{15}
$$

If $\lambda_s$ is taken as a constant $\lambda$, then we can solve:

$$
\eta_s \approx \frac{\eta_{\max}}{\lambda \eta_{\max} s + 1} \tag{16}
$$

This is the **Best LR Schedule under constant Weight Decay**. It does not require a preset endpoint $t$ and minimum learning rate $\eta_{\min}$, which means it can be trained infinitely, similar to the Stable stage of WSD, but it automatically balances the coefficients of the gradient at each step. However, it also has a drawback: when $s \to \infty$, it tends to 0. From *«Asymptotic Estimation of AdamW's Weight RMS (Part 2)»*, we know that Weight RMS will tend to $\lim_{s \to \infty} \eta_s / 2\lambda_s$, so this drawback may bring the risk of weight collapse.

To solve this problem, we can consider letting $\lambda_s = \alpha \eta_s$, where $\alpha = \lambda_{\max} / \eta_{\max}$ is a constant. In this case, we can solve:

$$
\eta_s \approx \frac{\eta_{\max}}{\sqrt{2 \lambda_{\max} \eta_{\max} s + 1}}, \quad \lambda_s \approx \frac{\lambda_{\max}}{\sqrt{2 \lambda_{\max} \eta_{\max} s + 1}} \tag{17}
$$

Correspondingly, $e^{\kappa_s} \approx \sqrt{2 \lambda_{\max} \eta_{\max} s + 1}$, $e^{\kappa_s} \eta_s \approx \eta_{\max}$, $z_t \approx \eta_{\max} t$, and $\bar{\beta}_1(j,t) = \bar{\beta}_2(j,t) \approx 1/t$.

### General Results #

The current results, such as Eq (16) and Eq (17), are based on $\beta_1, \beta_2 = 0$. Do the results need to change when they are not equal to 0? More generally, the above results are based on the Adam optimizer; to what extent can they be generalized to other optimizers?

First, let's look at the problem when $\beta_1, \beta_2 \neq 0$. The answer is that when $t$ is large enough, the conclusion does not need to be changed much. Taking $\bar{\beta}_1(j,t)$ as an example, under the above optimal schedule, $e^{\kappa_i} \eta_i$ is equal to a constant (related to $t$). Then according to the definition:

$$
\bar{\beta}_1(j,t) = \frac{1-\beta_1}{z_t} \sum_{i=j}^t e^{\kappa_i} \beta_1^{i-j} \eta_i \propto \sum_{i=j}^t \beta_1^{i-j} = \frac{1 - \beta_1^{t-j+1}}{1-\beta_1} \tag{18}
$$

When $t$ is large enough, $\beta_1^{t-j+1} \to 0$, so this can also be seen as a constant independent of $j$. As mentioned earlier, for $\beta_1, \beta_2$, the fact that "$t$ is large enough" is almost certain, so the results for $\beta_1, \beta_2 = 0$ can be used directly.

As for the optimizers, the ones we mentioned earlier are SGDM, RMSProp, Adam, SignSGDM, and Muon, which can be divided into two categories. Among them, SGDM is one category; its $\bar{\boldsymbol{u}}_t$ is directly $\bar{\boldsymbol{m}}_t$, and does not even need to use the mean field approximation, so the results up to Eq (15) are applicable. However, Eq (16) and Eq (17) are probably not the most suitable, because the asymptotic Weight RMS of SGDM also depends on the gradient norm [Reference], so the gradient norm needs to be taken into account, which is relatively more complicated.

The remaining RMSProp, Adam, SignSGDM, and Muon are classified into another category, all belonging to adaptive learning rate optimizers. Their update rules all have the homogeneous form of $\text{gradient} / \sqrt{\text{gradient}^2}$. In this case, if we still believe in the mean field approximation, we can get the same $\bar{\boldsymbol{m}}_t$, same $\beta_1(j,t)$, so the results up to Eq (15) are also applicable; and for this type of homogeneous optimizer, it can be proven that Weight RMS is also asymptotically proportional to $\sqrt{\eta / \lambda}$, so Eq (16) and Eq (17) can also be reused.

### Hypothesis Discussion #

Our derivation has come to a temporary halt. In this section, we will discuss the assumptions relied upon in the derivation.

Throughout the text, there are two main large assumptions used in the derivation worth discussing. The first assumption is the **Mean Field Approximation**, first introduced in *«Rethinking Learning Rate and Batch Size (Part 2): Mean Field»*. The Mean Field itself is certainly not new; it is a classic approximation in physics. However, using it to analyze the dynamics of optimizers appears to be a first by the author. Currently, it has been used to estimate the Batch Size, Update RMS, Weight RMS, etc., of optimizers, and the results seem reasonable.

Regarding the validity of the Mean Field Approximation, we cannot comment too much; it reflects a kind of belief. On the one hand, based on the reasonableness of existing estimation results, we believe it will continue to be reasonable, at least providing valid asymptotic estimates for some scalar indicators. On the other hand, for adaptive learning rate optimizers, due to the nonlinearity of their update rules, the difficulty of analysis increases greatly. Apart from the Mean Field Approximation, we actually have no other calculation tools to use.

The most typical example is Muon. Because it is a non-Element-wise operation, previous calculation methods like those for SignSGD that work component-wise lose their effect, while the Mean Field Approximation still works (refer to *«Rethinking Learning Rate and Batch Size (Part 3): Muon»*). Therefore, the Mean Field Approximation actually provides a unified, effective, and concise calculation method for the analysis and estimation of a large class of adaptive learning rate optimizers. Currently, there seems to be no other method with the same effect, so we can only continue to believe in it.

The second major assumption is **"the gradient at each step only contains the information of the current Batch."** This assumption is inherently wrong because the gradient depends not only on the data of the current Batch but also on the parameters of the previous step, and the parameters of the previous step naturally contain historical information. However, we can try to remedy this. Theoretically speaking, every Batch brings new information; otherwise, this Batch would have no meaning. So the remedy is to change it to "the gradient at each step contains roughly the same incremental information."

Of course, upon careful reflection, this statement is also controversial, because the more you learn and the wider the coverage, the less unique information later Batches have. However, we can struggle a bit more, that is, divide knowledge into two major categories: "laws" and "facts." Factual knowledge—for example, that a certain theorem was discovered by a certain mathematician—can only be relied upon by memory. So we can consider changing it to "the gradient at each step contains roughly the same factual knowledge." In short, from practice, the LR Schedule obtained by "treating every gradient step equally" seems to really have benefits, so we can always try to construct an explanation for it.

The recent paper *«How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining»* provides indirect evidence. It considered curriculum learning where data quality goes from low to high and found that aggressive LR Decay would negate the advantages of curriculum learning. Our result is that the weight of each Batch is Eq (13), proportional to the Learning Rate. If LR Decay is too aggressive, the weight of the high-quality data later on will be too small, resulting in poor performance. Being able to reasonably explain this phenomenon conversely shows the rationality of our assumption.

### Summary #

This article understands Weight Decay (WD) and Learning Rate (LR) from the perspective of sliding average, and explores the optimal WD Schedule and LR Schedule under this perspective.