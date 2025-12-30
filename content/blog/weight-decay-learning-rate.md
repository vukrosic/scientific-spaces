---
title: "Weight Decay and Learning Rate from a Moving Average Perspective"
date: "2025-12-05"
author: "Jianlin Su"
excerpt: "Exploring the relationship between Weight Decay and Learning Rate by reinterpreting the model parameter update rule as an Exponential Moving Average (EMA)."
tags: ["Mathematics", "Deep Learning", "Optimization", "LLM"]
---

Weight Decay (WD) and Learning Rate (LR) are fundamental components of LLM pre-training. Their proper configuration is often the deciding factor in a model's success. Since the introduction of [AdamW](https://papers.cool/arxiv/1711.05101), decoupling Weight Decay from traditional L2 regularization has become standard practice. However, beyond this decoupling, there has been limited theoretical progress on how to optimally set these two parameters.

In this post, I will share a new perspective: viewing the training process as a moving average memory of the training data. From this viewpoint, we can discuss how to set Weight Decay and Learning Rate more scientifically.

## Moving Average Perspective

The general form of Weight Decay is:
\begin{equation}\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t + \lambda_t \boldsymbol{\theta}_{t-1})\end{equation}
Where $\boldsymbol{\theta}$ represents the parameters, $\boldsymbol{u}$ is the update vector from the optimizer, and $\lambda_t, \eta_t$ are the Weight Decay and Learning Rate at step $t$. We refer to the sequences $\{\lambda_t\}$ and $\{\eta_t\}$ as the "WD Schedule" and "LR Schedule" respectively.

We can rewrite this update rule as:
\begin{equation}\boldsymbol{\theta}_t = (1 - \lambda_t \eta_t)\boldsymbol{\theta}_{t-1} - \eta_t \boldsymbol{u}_t = (1 - \lambda_t \eta_t)\boldsymbol{\theta}_{t-1} + \lambda_t \eta_t ( -\boldsymbol{u}_t / \lambda_t)\label{eq:wd-ema}\end{equation}
In this form, Weight Decay manifests as a weighted average (Exponential Moving Average, EMA) between the current parameters and the "target" update $-\boldsymbol{u}_t / \lambda_t$. This perspective is not entirely new—articles like [*How to set AdamW's weight decay as you scale model and dataset size*](https://papers.cool/arxiv/2405.13698) have discussed it—but here we will dive deeper into the implications.

## Iterative Expansion

For simplicity, let's first consider constant $\lambda$ and $\eta$. Let $\beta_3 = 1 - \lambda\eta$. Then:
\begin{equation}\boldsymbol{\theta}_t = \beta_3 \boldsymbol{\theta}_{t-1} + (1 - \beta_3)( -\boldsymbol{u}_t / \lambda)\end{equation}
Expanding this yields:
\begin{equation}\boldsymbol{\theta}_t = \beta_3^t \boldsymbol{\theta}_0 + (1 - \beta_3)\sum_{i=1}^t \beta_3^{t-i} (-\boldsymbol{u}_i / \lambda) \end{equation}
The final weights $\boldsymbol{\theta}_t$ are a weighted average of the initial weights $\boldsymbol{\theta}_0$ and the data-dependent updates. 

## The Memory Cycle

In LLM pre-training, which is typically "Single-Epoch," we only see most data once. A key to success is not forgetting early data. If we assume each batch is equally important, the weights in the sum above shouldn't decay too rapidly.

The "Memory Cycle" of the model is approximately $\mathcal{O}(1/\lambda\eta)$. 
- If $\lambda\eta$ is too large, the memory window is too small, and the model "forgets" the beginning of the dataset.
- If $\lambda$ is too small (e.g., 0), the memory is infinite, but we risk retaining too much of the initialization $\boldsymbol{\theta}_0$ (underfitting) and potential weight explosion (Weight RMS grows as $\sqrt{t}$).

For AdamW, the steady-state Weight RMS is approximately:
\begin{equation}\text{RMS}(\boldsymbol{\theta}) \approx \sqrt{\frac{\eta}{2\lambda}}\end{equation}
This confirms that $\lambda$ is necessary to keep the weights stable.

## Optimal Scheduling

What if we want every batch to contribute equally to the final parameters? In the dynamic case where $\eta_t$ and $\lambda_t$ vary, the update rule becomes:
\begin{equation}\boldsymbol{\theta}_t \approx e^{-\kappa_t}\left(\boldsymbol{\theta}_0 - \sum_{i=1}^t e^{\kappa_i}\eta_i\boldsymbol{u}_i\right)\end{equation}
Where $\kappa_t = \sum_{i=1}^t \eta_i\lambda_i$. 

For the $j$-th update to have a constant weight in the final result, we require $e^{\kappa_j}\eta_j$ to be constant. Using a continuous approximation, this leads to the differential equation:
\begin{equation}\lambda_s \eta_s + \frac{\dot{\eta}_s}{\eta_s} \approx 0 \label{eq:lr-wd-ode}\end{equation}

If Weight Decay $\lambda$ is constant, the optimal learning rate schedule is:
\begin{equation}\eta_s \approx \frac{\eta_{\max}}{\lambda\eta_{\max} s + 1}\end{equation}
This schedule automatically balances the importance of each gradient step without needing to pre-define a total step count $t$.

## Conclusion

By viewing Weight Decay through the lens of a moving average, we see that it defines the "memory" of our model. The product $\lambda \eta$ determines how much past information is retained versus forgotten. Balancing this "memory cycle" with the need for regularization and training stability is key to designing better optimization schedules.
