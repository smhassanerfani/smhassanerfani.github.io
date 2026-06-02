# On the Genealogy of Machine Learning Weather Prediction (MLWP)

## Physics Inherited, Data Forgotten — Toward a Principled Trade-off in Surrogate Modeling

## Introduciton
Traditionally, physics has been responsible for explaining the Earth as a dynamical system. Fundamental physical laws such as Newton’s Second Law of Motion and the First Law of Thermodynamics govern the evolution of the atmosphere. These laws are converted into mathematical equations that form the foundation of numerical weather prediction (NWP).

Because physics has historically dominated weather prediction, machine learning weather prediction (MLWP) inherited not only atmospheric datasets and forecasting targets from NWP, but also many of its conceptual assumptions. In particular, modern MLWP frequently inherits the scientific formulation of weather prediction as an initial-value problem (IVP). Consequently, many ML models are designed as learned time-stepping systems that autoregressively evolve atmospheric states forward in time. This inheritance is not necessarily misguided. Weather prediction is fundamentally temporal, and any successful MLWP system must respect the time-evolving nature of the atmosphere. However, the inherited framing also constrains the way the learning problem itself is defined. Instead of first asking what the structure of the data suggests from a machine learning perspective, MLWP often begins by reproducing the ontology of numerical solvers: prognostic variables, diagnostic variables, tendencies, and time-marching operators.

This article argues that modern MLWP sits between two different traditions:

1. Scientific surrogate modeling, where machine learning is embedded inside an existing physical system and must preserve its structure.
2. Free-form data-driven modeling, where the task is defined primarily by the statistical structure of the data itself.

The central claim of this article is therefore genealogical rather than purely methodological. 
*Modern MLWP inherited its dominant autoregressive rollout paradigm from the historical construction of NWP as an IVP. If the same atmospheric data were presented without their physical semantics, i.e., simply as high-dimensional spatiotemporal tensor sequences, the resulting modeling instincts might look very different.*

The purpose is not to reject physics-informed MLWP. Rather, it is to make this inheritance explicit so that we can ask a sharper question:

*Are we constructing a scientific surrogate that should preserve the ontology of the governing system, or are we constructing a free-form spatiotemporal learner chosen primarily from the structure of the data?*

Here we try to answer the question by looking at (i) the results that have already been reported in the literature and (ii) what we know from the mathematical formulation of the differential equations used to describe the atmosphere or any other natrual system.

First, we restate NWP in three basic pieces: (1) **numerical integration** of the governing equations, (2) **prognostic variables** that are stepped forward in time, and (3) **diagnostic (or closure) operators** that compute derived fields.  The goal is not to change the physics; we are simply rewriting it in terms that map cleanly onto a machine‑learning framework.  This gives us a common vocabulary for talking about a principled trade‑off between “physics‑first” surrogate models and purely data‑driven models.

Second, we consider how the data‑driven side defines its tasks.  By separating **prognostic** operators (the ones that evolve the state) from **diagnostic/closure** operators (the ones that evaluate constraints or derived quantities), we can see where a model should honor physical laws and where it can be more flexible and learn directly from the data.  This distinction helps us choose model architectures, loss functions, and evaluation metrics.

Finally, we look at who built today’s ML‑weather models, why they ended up looking like learned NWP time‑steppers, and how they differ from generic spatiotemporal sequence models.  Tracing this history shows how the inherited IVP framing from NWP has shaped current practice and points to the need for a more balanced view that also embraces data‑centric modeling.

## Numerical Weather Prediction
The following overview is intentionally schematic. Its purpose is not to reproduce the full complexity of operational NWP, but to isolate the parts of the system that matter for the surrogate-modeling argument.

Vilhelm Bjerknes first recognized that numerical weather prediction was possible in principle in 1904. He proposed that weather prediction could be viewed as an **initial value problem (IVT)** in mathematics: since physical laws govern how meteorological variables evolve over time, if we possess an accurate representation of the atmosphere’s initial state, we can numerically integrate these governing equations forward in time to generate a forecast.

At its core, NWP involves solving a set of partial differential equations, commonly referred to as the **Primitive Equations**. These equations are designed to resolve six fundamental resolved variables: three-dimensional wind velocity components ($u, v, \omega$), temperature ($T$), moisture ($q$), and geopotential height ($z$).

### The Primitive Equations
The following system serves as the foundational framework for atmospheric motion and thermodynamics:

#### Wind Forecast Equations

**1a.** $$\frac{\partial u}{\partial t} = - u \frac{\partial u}{\partial x} - v \frac{\partial u}{\partial y} - \omega \frac{\partial u}{\partial p} + fv - g \frac{\partial z}{\partial x} + F_x$$

**1b.** $$\frac{\partial v}{\partial t} = - u \frac{\partial v}{\partial x} - v \frac{\partial v}{\partial y} - \omega \frac{\partial v}{\partial p} - fu - g \frac{\partial z}{\partial y} + F_y$$

#### Continuity Equation

**2.** $$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial \omega}{\partial p} = 0$$

#### Temperature Forecast Equation

**3.** $$\frac{\partial T}{\partial t} = - u \frac{\partial T}{\partial x} - v \frac{\partial T}{\partial y} - \omega \left( \frac{\partial T}{\partial p} - \frac{RT}{c_p p} \right) + \frac{H}{c_p}$$

#### Moisture Forecast Equation

**4.** $$\frac{\partial q}{\partial t} = - u \frac{\partial q}{\partial x} - v \frac{\partial q}{\partial y} - \omega \frac{\partial q}{\partial p} + E - P$$

#### Hydrostatic Equation

**5.** $$\frac{\partial z}{\partial p} = - \frac{RT}{pg}$$

*Source: MetEd course: Impact of Model Structure and Dynamics*

<!-- ### From Theory to Numerical Integration -->
Because these non-linear partial differential equations do not possess closed-form analytical solutions, we must rely on numerical schemes to solve them. In practice, solving these equations is a process of discrete integration over time and space. We can represent the essence of this integration using a simple **Euler forward scheme**.

If $\psi$ represents any of our resolved variables, the state at time $t + \Delta t$ can be approximated by the current state and its time tendency:

$$\psi(t + \Delta t) \approx \psi(t) + \left( \frac{\partial \psi}{\partial t} \right) \Delta t$$

In this framework, the "model" acts as an engine that calculates the tendency term ($\frac{\partial \psi}{\partial t}$) using the physical laws shown above, then iteratively updates the state of the atmosphere. This time-stepping structure strongly influenced the dominant formulation of MLWP. A learned model of the form

$$
X_{t+\Delta t} = \mathcal{M}_{\theta}(X_t)
$$

behaves analogously to a learned numerical integrator: it repeatedly applies a transition operator to evolve the atmospheric state.

The important point is not that autoregression is incorrect. For evolving prognostic states, autoregression is often physically justified. The deeper issue is genealogical:

> MLWP inherited this rollout structure from numerical integration before asking whether the statistical structure of the data alone would naturally suggest the same formulation.

### Physical Processes
In the set of NWP equations, some variables, specifically $F_x, F_y, H, E,$ and $P$, represent physical processes that impact our primary variables. These processes are inherently complex; they often involve scales far smaller than the grid spacing of the model (such as individual convective clouds) or rely on physical mechanisms (like radiation transfer) that are too computationally expensive to resolve from first principles.

Because we cannot calculate these effects directly within the core equations, we must estimate them using empirical approximations. In numerical modeling, this technical estimation process is known as **parameterization**. The accuracy of an NWP forecast is fundamentally linked to how well these parameterizations mimic reality.

For surrogate‑modeling purposes, parameterizations are especially notable because they already sit at the intersection of physics and empiricism. They are not independent atmospheric states; rather, they are evaluated from the current model state often using additional constants, closure assumptions, or internal variables and then inserted into the tendency equations. In this sense they can be viewed as dependent variables that are functions of the resolved variables. Recognizing parameterizations as functions of the resolved state clarifies the mapping between predictor variables (the "prognostic" fields that the model directly resolves) and predictand variables (the tendencies, "diagnostics," or closure terms produced by the parameterizations). 

This observation opens a new perspective for the article: the distinction between prognostic and diagnostic variables. A critical difference in physical modeling lies in how these two types of quantities are treated.

* **Prognostic Variables ($u, v, T, q$):** These are our "state" variables. These variables possess explicit time-evolution equations. Their future values are obtained through numerical integration.
* **Diagnostic Variables ($\omega, z$):** Diagnostic variables occupy a different role. Rather than evolving independently through explicit time derivatives, they are evaluated from the current atmospheric state, i.e., they are derived directly from the prognostic variables at any given time step.

For simplicity we also categorize all quantities produced by parameterized physical processes as diagnostic or closure operators. Like other diagnostics, they are calculated from the current prognostic state. But this does not imply that they are always purely algebraic; some closures may incorporate memory, stochastic components, or internal state variables. 
better understanding of prognostics and diagnostics and their characteristics can help us to better define prediction task in ML side. 

## Machine Learning
> **Disclaimer:** The data-driven landscape is more amorphous than traditional physics, defined largely by experimental conventions and evolving best practices. The ML terminology in this article is pragmatic rather than taxonomic. we choose terms in a way that help us better make our cases and help reader to easier follow what we are talking about and facilitate communiting between two realms (physics and data-driven)

Choosing the right machine learning model requires a systematic approach that balances problem type, data characteristics, and practical constraints. Given what has been discussed in the previous sections, MLWP—whether at the dynamical core level or for specific physical processes—is generally framed as a **supervised learning** and **regression** problem. However, the ML literature often uses the word *prediction* broadly, which can obscure an important distinction relevant to MLWP.

This article therefore distinguishes between:

1. **Evolution operator learning**
2. **State-conditioned operator evaluation**

### Evolution Operator Learning
Evolution operator learning concerns estimating the future state of a dynamical system. This can take the form of a transition operator,

$$X_{t+\Delta t} = \mathcal{M}_{\theta}(X_t),$$

or a tendency operator,

$$X_{t+\Delta t} = X_t + \Delta t\,\mathcal{F}_{\theta}(X_t),\qquad \frac{dX}{dt} = \mathcal{F}_{\theta}(X_t)$$

Both formulations can be deployed autoregressively, since the output updates the evolving state. In other words, the model approximates either the transition or the tendency operator of a dynamical system. 

This framing is not exclusive to the physical sciences. Hydrological time series forecasting, for example, applies the same principle, using statistical and machine learning methods to predict streamflow, rainfall, and groundwater levels from sequentially measured records. This does not imply that the variable is purely dependent on its own past values, but this approach is adopted pragmatically, justified by two key arguments: (1) understanding the variability of hydrological processes is inherently difficult due to their complex and stochastic nature, making explicit driver-based modeling challenging; and (2) the target variable carries implicit correlations with its physical drivers, since those drivers are themselves autocorrelated in time, i.e., past precipitation, soil moisture, and temperature leave detectable signatures in the runoff record.

### State-Conditioned Operator Evaluation
The second formulation is the simpler of the two—a standard regression mapping:

$$Y_t = g_{\theta}(X_t).$$

At its core, this is a functional mapping from inputs to outputs at a **single time level**. Unlike evolution operator learning, regression here treats data points as independent observations within a feature space, with no inherent requirement that the target must lie in the future. The goal is to learn a mapping function rather than to propagate a temporal trajectory.

Unlike evolution operator learning, which is inherently sequential and relies on the temporal memory of the system's own state, state-conditioned operator evaluation is fully cross-sectional with respect to the time domain, conditioning predictions exclusively on concurrent driver variables rather than on the history of the target variable itself.

### Multivariate Time Series (Adding the Drivers)
Time series analysis need not operate in isolation from the physical system. In multivariate approaches, the physical drivers, the exogenous variables, are explicitly incorporated into the framework:

$$y_t = \beta_1 {x_1}_t + \beta_2 {x_2}_t + \dots + \beta_n {x_n}_t + \phi\, y_{t-1} + \epsilon_t$$

This allows the model to respond to external physical forcings $(x_{1,t}, x_{2,t}, \dots, x_{n,t})$ while the autoregressive term $\phi\, y_{t-1}$ accounts for temporal memory and delayed system response. In machine learning, this formulation can be generalized through a parametric state-update equation:

$$h_t = f_\delta(X_t,\, h_{t-1})$$

where $X_t$ represents all driver variables at the current time step, and $h_t$ encodes whatever information is inherited from previous time steps, typically in a latent space that can accumulate context over a sequence length extending well beyond a single lag.

---

the second important aspect is important aspect about the data characteristics is to understand its dependency whether in time or space. simply we can say MLWP data is spatiotemporal, ie having dependency on both time and space domains. so based on these we can define the task for ML. 

<!-- 
<span style="color:#137333">Choosing a machine learning model requires balancing the physical role of the target, the structure of the data, and practical constraints such as stability, resolution, and computational cost. For MLWP, the most important first question is: are we learning an evolution operator for the state, or are we learning an operator evaluated from the state?</span> -->

However!, problem is who is supposed to defin the task? If the task is defined by physical science, then the data are not just tensors; they are atmospheric states, diagnostic variables, tendencies, closures, and constraints. If the task is defined in a free-form data-driven context, then the same object may be described simply as a multichannel image sequence. These two descriptions can lead to different modeling instincts.

Under the scientific formulation, current MLWP often learns a one-step transition operator and rolls it forward:

$$X_{t+\Delta t} = \mathcal{M}_{\theta}(X_t), \qquad X_{t+n\Delta t} = \mathcal{M}_{\theta}^{(n)}(X_t).$$

This mirrors NWP time stepping. 

Under a free-form spatiotemporal formulation, however, one may instead define the problem as sequence-to-sequence prediction:
<!-- TODO change the formula to recurrence formula -->

$$\{X_{t-k}, \ldots, X_t\} \mapsto \{X_{t+\Delta t}, \ldots, X_{t+m\Delta t}\}.$$

A ConvLSTM, for example, may still generate future frames autoregressively, but it treats the problem as latent spatiotemporal pattern evolution rather than as an explicit learned numerical time-stepper. 

This is the sense in which the genealogy matters: MLWP's dominant rollout strategy is inherited from the physical construction of weather prediction, not simply selected from generic data characteristics.

### Time-Series vs Regression Analysis
the second seciton of ML focus on the second issue, as discussed before to facilitate communiting between two realms, here we discriminate between time-series and regression analysis and define them equivallently with forecasting and prediciton, respectively.

In this article, we define forecasting as estimating future states based on the history of a time series. The defining characteristic is the **temporal anchor**. We rely on the autocorrelation of the system—the principle that the state at time $t$ fundamentally influences the state at $t+1$. The objective is to project the known trajectory of the atmosphere forward into an unknown future, maintaining the continuity of the system's evolution.

In MLWP, this corresponds to learning a transition operator such as</span>

$$X_{t+\Delta t} = \mathcal{M}_{\theta}(X_t),$$

or a tendency operator such as

$$\frac{dX}{dt} = \mathcal{F}_{\theta}(X_t), \qquad X_{t+\Delta t} = X_t + \Delta t\,\mathcal{F}_{\theta}(X_t).$$

Both formulations may be deployed autoregressively because the output updates the evolving state. But this is a specific scientific kind of autoregression: the model approximates either the transition operator or the tendency operator of a dynamical system.

We define diagnostic prediction as estimating a quantity $Y_t$ from the instantaneous state $X_t$:

$$Y_t = g_{\theta}(X_t).$$

At its core, this is a functional mapping. Unlike time-series analysis, regression often treats data points as independent observations within a feature space, without an inherent requirement that the target must exist in the future, ie, it is an operator evaluation at a single time level.
the goal is to learn a mapping function rather than project a temporal trajectory.
The output may affect the future indirectly by entering a tendency equation, but it does not possess an independent time-integration rule unless the model explicitly gives it one. 
<!-- TODO: it should get integrated, now the tone is not coherent -->

By distinguishing between **forecasting** (projecting the trajectory forward via temporal correlation) and **prediction** (mapping inputs to outputs within a state space), we can better classify which components of the NWP system belong to which ML approach. This is the crux of our "principled trade-off": knowing whether our surrogate should be acting as a time-evolving forecaster or a diagnostic predictor.

## Prognostics vs. Diagnostics in MLWP: A Misunderstood Distinction

A critical yet frequently overlooked distinction in the geosciences community concerns the appropriate machine learning paradigm for prognostic versus diagnostic variables, and conflating the two leads to both conceptual and methodological errors. Prognostic variables — such as wind components, temperature, or moisture — are governed by PDEs containing explicit time derivatives, meaning their evolution constitutes an **Initial Value Problem (IVP)**. Their estimation is inherently a **forecasting** task: the future state $\psi(t + \Delta t)$ is obtained by numerically integrating the tendency $F(\psi)$ forward in time. 

<!-- Among the ML methods, spatiotemporal task (wether autoregressive rollout or recurrent approach) is the natural and physically justified ML paradigm for this task — whether the surrogate emulates only the tendency operator (partial integration) or the complete one-step transition (full integration), the autoregressive structure faithfully mirrors the Markovian, time-marching nature of the underlying physics. -->

<!-- <span style="color:#137333">A critical but often blurred distinction in MLWP concerns the physical role of prognostic versus diagnostic quantities. Prognostic variables—such as wind components, temperature, or moisture—are governed by equations containing explicit time derivatives. Their evolution constitutes an **initial-value problem (IVP)**. A model that predicts their future value, or their tendency, is approximating a time-evolution operator. Autoregressive rollout is therefore physically justified when the learned output is the evolving atmospheric state or a tendency used to update that state.</span> -->

<!-- <span style="color:#b00020"><s>Diagnostic variables, however, are categorically different: they carry no time derivative, possess no memory, and are determined entirely by an instantaneous functional mapping of the current prognostic state, i.e., D(t) = g(ψ(t)). Their estimation is therefore fundamentally a **regression** (prediction) task, not a forecasting task — there is no temporal constraint, no IVP to solve, and no integration to emulate.</s></span> -->

Diagnostic variables and many closure terms have a different role. In the simplest case, they are determined by an instantaneous functional mapping of the current prognostic state:

$$D_t = g(\psi_t).$$

Their estimation is therefore better understood as prediction operator rather than state forecasting. The practical implication is not that such quantities lack temporal correlation in observed data but their temporal dependence is inherited from the prognostic state, not from an independent evolution law.

To illustrate this distinction, consider the one-dimensional shallow water equations. These equations express conservation of mass and momentum for a depth-averaged, free-surface flow. In conservation form they read:

$$
\frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} = 0
$$

$$
\frac{\partial (hu)}{\partial t} + \frac{\partial}{\partial x}\!\left(hu^2 + \tfrac{1}{2}gh^2\right) = -gh\,\frac{\partial z_b}{\partial x} - S_f
$$

where

- $h$ — water depth $[\text{m}]$
- $u$ — depth-averaged velocity $[\text{m/s}]$
- $g$ — gravitational acceleration $[\text{m/s}^2]$
- $z_b$ — bed elevation $[\text{m}]$
- $S_f$ — friction slope (source term) encoding energy losses due to bed resistance

For many friction-loss formulations, the friction slope is written in a general power-law form:

$$
S_f = \frac{C\,V\,|V|^{m-1}}{R^p}
$$

where $C$ and $p$ are coefficients determined by the chosen friction law, $V$ is the flow velocity magnitude, $R$ is the hydraulic radius, and $m$ depends on the flow regime.

In the SWE, the prognostic variables are the conserved quantities $h$ and $hu$, which are marched forward in time. The friction slope $S_f$, by contrast, is a diagnostic source term calculated from the current flow state and bed geometry. This makes $S_f$ a natural regression target rather than a forecasting variable.

<!-- A concrete example is the friction slope in the Shallow Water Equations (SWE): it is a diagnostic quantity computed algebraically from the current flow state, including the prognostic depth $h$ through the hydraulic radius $R$. The term $R$ is typically a function of flow depth and channel geometry, so $S_f$ depends on both the velocity and the current state of $h$. It then feeds back into the momentum equation to forecast $hu$ at the next time step. The proper ML treatment here is therefore a direct functional mapping from the instantaneous prognostic state $(h(t),u(t))$ to the diagnostic friction slope $S_f(t)$, making it a regression problem contained within a single time level. -->

A second example comes from atmospheric tracer transport. When numerically integrating the continuity equation for a tracer $\mu$ over a grid, the operator is typically split into horizontal advection and vertical convection. While horizontal subgrid-scale closure is generally ignored, vertical convection must be parameterized, yielding the equation:

$$
\frac{d\mu}{dt} + u\frac{d\mu}{dx} + v\frac{d\mu}{dy} + w\frac{d\mu}{dz} = \Sigma
$$

as we dicussed before, the vertical velocity $w$ is a diagnostics and is function of updraft $\omega$, temperature $T$, specific humidity $q$, and geopotential height $z$. i.e., $w = f(\omega, T, q, z)$.

apart from parameterizing just the vertical components, if we wanna surrogate the whole system in Neural network, we can consider the overall instantaneous time derivative (tendency) as follows 

$$
\frac{d\mu}{dt} = f(\mu, u, v, \omega, T, q, z, \dots; \theta)
$$

In this formulation, the neural network $f(\cdot; \theta)$ performs a pure regression task—mapping the current, instantaneous state variables to the diagnostic tendency $\frac{d\mu}{dt}$. Because this mapping possesses no memory, it is a regression problem contained entirely within a single time level. Once this diagnostic regression outputs the approximated tendency $\Delta\mu_t$, the prognostic tracer $\mu$ is marched forward in time via an integration scheme, such as an Euler step $\mu_{t+1} = \mu_t + \frac{d\mu}{dt}$. in the bigger lense, we can see the overall itendency as a parametrization inside the bigger formula (NWP) which require the prognostics ingredients as well as some other spices.

## Embedded Surrogates vs. Free-Form Data-Driven Models

In this work, we distinguish between two fundamentally different uses of data-driven models in scientific systems. Both aim to replace an expensive function with a cheaper approximation, but they differ in key ways. The first approach involves building a computationally efficient approximate model that mimics a more expensive or complex system, such as a CFD simulation, climate model, or physical experiment. This is like physics-aware substitution, where the goal is to replace a specific component of a known governing model while respecting its overall structure, constraints, and meaning. The ultimate aim is efficiency without breaking the model. In contrast, the second approach treats the original system as an unknown function. Here, we do not focus on internal mechanisms, governing equations, or physical interpretation. This is like context-agnostic learning, with no commitment to physical meaning. The goal is to learn patterns, not the system itself. Hereinafter, we call the first approach embedded surrogate modeling, where machine learning is integrated inside a governing system. The second is called free-form data-driven modeling, with no physical constraints or system knowledge.

Choosing the best machine learning model requires a systematic approach that balances the problem type (first approach) with data characteristics (second approach).

Both approaches can be useful, but they should not be evaluated by the same conceptual standard. Embedded surrogates are judged by whether they preserve the meaning and stability of the larger system. Free-form models are judged by whether they forecast well from the data representation they are given. MLWP often sits between these extremes, which is why clarity about the target's physical role matters.

<!-- The central tension in MLWP is that many benchmark datasets, such as WeatherBench-style gridded reanalysis datasets, look like free-form tensor sequences to ML, but they are assembled from variables whose meaning comes from NWP and atmospheric dynamics. -->

## Why Genealogy Matters

So far, this discussion may sound like an attempt to build a taxonomy or introduce new terminology. However, I want to emphasize that this is not our goal. Establishing a formal taxonomy requires broad agreement within the community, and any such effort should align with existing conventions. We intentionally avoid that kind of pedantry and its associated pitfalls. Instead, our aim is simply to clarify the perspective we are taking. From this standpoint, some components appear conceptually distorted, and we will briefly explain how we interpret them so that we are on the same page before introducing the core problem.

The issue arises when we refer to a *data-driven method*. Ideally, this would mean a method dictated by the intrinsic characteristics of the data itself. However, this is not quite what happens in machine learning–based weather prediction. If we present the data to domain experts and ask—strictly in a context-free, domain-agnostic manner—what class of models would be appropriate, very few would suggest an auto-regressive rollout approach. Instead, they would typically propose spatiotemporal modeling strategies.

Modern MLWP inherited the autoregressive rollout not solely from machine learning sequence models, but also from the NWP view of weather as an initial-value problem. Reanalysis datasets provide gridded atmospheric states at successive times; numerical models march states forward; ML models are then trained to imitate the observed or analyzed transition from $X_t$ to $X_{t+\Delta t}$. This genealogy explains why state-to-state autoregression became the dominant template. In this setting, a simple autoregressive rollout is not an arbitrary ML choice. It is the learned analogue of a numerical weather model's time step.

But if we remove the physical semantics and describe the same object only as a multichannel gridded tensor sequence, the modeling instinct changes. A free-form ML formulation would emphasize spatiotemporal representation learning: ConvLSTM, encoder-decoder recurrent networks, temporal convolutional networks, video-prediction models, or transformer-based sequence models. A ConvLSTM can still be autoregressive, but its recurrence is organized around latent spatiotemporal memory and frame evolution, not around an explicit approximation of an NWP time-step operator. Therefore, saying "both are autoregressive" hides the important difference: one autoregression is genealogically tied to numerical integration, while the other is tied to generic sequence modeling.

At this point, the question of *genealogy* becomes important. We need to step back and examine how the current approach emerged: when it started, how it developed, and how this apparent deviation from a purely data-driven perspective came about.


> **Prompt:** Based on the following description of my data, what type of ML model would you recommend? Do not consider what we have already discussed in our previous chats.
>
> Consider a high-dimensional, regularly gridded array-valued signal evolving over discrete time steps. At each time $t$, the system state is represented as a multi-channel tensor
> $$X_t \in \mathbb{R}^{C \times H \times W},$$
>
> where $C$ denotes the number of channels (features), and $H \times W$ defines a fixed 2D lattice. Given a sequence of past states $\{X_{t-k}, \dots, X_t\}$, the objective is to learn a mapping that predicts one or more future states $\{X_{t+1}, \dots, X_{t+m}\}$.
> The data exhibit strong spatiotemporal dependencies, including:
>
> - Local correlations across neighboring grid points
> - Nonlocal interactions across distant regions
> - Temporal continuity with both short-term and long-range dependencies
> - Multi-scale structure, where patterns evolve at different spatial and temporal resolutions
>
> The learning task can be formulated either as:
>
> - Direct state prediction: learning $X_{t+1} = f(X_t)$, or
> - Increment prediction: learning $\Delta X_t = g(X_t)$ such that $X_{t+1} = X_t + \Delta X_t$
>
> No assumptions are made about the underlying generative process, the semantics of the channels, or any governing equations. The problem is treated purely as learning a mapping between sequences of high-dimensional tensors, analogous to sequence modeling or video prediction tasks.

## Conclusion

<span style="color:#1a73e8">The article therefore makes two claims. The first is a modeling claim: MLWP should distinguish prognostic state variables from diagnostic variables, closures, and parameterized tendencies. These quantities may live in the same dataset, but they do not play the same role inside the physical system. The second is a genealogical claim: the simple autoregressive rollout used in much of modern MLWP is inherited from the NWP framing of weather as an initial-value problem. In a free-form ML context, the same gridded tensor sequence could motivate a different family of models, such as ConvLSTM or other spatiotemporal sequence architectures. These models may also be autoregressive, but they are autoregressive in a different sense: they model sequence evolution through learned spatiotemporal memory rather than imitating the time step of a numerical weather model.</span>

<span style="color:#1a73e8">The purpose is not to reject NWP-style MLWP. It is to make the inheritance explicit. Once the genealogy is visible, we can ask a sharper question: are we building a scientific surrogate that should preserve the structure of the governing system, or are we building a free-form data-driven model that should be chosen from the structure of the data alone?</span>
