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

Choosing the right machine learning model requires a systematic approach that balances problem type, data characteristics, and practical constraints such as stability, resolution, and computational cost.

### Problem Characteristics
Given what has been discussed in the previous sections, MLWP—whether at the dynamical core level or for specific physical processes—is generally framed as a **supervised learning** and **regression** problem. However, the ML literature often uses the word *prediction* broadly, which can obscure an important distinction relevant to MLWP. This article therefore distinguishes between:

1. **State-conditioned operator evaluation**
2. **Evolution operator learning**

#### State-Conditioned Operator Evaluation
The first formulation is the simpler of the two—a standard regression mapping:

$$Y_t = g_{\theta}(X_t).$$

At its core, this is a **functional mapping from inputs to outputs at a single time level** — a direct, memoryless transformation with no temporal dependencies encoded in the model itself. Each data point is treated as an **independent observation** within a feature space, making this fundamentally an interpolation problem. The model learns *what the output should look like given the current state*, rather than *how the system evolves over time*. Crucially, the temporal index $t$ plays no structural role here, i.e., it merely labels the sample. The goal, then, is to approximate the operator $g_\theta$ as faithfully as possible across the input domain, not to propagate a trajectory through time.

#### Evolution Operator Learning
Evolution operator learning concerns estimating the future state of a dynamical system. This can take the form of a transition operator,

$$X_{t+\Delta t} = \mathcal{M}_{\theta}(X_t),$$

or a tendency operator,

$$X_{t+\Delta t} = X_t + \Delta t\,\mathcal{F}_{\theta}(X_t),\qquad \frac{dX}{dt} = \mathcal{F}_{\theta}(X_t)$$

Both formulations can be deployed autoregressively, since the output updates the evolving state. In other words, the model approximates either the transition or the tendency operator of a dynamical system. Unlike state-conditioned operator evaluation which is fully cross-sectional with respect to the time domain, evolution operator learning is fundamentally sequential: predictions are conditioned on the historical trajectory of the target variable, making temporal memory an intrinsic component of the formulation.

This does not imply, however, that the variable evolves solely under its own past values, independent of external forcings or drivers. Rather, this approach is adopted pragmatically, supported by two key arguments:

1. **Complexity of process variability**: the underlying dynamics are often intricate and stochastic in nature, making explicit driver-based modeling difficult to formulate and generalize.
2. **Implicit driver signatures**: since the physical drivers are themselves autocorrelated in time, their influence leaves detectable imprints on the target variable's historical record, allowing past values to serve as indirect proxies for those drivers.

In hydrological time series forecasting, streamflow forecasting in particular, this rationale is well-grounded: the effects of antecedent precipitation, soil moisture, and temperature are embedded in the runoff record, making the variable's own history an informative, if indirect, reflection of its physical forcings.

Time series analysis need not operate in isolation from the physical system. In **multivariate approaches**, the physical drivers, the exogenous variables, are explicitly incorporated into the framework:

$$y_t = \beta_1 {x_1}_t + \beta_2 {x_2}_t + \dots + \beta_n {x_n}_t + \phi\, y_{t-1} + \epsilon_t$$

This allows the model to respond to external physical forcings $(x_{1,t}, x_{2,t}, \dots, x_{n,t})$ while the autoregressive term $\phi\, y_{t-1}$ accounts for temporal memory and delayed system response. In machine learning, this formulation can be generalized through a parametric state-update equation:

$$h_t = f_\delta(X_t,\, h_{t-1})$$

<!-- TODO: which is the key role here, as it can include short-term pattern and long-term trends in inside -->
where $X_t$ represents all driver variables at the current time step, and $h_t$ encodes whatever information is inherited from previous time steps, typically in a latent space (hidden state) that can accumulate context over a sequence length extending well beyond a single lag. The output $y_t$ is then computed from the updated hidden state.
 
### Data Characteristics
In data-driven modeling, the data ultimately constrain the choice of model. Understanding data characteristics, therefore, is not a preliminary formality; it is the foundation of the entire modeling process. For MLWP, the defining characteristic is spatiotemporal dependency: the data carry structure in both the time and space domains simultaneously. Recognizing this dependency allows us to translate a physical problem into a well-posed machine learning task.

With both the problem type and the data structure in hand, one might expect that configuring the experimental setup would be an easy task. However, it is not. The reason involves a more subtle question: who defines the task? The answer shapes not just the model architecture, but the entire framing of the problem. If the task is defined through the lens of physical science, the data are no longer mere tensors; they are atmospheric states, diagnostic variables, tendencies, closures, and physical constraints, each carrying domain-specific meaning. If, instead, the task is framed in a free-form, data-driven context, those same objects are reduced to multichannel spatiotemporal arrays—effectively, a video sequence. These two descriptions are not equivalent in practice: they activate different modeling instincts, favor different architectures, and carry different assumptions about what the model is expected to learn.
<!-- For MLWP, the most important first question is: are we learning an evolution operator for the state, or are we learning an operator evaluated from the state?</span> -->

Under the scientific formulation, current MLWP often learns a one-step transition operator and rolls it forward:

$$X_{t+\Delta t} = \mathcal{M}_{\theta}(X_t), \qquad X_{t+n\Delta t} = \mathcal{M}_{\theta}^{(n)}(X_t).$$

As it disscused before, this mathematical formalization mirrors NWP time-stepping. 

Under a free-form spatiotemporal formulation, however, one may instead define the problem as sequence-to-sequence prediction:
<!-- TODO change the formula to recurrence formula -->

$$\{X_{t-k}, \ldots, X_t\} \mapsto \{X_{t+\Delta t}, \ldots, X_{t+m\Delta t}\}.$$

This formalization may still generate future frames autoregressively, but it treats the problem as latent spatiotemporal pattern evolution rather than as an explicit learned numerical time-stepper. 

This is the sense in which the genealogy of these models matters. The dominant rollout strategy in MLWP, specifically at the level of the dynamical core rather than parameterization schemes, is not simply selected from generic data-driven principles; it is inherited from the physical construction of NWP models. Revisiting the foundational papers in this field, one finds they were written by atmospheric scientists deeply shaped by the NWP tradition, which led them to deploy machine learning primarily as a surrogate for the numerical integration operator over the spatial domain.

In this light, MLWP is best understood as an embedded surrogate: a computationally efficient approximate model trained to mimic the behavior of a more expensive and physically explicit system, namely NWP. What this surrogate is tasked with predicting are the **prognostic variables** — wind components, temperature, moisture, and related quantities — whose evolution is governed by PDEs with explicit time derivatives. Their dynamics therefore constitute an **Initial Value Problem (IVP)**, and their estimation is, at its core, a **time series forecasting** task, most naturally framed under evolution operator learning.

The appropriate machine learning paradigm for diagnostic variables and the targets of physical process parameterizations, however, does not follow the same logic. Unlike prognostics, these quantities do not evolve independently through time, i.e., they are derived from, or functionally related to, the current prognostic state. Their suitable ML formulation therefore depends on how that relationship is parameterized: whether the target is expressed as an instantaneous function of the current state, or whether it carries its own temporal structure. The distinction matters in practice, and the following examples illustrate how different parameterization choices lead to different modeling approaches.

## Diagnostics and Parameterization of Physical Processes
Not all variables in a physical model evolve through time in the same way. Some variables — the **prognostic variables** — have explicit time derivatives governing their evolution. They are marched forward in time step by step, and their estimation is naturally a time series forecasting problem framed as evolution operator learning, as discussed above. Other variables, however, do not have their own evolution equations. Instead, they are determined entirely by the current state of the prognostic variables at each time step. These are the **diagnostic variables**, and in the simplest case their relationship to the prognostic state takes the form of a direct functional mapping:

$$D_t = g(\psi_t)$$

where $\psi_t$ is the current prognostic state. Because there is no independent evolution law governing $D_t$, its estimation is better understood as **state-conditioned operator evaluation** rather than evolution operator learning. An important clarification here: diagnostic variables do exhibit temporal correlation in observed data, but that correlation is inherited from the prognostic state driving them, not from any autonomous dynamics of their own.

Whether a physical process is treated as prognostic or diagnostic often depends on how it is **parameterized** — that is, how its relationship to the resolved model variables is expressed mathematically. This parameterization choice directly determines which ML formulation is appropriate.

### Friction in the Shallow Water Equations
To make this concrete, consider the one-dimensional **shallow water equations (SWE)**, which describe conservation of mass and momentum for a depth-averaged free-surface flow:

$$\frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} = 0$$

$$\frac{\partial (hu)}{\partial t} + \frac{\partial}{\partial x}\!\left(hu^2 + \tfrac{1}{2}gh^2\right) = -gh\,\frac{\partial z_b}{\partial x} - S_f$$

where $h$ is water depth $[\text{m}]$, $u$ is depth-averaged velocity $[\text{m/s}]$, $g$ is gravitational acceleration $[\text{m/s}^2]$, $z_b$ is bed elevation $[\text{m}]$, and $S_f$ is the friction slope — a source term encoding energy losses due to bed resistance.

The **prognostic variables** here are $h$ and $u$: they carry explicit time derivatives and are stepped forward in time, making their estimation an IVP and a natural evolution operator learning task.

**Friction**, by contrast, represents a physical process that must be parameterized rather than directly resolved. The frictional force arises from shear stress between the flowing water and the channel bed and sides, and it is expressed through the friction slope $S_f$ as:

$$S_f = \frac{C\,V\,|V|^{m-1}}{R^p}$$

where $C$ and $p$ are coefficients determined by the chosen friction law, $V$ is the flow velocity magnitude, $R$ is the hydraulic radius, and $m$ depends on the flow regime.

Critically, $S_f$ has no time derivative of its own. It is computed directly from the current flow state $(h, u)$ at each time step. This makes it a **diagnostic quantity**: its value is fully determined by the present prognostic state, with no memory of its own past values. As a result, learning $S_f$ is a state-conditioned operator evaluation where the model learns to map the current flow conditions to the corresponding frictional response. It is not, and should not be treated as, a forecasting problem.

### Atmospheric Tracer Transport
A second, richer example comes from atmospheric tracer transport. What makes it particularly instructive is that it can be viewed at two different levels of abstraction — each revealing a different modeling challenge. At the finer level, the tracer transport equation itself contains a physical process that must be parameterized: **vertical convection**. At the coarser level, the entire tracer transport model can be viewed as a parameterized physical process coupled within a larger climate or NWP model. Both perspectives are worth examining.

#### The Governing Equation
When numerically integrating the continuity equation for a tracer $\mu$ over a discrete grid, the operator is typically split into horizontal advection and vertical convection. Horizontal subgrid-scale closure is generally neglected, but vertical convection must be explicitly parameterized, yielding:

$$\frac{d\mu}{dt} + u\frac{\partial\mu}{\partial x} + v\frac{\partial\mu}{\partial y} + w\frac{\partial\mu}{\partial z} = \Sigma$$

Here, the tracer concentration $\mu$ is the **prognostic variable** — it carries an explicit time derivative and is stepped forward in time, making its evolution an IVP governed by evolution operator learning.

The vertical velocity $(w)$, however, is a **diagnostic variable**. It has no evolution equation of its own and is instead determined from the current atmospheric state:

$$w = f(\omega, T, q, z)$$

where $\omega$ is the updraft velocity, $T$ is temperature, $q$ is specific humidity, and $z$ is geopotential height. Its value is fully determined by the present prognostic state at each time step, making it a natural target for state-conditioned operator evaluation, a regression problem with no temporal memory of its own.

If instead of parameterizing only the vertical component one wishes to surrogate the entire right-hand side with a neural network, the problem can be recast in terms of the overall instantaneous tendency of the tracer:

$$\frac{d\mu}{dt} = f_{\theta}(\mu,\, u,\, v,\, \omega,\, T,\, q,\, z,\, \dots)$$

Once the network produces the approximate tendency, the prognostic tracer is marched forward using a numerical integration scheme — for example, a simple Euler step:

$$\mu_{t+1} = \mu_t + \frac{d\mu}{dt} \cdot \Delta t$$

so if we see these two euqaiton at once, the evolution operator learning can be applied to not only the approximate tendency, but handles the prognostic evolution, progressively. in this formalizaiton, autoregressive rollout can natually expalin this formalizaiton. 

#### The Broader View
Zooming out, this same tendency network can itself be seen as a parameterization embedded inside a larger NWP or climate model. From that perspective, the entire tracer transport module is one physical process among many, coupled to the broader model through the shared prognostic state. The inputs to this parameterization are not just tracer-specific quantities but common atmospheric state variables — temperature, humidity, and winds — that are resolved by the host NWP model.

This broader view naturally motivates a **multivariate evolution operator learning** approach. The NWP-resolved prognostics, together with prescribed boundary fluxes, serve as the external drivers fed into the model. All internal processes (e.g., advection, convection, and subgrid-scale mixing) are then absorbed implicitly into the learned hidden state dynamics, without needing to be explicitly represented. In this formulation, the output layer acts directly as the transition operator, mapping the current hidden state to the tracer concentration field at the next time step.

This is an important shift in perspective. At the finer scale, the tracer tendency is approximated by a neural network and rolled out autoregressively, making the tracer a prognostic quantity governed by evolution operator learning. At the coarser scale, however, when the entire transport module is embedded within a host NWP model, the tracer field becomes a parameterization of a physical process — and by our earlier convention, this makes it a diagnostic quantity whose evolution is fully driven by the externally resolved atmospheric state ($u$, $v$, $T$, and $q$). In this view, the appropriate surrogate is no longer an autoregressive tendency model but a learned transition operator, most naturally realized through a recurrent neural network, that maps the sequence of atmospheric states directly to the tracer concentration field at the next time step.

## Conclusion
In this work, we distinguish between two fundamentally different uses of data-driven models in scientific systems. Both aim to replace an expensive function with a cheaper approximation, but they differ in key ways. The first approach involves building a computationally efficient approximate model that mimics a more expensive or complex system, such as a CFD simulation, climate model, or physical experiment. This is like physics-aware substitution, where the goal is to replace a specific component of a known governing model while respecting its overall structure, constraints, and meaning. The ultimate aim is efficiency without breaking the model. In contrast, the second approach treats the original system as an unknown function. Here, we do not focus on internal mechanisms, governing equations, or physical interpretation. This is like context-agnostic learning, with no commitment to physical meaning. The goal is to learn patterns, not the system itself. We can call the first approach embedded surrogate modeling, where machine learning is integrated inside a governing system. The second is called free-form data-driven modeling, with no physical constraints or system knowledge.

Choosing the best machine learning model requires a systematic approach that balances the problem type (first approach) with data characteristics (second approach). Both approaches can be useful, but they should not be evaluated by the same conceptual standard. Embedded surrogates are judged by whether they preserve the meaning and stability of the larger system. Free-form models are judged by whether they forecast well from the data representation they are given. MLWP often sits between these extremes, which is why clarity about the target's physical role matters.

<!-- Part of this discussion may sound like an attempt to build a taxonomy or introduce new terminology. However, we want to emphasize that this is not our goal. Establishing a formal taxonomy requires broad agreement within the community, and any such effort should align with existing conventions. We intentionally avoid that kind of pedantry and its associated pitfalls. Instead, our aim is simply to clarify the perspective we are taking. From this standpoint, some components appear conceptually distorted, and we will briefly explain how we interpret them so that we are on the same page before introducing the core problem. -->

The issue arises when we refer to a *data-driven method*. Ideally, this would mean a method dictated by the intrinsic characteristics of the data itself. However, this is not quite what happens in machine learning–based weather prediction. If we present the data to domain experts and ask—strictly in a context-free, domain-agnostic manner—what class of models would be appropriate, very few would suggest an auto-regressive rollout approach. Instead, they would typically propose spatiotemporal modeling strategies.

Modern MLWP inherited the autoregressive rollout not solely from machine learning sequence models, but also from the NWP view of weather as an initial-value problem. Reanalysis datasets provide gridded atmospheric states at successive times; numerical models march states forward; ML models are then trained to imitate the observed or analyzed transition from $X_t$ to $X_{t+\Delta t}$. This genealogy explains why state-to-state autoregression became the dominant template. In this setting, a simple autoregressive rollout is not an arbitrary ML choice. It is the learned analogue of a numerical weather model's time step.

But if we remove the physical semantics and describe the same object only as a multichannel gridded tensor sequence, the modeling instinct changes. A free-form ML formulation would emphasize spatiotemporal representation learning: ConvLSTM, encoder-decoder recurrent networks, temporal convolutional networks, video-prediction models, or transformer-based sequence models. These models can still be autoregressive, but its recurrence is organized around latent spatiotemporal memory and frame evolution, not around an explicit approximation of an NWP time-step operator. Therefore, saying "both are autoregressive" hides the important difference: one autoregression is tied to numerical integration, while the other is tied to generic sequence modeling.


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