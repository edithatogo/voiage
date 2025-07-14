Introduction to voiage
=======================

**What is Value of Information (VOI) Analysis?**

Value of Information (VOI) analysis is a quantitative decision analysis technique
used to estimate the economic benefit of collecting additional information before
making a decision under uncertainty. It helps answer questions like:

*   "Is it worth investing in more research to reduce uncertainty about this decision?"
*   "Which specific uncertainties, if resolved, would provide the most value?"
*   "What is the maximum amount we should be willing to pay for a particular piece of research?"
*   "Which proposed study design offers the best value for money?"

VOI is particularly prominent in health technology assessment (HTA), where decisions
about adopting new medical treatments or technologies involve significant uncertainty
and potentially large population health and budget impacts. However, its principles
are applicable across many fields including environmental management, engineering,
finance, and public policy.

**Key VOI Metrics**

voiage aims to implement a range of VOI metrics, including:

*   **Expected Value of Perfect Information (EVPI):**
    The expected increase in net benefit if all uncertainty about model parameters
    were eliminated. It represents the maximum value of any further research.

*   **Expected Value of Partial Perfect Information (EVPPI):**
    The expected increase in net benefit if uncertainty about a specific subset
    of model parameters were eliminated. Useful for identifying key drivers of
    decision uncertainty.

*   **Expected Value of Sample Information (EVSI):**
    The expected increase in net benefit from conducting a particular research study
    (e.g., a clinical trial of a specific design and sample size). This is often
    the most practical VOI metric for guiding research decisions.

*   **Expected Net Benefit of Sampling (ENBS):**
    Calculated as EVSI minus the cost of the proposed research. A positive ENBS
    suggests the research is economically worthwhile.

**Why voiage?**

While VOI methods are well-established, and implementations exist (notably in R,
e.g., BCEA, dampack, voi packages), a comprehensive, modern, and extensible
Python library for VOI analysis is still a developing area. voiage aims to:

*   Provide a **user-friendly Python API** for common and advanced VOI calculations.
*   Leverage the **Python scientific computing ecosystem** (NumPy, SciPy, Pandas, xarray, PyMC, JAX)
    for performance and flexibility.
*   Offer implementations for a **wide range of VOI analyses**, including those not
    commonly found in existing packages (e.g., structural VOI, adaptive design EVSI,
    portfolio VOI).
*   Facilitate **integration with modern Bayesian modeling tools** (like PyMC or NumPyro).
*   Support **computationally intensive analyses** through efficient algorithms and
    potential backend abstractions (e.g., JAX for GPU/TPU acceleration).
*   Be **well-documented and tested** to ensure reliability and ease of use for
    researchers, health economists, and decision analysts.

**Target Audience**

voiage is intended for:

*   Health economists and HTA practitioners.
*   Decision analysts and operations researchers.
*   Statisticians involved in clinical trial design and Bayesian analysis.
*   Researchers in any field applying decision theory and uncertainty quantification.
*   Students learning about VOI methods.

This documentation will guide you through installing voiage, understanding its
core concepts, using its API for various analyses, and contributing to its
development.
