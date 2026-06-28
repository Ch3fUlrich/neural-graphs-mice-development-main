# Method Decision Reasoning

This document summarizes how the notebooks in `notebooks/` use the repository’s methods to quantify developmental changes in mouse neural activity. My reading is that the repository turns longitudinal calcium-imaging sessions into functional activity graphs, then asks whether graph structure changes across postnatal development.

## Biological Setup

The dataset is organized by animal (`DON-*`) and recording session (`YYYYMMDD`). Each session has a developmental age, `dev_stage`, computed as postnatal day (`PDay`) from the date of birth and the recording date. The notebooks compare sessions across developmental windows returned by `ngmd.utils.get_default_dev_stage_cuts()`:

-   `PDay 15-24`: young/juvenile window.
-   `PDay 24-55`: early-to-late adolescent transition.
-   `PDay 55-90`: late adolescence/young adulthood.
-   `PDay 90-300`: adult.

Those windows are biologically plausible but should be interpreted as analysis bins, not hard species-wide boundaries. External references commonly place mouse adolescence around the post-weaning period through roughly P50-P60, with PFC maturation continuing through adolescence and early adulthood. See, for example, the review-level discussion of mouse adolescence in the PFC literature and sleep-development literature: [https://pmc.ncbi.nlm.nih.gov/articles/PMC4596535/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4596535/) and [https://www.mdpi.com/2076-3425/3/1/318](https://www.mdpi.com/2076-3425/3/1/318).

The measured object is not anatomical synaptic connectivity. It is functional co-activity among imaged ROIs/cells derived from binarized calcium-event traces and precomputed similarity matrices. A graph node is an ROI/neuron; an edge means that a pair of ROIs has similarity above a chosen threshold.

## Data And Preprocessing Assumptions

The repository assumes a calcium-imaging preprocessing workflow similar to Suite2p-style outputs: ROIs/cells are detected, fluorescence traces are extracted, and activity-related traces can be converted into event/spike-like time series. Suite2p itself is described as a two-photon calcium-imaging pipeline for registration, cell detection, signal extraction, spike deconvolution, neuropil subtraction, and curation: [https://github.com/cortex-lab/suite2p](https://github.com/cortex-lab/suite2p).

In this repository, `ngmd.dataset.Session` loads:

-   `F_upphase.npy`: binarized calcium-event traces.
-   `cell_drying.npy`: mask used to remove gel-drying affected cells.
-   `allcell_clean_corr_pval_zscore.npy`: precomputed correlation, p-value, and z-score matrices.
-   `sttc_arrays.npz`: precomputed STTC matrix and surrogate-percentile matrix.

`Session.load_binarized_traces(clean=True)` removes gel-drying cells and silent ROIs. `AnimalDataset.load_traces_from_good_rois(session)` additionally removes ROIs listed in `bad_rois.csv`. This matters because most graph metrics are sensitive to silent cells, failed ROIs, and recording artifacts.

2. bad rois and sessions are located in `{dataset_root}/bad_rois.csv` or `{dataset_root}/bad_sessions.csv`

2. Bad ROIs are identified using Tukey's fences method on neuronal metrics. The detection happens in [01a_binarized_traces_statistics.ipynb](vscode-file://vscode-app/c:/Users/mauls/AppData/Local/Programs/Microsoft%20VS%20Code/f6cfa2ea24/resources/app/out/vs/code/electron-browser/workbench/workbench.html).
   1. Calculates quartiles (Q1, Q3) and interquartile range (IQR) for a chosen metric
   2. Inner fence: Q3 + 1.5 × IQR
   3. Outer fence: Q3 + 3 × IQR
   4. ROIs with values exceeding these fences are flagged as outliers


## Core Graph Construction Logic

Graph construction is centralized in `ngmd.graphs.SimilarityGraph`.

1.  Load a square pairwise similarity matrix.
2.  Convert NaN to zero, infinities to finite values, and similarities to absolute values.
3.  Threshold the matrix with `connect_thresh`.
4.  Remove self-connections by zeroing the diagonal.
5.  Build a NetworkX graph from the weighted adjacency matrix.

The main similarity choices are:

-   `corrs`: pairwise correlations of binned calcium-event counts.
-   `zscores`: correlation z-scores relative to circular-shift surrogates.
-   `sttc`: Spike Time Tiling Coefficient between binary event trains.
-   `sttc_percentiles`: percentile rank of observed absolute STTC values against surrogate STTC values.

The notebooks increasingly favor `sttc_percentiles`, especially around threshold `95`, because raw STTC thresholds are difficult to choose consistently across sessions with different activity levels.

## Why STTC Is Central

STTC is used because calcium-event traces are treated as spike/event trains. The STTC literature argues that simple spike-train correlation indices can be confounded by firing rate and periods of silence, whereas STTC is designed to compare spike timing within a synchronicity window. The original Cutts & Eglen paper frames STTC as a replacement for correlation indices in spike-time analyses and explicitly discusses developmental neural activity applications: [https://www.repository.cam.ac.uk/items/a21a8743-f2e0-432f-8d3c-9fdbfbfb1a84](https://www.repository.cam.ac.uk/items/a21a8743-f2e0-432f-8d3c-9fdbfbfb1a84).

In this repository, `ngmd.sttc.matrix_spike_time_tiling_coefficient()`:

-   tiles each binary event train by a `dt` synchronicity window;
-   computes the fraction of each train’s events matched by another train;
-   computes the fraction of recording time covered by each tiled train;
-   combines those terms into the symmetric STTC matrix.

The configured default is `dt: 0.25` seconds in `configs/config.yaml`. That is much wider than electrophysiology-scale spike synchrony windows, but it can be reasonable for calcium-event timing because calcium signals are slower and the traces here are already event-like proxies rather than intracellular spikes.

## Surrogates And Threshold Choice

The key thresholding decision is made through surrogate STTC distributions rather than a fixed raw STTC value. `scripts/data_generation/precompute_sttc_arrays.py` computes:

-   the observed STTC matrix;
-   surrogate STTC values from randomized traces;
-   a `percentiles_mat` where each observed pair receives its percentile rank among observed plus surrogate values.

The default surrogate method is `pulse_shuffle`. `ngmd.timeseries.make_pulse_shuffle_surrogates()` preserves the global timing of pulse windows but randomly permutes ROI assignments within those windows. Biologically, this asks whether a given ROI pair is more synchronized than expected if the population-level event timing remains but neuron identities are shuffled. This is a good null for rejecting connections driven only by global co-activation epochs.

Other surrogate methods explored in `01b_choosing_sttc_thresholds.ipynb` are:

-   `time_shuffle`: circularly shifts each ROI trace, preserving individual event structure but destroying alignment.
-   `pulse_shuffle`: preserves pulse windows while shuffling ROI identities.
-   `time_and_pulse_shuffle`: combines both.
-   `poisson`: generates synthetic event trains from each ROI’s pulse rate and widths.

The practical conclusion in the notebooks is that percentile thresholding is more defensible than a universal raw STTC threshold. A threshold like `sttc_percentiles > 95` means “keep ROI pairs whose synchrony is in the extreme tail of a session-specific surrogate null.” Higher thresholds such as 96-99 are also run in the summary notebook to inspect robustness.

## Notebook-By-Notebook Understanding

### `00_loading_data_and_creating_graphs.ipynb`

This is the onboarding notebook. It shows how to instantiate `AnimalDataset`, select animals and sessions, load binarized traces, and create graphs from correlations, z-scores, STTC values, or STTC percentiles. Its biological role is definitional: it establishes that “neural activity network” means a functional graph derived from pairwise activity similarity among imaged ROIs.

It also demonstrates the interpretive difference between raw correlation, z-score, raw STTC, and percentile-thresholded STTC graphs. The notebook makes clear that z-score and percentile graphs can contain connections with modest raw similarity if those values are unusually large relative to a null distribution.

### `01a_binarized_traces_statistics.ipynb`

This notebook quantifies the activity-event traces before graph construction. It loads precomputed outputs from `scripts/data_generation/binarized_traces_statistics.py`:

-   pulse rate per ROI;
-   active fraction per ROI;
-   pulse width;
-   pulse interval.

It plots these quantities against developmental stage and identifies outlier ROIs using Tukey-style fences. The biological point is quality control plus basic phenotyping: before interpreting graph changes, one needs to know whether event rates, durations, or silent/overactive cells are changing with age or being contaminated by bad ROIs.

### `01b_sim_mat_statistics.ipynb`

This notebook summarizes pairwise similarity values across sessions. It loads `corrs_statistics.csv` and `sttc_statistics.csv`, each with one row per session and histograms/statistics of off-diagonal matrix values. It groups sessions by developmental windows and overlays value distributions.

The biological question is whether pairwise functional co-activity itself shifts over development before any graph thresholding is applied. This is important because downstream graph metrics can change either because of true topological organization or because the entire similarity distribution shifts.

### `01b_choosing_sttc_thresholds.ipynb`

This is the methodological decision notebook for STTC graph edges. It compares low-activity and high-activity sessions, examines raw STTC distributions, and then explores surrogate distributions from multiple null models.

Its main contribution is to justify edge definition by percentile against surrogates. The notebook shows why a raw threshold such as `STTC > 0.2` is arbitrary across sessions, while `sttc_percentiles > 95` adapts to activity level and session-specific null expectations.

### `02_degree_distributions.ipynb`

This notebook treats graph degree as a neuron-level summary. For `SIMILARITY_METHOD = "sttc_percentiles"` and a chosen `CONNECT_THRESH`, it loads precomputed degree lists from `scripts/data_generation/degree_distributions.py`.

A node’s degree is the sum/count of its graph connections as represented by NetworkX. The notebook then normalizes degrees by the number of neurons in the session. Biologically, this asks whether individual neurons become more or less broadly functionally coupled to the recorded population across development.

It also compares empirical degree distributions to classical random graph families such as Erdos-Renyi, grid/geometric graphs, and Barabasi-Albert-like models. Those comparisons are interpretive reference points, not model fits.

### `03_network_summaries.ipynb`

This notebook analyzes scalar graph metrics precomputed by `scripts/data_generation/network_summaries.py` and, optionally, `scripts/data_generation/small_worldness.py`.

The main metrics are:

-   number of neurons;
-   number of connected components;
-   edge density;
-   transitivity;
-   average clustering coefficient;
-   size of largest connected component;
-   average shortest path length on the largest connected component;
-   radius;
-   diameter;
-   median PageRank;
-   small-worldness omega.

The script can subsample graphs, with defaults in `configs/config.yaml` set to 20 trials and `subsample_size: 250`. This controls for different numbers of detected neurons across sessions. The notebook compares distributions by developmental window and uses permutation tests with multiple-comparison correction to ask whether windows differ statistically.

Biologically, these metrics translate activity co-synchrony into population-level organization: fragmentation, integration, local clustering, path efficiency, and hub-like centrality.

Small-world omega follows the Telesford et al. formulation comparing the observed graph with lattice and random references. Telesford et al. introduced omega for small-world quantification in brain networks: [https://pmc.ncbi.nlm.nih.gov/articles/PMC3604768/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3604768/). The repository comments also note Neal’s caution about small-world indices and the choice of omega over sigma; Neal compares omega, sigma/Q, and SWI as small-worldness measures: [https://www.cambridge.org/core/product/identifier/S2050124217000054/type/journal\_article](https://www.cambridge.org/core/product/identifier/S2050124217000054/type/journal_article).

### `04_laplacian_spectra.ipynb`

This notebook uses normalized graph Laplacian eigenvalues as a global fingerprint of graph shape. `scripts/data_generation/laplacian_spectra.py` stores the sorted eigenvalues for each session graph.

The biological intuition is that spectra summarize graph organization beyond one scalar metric. For example:

-   near-zero eigenvalues reflect connected components;
-   the overall spectrum shape reflects connectivity, bottlenecks, and global organization;
-   spectral distributions can be compared across sessions or developmental windows.

The notebook uses several complementary spectral approaches:

-   visual comparison of spectra by developmental window;
-   comparison with random graph reference families;
-   NetLSD heat-kernel descriptors;
-   1-Wasserstein distances between spectral probability masses;
-   Wasserstein barycenters for developmental windows;
-   assigning a session to the closest developmental barycenter as an exploratory “network age” idea.

This is the most shape-oriented notebook: it asks whether a whole graph looks developmentally young or adult based on spectral geometry, rather than any single local network statistic.

### `05_link_predictability.ipynb`

This notebook applies structural consistency from Lü et al. to ask how predictable network links are from the remaining graph structure. The original structural-consistency idea treats predictability as an inherent property of the network and uses matrix perturbation to remove links, reconstruct likely missing links, and score recovery. A later review of link-predictability methods summarizes Lü et al.'s contribution as using matrix perturbation and proposing structural consistency as a network predictability property: [https://www.sciencedirect.com/science/article/abs/pii/S0957417422012271](https://www.sciencedirect.com/science/article/abs/pii/S0957417422012271).

In `ngmd.structural_consistency.structural_consistency()`:

1.  randomly remove a fraction of observed edges;
2.  eigendecompose the remaining adjacency matrix;
3.  approximate the perturbation effect on the adjacency structure;
4.  rank unobserved links;
5.  compute the fraction of removed links recovered among the top-ranked predictions.

The notebook compares Erdos-Renyi and Watts-Strogatz examples, then loads repository-wide structural-consistency tables generated by `scripts/data_generation/structural_consistency.py`. Biologically, high structural consistency suggests that functional links are embedded in a regular, redundant, or organized topology; low structural consistency suggests more random or idiosyncratic pairwise synchrony.

### `06_all_analysis_summary.ipynb`

This notebook is the orchestration and reporting layer. It can run missing analysis scripts, load generated CSV tables, add developmental-window labels, and create summary plots for:

-   binarized trace summaries;
-   similarity matrix summaries;
-   degree distributions;
-   scalar network metrics;
-   structural consistency;
-   Laplacian spectra.

It also explicitly runs several graph specifications, including `("zscores", 4)` and multiple `("sttc_percentiles", threshold)` settings from 95 to 99. This makes it the best high-level entry point for checking whether conclusions depend on the edge-definition threshold.

## Interpretation Of The Method Stack

The pipeline quantifies biology at three levels.

First, it quantifies cellular event activity directly: pulse rate, active fraction, pulse width, and pulse interval. This level answers whether the raw event process changes with development and supports ROI/session quality control.

Second, it quantifies pairwise functional co-activity: correlations, z-scores, STTC, and surrogate percentiles. This level asks whether cell pairs become more or less synchronized beyond chance expectations.

Third, it quantifies population organization by turning pairwise co-activity into graphs. Degree distributions describe neuron-level participation; scalar network summaries describe integration and clustering; Laplacian spectra describe whole-network shape; structural consistency describes how predictable or organized the edge set is.

## Method Decisions I Would Preserve

-   Use `sttc_percentiles` as the primary graph-building matrix for event synchrony, because it anchors edge calls to session-specific surrogate nulls.
-   Keep threshold sweeps from 95 to 99 percentiles when making biological claims, because graph metrics can be threshold-sensitive.
-   Keep subsampling for scalar network summaries and structural consistency, because sessions vary substantially in number of detected neurons.
-   Continue filtering bad sessions and bad ROIs before interpreting developmental changes.
-   Treat random graph examples as interpretive controls, not mechanistic claims.
-   Report graph results alongside pulse/activity summaries, because changes in network topology can be confounded by changes in event rates or active fractions.

## Main Caveats

-   These are functional activity graphs, not anatomical wiring diagrams.
-   Calcium-event timing is slower and more indirect than electrophysiological spike timing; STTC `dt` should therefore be justified for the imaging/event extraction timescale.
-   Percentile thresholds depend on the surrogate null. `pulse_shuffle` preserves population event windows but changes ROI identity, so it is most sensitive to pair-specific synchrony beyond global events. A different biological null could produce different edge calls.
-   Graph metrics can be dominated by edge density and connected-component structure. Threshold sweeps, density-aware interpretation, and subsampling are essential.
-   Developmental bins are biologically motivated but broad. Claims should distinguish “this graph metric differs between repository-defined PDay windows” from stronger claims about named developmental phases.