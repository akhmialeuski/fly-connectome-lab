# fly-connectome-lab

`fly-connectome-lab` is an early-stage open-source research project for reproducible computational experiments built on Drosophila connectomes, starting with MaleCNS v1.0.

## Status

This repository is currently a documentation foundation for planned experiments. It does not yet contain experiment code, datasets, models, or generated results.

## What is a connectome?

A connectome is a structural wiring diagram of a nervous system: which neurons connect to which other neurons, and how those connections are organized. In this project, a connectome is used as a biological structural prior for reproducible machine-learning experiments.

MaleCNS is a structural connectome release. It is not a complete biological simulation of a living fly brain, and work in this repository should not describe it as one.

## Planned experiments

### 1. Sequential Visual Memory

This experiment line is planned around a frozen MaleCNS recurrent reservoir. The initial goal is to present faces or medical images as sequences of partial observations, using moving-window glimpses for 2D images and sequential slices for 3D medical data. Early comparisons will focus on persistent MaleCNS state versus reset-between-observations controls, initially using CelebA, LFW, OrganMNIST3D, and NoduleMNIST3D.

### 2. Brain Lesion and Functional Recovery

This experiment line is planned as an extension of Sequential Visual Memory. It will introduce controlled neuron and synapse lesions, measure functional degradation, and evaluate recovery through readout retraining, homeostatic compensation, and constrained plasticity of surviving connections. Recovery experiments must never silently create connections that do not exist in the biological connectome.

## Planned interfaces

The project is expected to grow toward:

- a CLI for running experiments reproducibly; and
- a local web interface for result visualization and interactive inference.

## Project scope

The goal of `fly-connectome-lab` is to support reproducible experiments with Drosophila connectomes while keeping biological-connectivity assumptions explicit and auditable.

## Data and licenses

This repository is licensed under Apache License 2.0. Third-party datasets, downloaded MaleCNS data, and external software dependencies are not covered by this repository's Apache-2.0 license unless their own upstream licenses explicitly say so.

See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) for dataset, software, attribution, citation, and redistribution notes.

## References and citations

See [REFERENCES.md](REFERENCES.md) for curated references, upstream publications, and citation guidance.

## Reproducibility

To keep experiments reproducible and the repository lightweight, datasets, generated traces, model checkpoints, caches, uploaded images, and experimental artifacts should be stored outside Git or in explicitly ignored paths. This repository is intended to track documentation, code, and configuration needed to reproduce experiments, not the resulting bulk data.

## Upstream projects and tools

- MaleCNS official project: <https://male-cns.janelia.org/>
- MaleCNS download page: <https://male-cns.janelia.org/download/>
- MaleCNS analysis and supplemental repository: <https://github.com/flyconnectome/2025malecns>
- fly.ai / flybrain: <https://github.com/alextitonis/fly.ai>
- ConnecTorch: <https://github.com/us/connectorch>
- MedMNIST: <https://medmnist.com/> and <https://github.com/MedMNIST/MedMNIST>

## Maintainer

Maintainer: <https://github.com/akhmialeuski>
