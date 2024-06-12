# RIS-Assisted High-Resolution Radar Sensing
This GitHub repository contains the code library used to create the results of the paper `RIS-Assisted High-Resolution Radar Sensing` submitted to IEEE Transactions on Signal Processing, June 2024.

Author: Martin Voigt Vejling

E-Mail: mvv@{math,es}.aau.dk

## Contents
### Modules
- Beamforming.py `Backend implementation of beamforming techniques: Bartlett, Capon, MUSIC.`
- CompressiveSensing.py `Backend implementation of OMP.`
- SensingMetrics.py `Implementation of the OSPA metric.`
- PositionEstimation_v2 `Module to do weighted non-linear least squares.`
- DataAssociation.py `Module to do data association.`
- system.py `System and signal model.`
- MainEstimationCore.py `Skeleton core for the estimation algorithm.`
- ChAnalysis.py `Module to run analysis of the detection probability and Cramér-Rao lower bounds.`
- TheoreticalAnalysis.py `Module supporting the theoretical analysis of coherence and detection probability.`
- MainEstimation.py `Main module for the estimation algorithm. Runs the algorithm.`

### Configuration file
- system_config.toml `File to specify experiment configurations.`

### Simulation scripts
- DetectionSimulationStudy.py `Run simulation study to evaluate the detection probability.`
- FisherSimulationStudy.py `Run simulation study to evaluate the Cramér-Rao lower bound.`
- MainSimulationStudy.py `Run simulation study to sensing algorithm performance.`

### Recreating figures
- Figure 2 & 4: TheoreticalCoerenceStudy.py
- Figure 3: PlotWorkingPrinciple.py
- Figure 5: TheoreticalTildeCStudy.py
- Figure 6 & 7: TheoreticalDetectionStudy.py
- Figure 8: results/FisherStudyPlot.py
- Figure 9a & 9b: results/DetectionMultiStudyPlot.py
- Figure 9c: results/MainStudyPlot.py

## Software Setup

### Dependencies
```
python 3
numpy
scipy
matplotlib
toml
multiprocessing
tqdm
```
