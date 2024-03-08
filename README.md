# Compromising anonymity in identity-reserved k-anonymous datasets through aggregate knowledge

This repository contains the results and the implementation of the mathematical models used in the paper "Compromising anonymity in identity-reserved k-anonymous datasets through aggregate knowledge".

## Abstract

Data processors increasingly rely on external data sources to improve strategic or operational decision taking.  Data owners can facilitate this by releasing datasets directly to data processors or doing so indirectly via data spaces.  As data processors often have different needs and due to the sensitivity of the data, multiple anonymized versions of an original dataset are often released.  However, doing so can introduce severe privacy risks.  

This paper demonstrates the emerging privacy risks when curious -- potentially colluding -- service providers obtain an identity-reserved and aggregated k-anonymous version of the same dataset.  We build a mathematical model of the attack and demonstrate its applicability in the presence of attackers with different goals and computing power.  The model is applied to a real world scenario and countermeasures are presented to mitigate the attack.

## Repository Content

### Examples

The **Examples** folder contains 2 example versions (*A* and *B*) of a non-aggregated dataset and its corresponding aggregated dataset.

### Results

The **Results** folder contains figures for all the results of the experiments performed with the targeted pseudonym attack.

### Models

The **Models** folder contains the python implementation of the models used for the targeted pseudonym attack and the population attack.
