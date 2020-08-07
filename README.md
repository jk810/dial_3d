# Distributed Ad-hoc Localization
## Rapid and distributed position estimation in large proliferated constellations

DiAL is for rapidly determining satellite position in a large, ad-hoc, proliferated network. Distributed localization is performed using crosslink and range information from each node's local neighborhood.

DiAL leverages algorithms and methods developed for terrestrial localization in wireless networks. Results from Project Packet are used to initialize the crosslinks and network topology. Multi-Dimensional Scaling (MDS) is used to map the networks with an algorithm called MDS-MAP. A distributed form of MDS-MAP will be implemented, called MDS-MAP(P), which enables parallel computation and mapping.

Local neighborhoods are independently mapped and then merged. As the local maps are aggregated into a total map, an absolute map can be constructed with the inclusion of 3 (in 2 dimensions) or 4 (in 3 dimensions) anchors nodes that are position-aware.
