# Architecture Enforced Disentanglement

Investigate three architectures to enforce disentanglement on Darcy flow problem. The idea is to alter the output with each latent dimension seperately rather than predicting the output using all latent factors at once. Hopefully this will lead to factors with different influences on the output prediction, producing disentangled representations.

- Architecture changes may vary with each commit. In each commit, models added will correspond to the model architecture in that particular commit
- General architectures are illustrated, but specifics in architecture may vary with each commit. 
- Trained models are too large to push and stored locally, but post-processing figures are illustrated.

- Model figures are stored according to the following directory structure:
`problem/generative_parameter_dimension/generative_parameter_distribution/general_network_architecture/latent_dim/VAE_<trial_num>`

# Architecture 1 (arch1)
![arch1](/architecture_diagrams/arch_1.PNG)

# Architecture 2 (arch2)
![arch2](/architecture_diagrams/arch_2.PNG)

# Architecture 3 (arch3)
![arch3](/architecture_diagrams/arch_3.PNG)
