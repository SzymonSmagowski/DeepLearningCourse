•Use the given dataset to generate images of cats.
•It is up to you which approach to apply.
•Potential methods include, but are not limited to, generative adversarial networks,
diffusion models, and variational autoencoders, as well as their extensions (e.g.,
VQ-VAE, VQ-VAE-2).
•Investigate the influence of hyperparameters on obtained results.
•Compare the results of the methods quantitatively, e.g., using the Fr´echet Inception
Distance (FID). Assess them qualitatively as well.
•Address the mode collapse problem if it occurs.
•Select two of your generated images together with their latent noise matrix;
interpolate linearly between the two latent matrices to generate 8 additional latent
matrices; use these 8 matrices to generate images from your model; present the 10
generated images (8 newly generated and 2 generated previously) and discuss the
importance of the results.


•Discuss any additional findings.
•Remark 1: The images have different resolutions. You should decide how to address
this issue.
•Remark 2: Training generative models may be time- and resource-consuming. An
essential part of this task is determining how to approach this issue; please refer
to ”Working with limited computing resources” slide.
•Additional task: Train a model on a dataset containing cats and dogs
(https://www.kaggle.com/competitions/dogs-vs-cats/). Compare the
results with those obtained from the dataset containing only cats. Do the generated
observations resemble cats and dogs, or rather a combination of both?