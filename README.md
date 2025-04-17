# scene-matching

## Possible Solutions

### NetVLAD
NetVLAD is a deep learning layer designed for image retrieval tasks like visual place recognition. It works by taking local features from a CNN (like parts of an image) and softly assigning them to a set of learned "cluster centers." For each center, it calculates how much each local feature differs (residuals) and then sums these differences. The result is a compact vector representing the whole image, capturing both what is in the image and where, making it ideal for comparing scenes even with viewpoint changes. <br />
Paper: https://arxiv.org/pdf/1511.07247

### SimCLR
SimCLR is a self-supervised learning method that teaches a network to understand images without labels. It does this by taking two different augmented views of the same image and pushing their embeddings closer together, while pulling embeddings of other images apart. This is done using a contrastive loss. It learns general-purpose features useful for many downstream tasks. The key idea is: if two images come from the same source, they should look similar in the embedding space. <br />
Paper: https://arxiv.org/pdf/2002.05709

### Bag-of-Visual-Words Prediction
This method builds on the idea of representing an image as a histogram of visual words (like keywords in a document). First, a CNN is trained to extract features from images, and then these features are clustered using k-means to form a vocabulary of visual words. Another CNN is trained to predict the visual word histogram of an original image when given a distorted version of it. This forces the network to learn robust and context-aware features, making it a powerful self-supervised alternative to pixel-level reconstruction tasks. <br />
Paper: https://arxiv.org/pdf/2002.12247

### SuperGlue
SuperGlue is a deep learning model designed to match keypoints between pairs of images with high accuracy, making it ideal for tasks like scene matching and visual localization. It builds on traditional keypoint detectors (like SuperPoint) by using a graph neural network to reason about spatial relationships and context between keypoints in both images. SuperGlue treats matching as a graph matching problem, leveraging attention mechanisms—similar to those in transformers—to find robust correspondences even under challenging conditions like changes in viewpoint, lighting, or partial occlusions. The result is a powerful, end-to-end trainable system that significantly improves matching quality compared to traditional approaches. <br />
Paper: https://arxiv.org/pdf/1911.11763

## Dataset

https://zenodo.org/records/1243106
