# Pixels to Words: Image Captioning Using CNN and LSTM

## Project Overview
This project, **Pixels to Words**, explores the application of deep learning to automate image captioning using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The goal is to bridge the gap between visual data and natural language by generating meaningful and coherent captions for diverse images. The project has practical implications in accessibility, content creation, and surveillance.

---

## Motivation
Automating image captioning is essential for enhancing applications like:
- **Accessibility for visually impaired individuals**
- **Content creation for social media and marketing**
- **Automated surveillance systems**

Advances in deep learning have made it possible to combine CNNs for image understanding and LSTMs for sequence generation to achieve this goal.

---

## Research Questions
1. **Effectiveness**: How effectively can a CNN + LSTM model generate meaningful and accurate captions for diverse images?
2. **Generalization**: Can the model generalize well to unseen images, and how can redundancy in captions be minimized?

---

## Data
- **Image Dataset**: Flickr8k Dataset
  - **Content**: 8,000 images featuring diverse activities, objects, and scenes.
- **Caption Dataset**: Each image is paired with one representative caption.

### Dataset Characteristics
- **Visual Diversity**: Includes actions like playing, running, swimming, and scenes such as beaches, parks, and sports events.
- **Applications**: Supports training models for image-to-text tasks and contextual understanding.

---

## Exploratory Data Analysis
- **Most Frequent Words in Captions**: Common words include "dog," "man," "woman," "white," etc., with structural markers like "startseq" and "endseq."
- **Caption Length Distribution**: Normal distribution centered around 10-12 words. Few captions exceed 20 words.

---

## Approach
### Image and Text Preprocessing
- **Text**: Captions are cleaned, tokenized, and appended with "startseq" and "endseq."
- **Images**: Resized to 224x224 pixels to match the input size of the DenseNet201 architecture.

### Model Architecture
1. **Encoder (CNN)**: Extracts image features using **DenseNet201**.
2. **Decoder (LSTM)**: Generates text sequences by combining image and caption features.
3. **Dense Layers**: Used for feature embedding and prediction.
4. **Dropout Layers**: Applied to reduce overfitting.

### Optimization
- **Loss Function**: Categorical cross-entropy.
- **Optimizer**: Adam with learning rate decay.
- **Regularization**: Early stopping to prevent overfitting.

---

## Results
- The model achieved a **BLEU score of 0.4**, which is considered a moderate score for image captioning tasks.
- The generated captions capture the essence of most images, indicating a good understanding of visual content.
- Some captions were detailed (e.g., "girl in red shirt is holding her head in the air"), showing the model's ability to pick up finer details.

### Key Insights
- **Training Loss**: Steady decline during training, indicating good learning behavior.
- **Validation Loss**: Slight increase suggests possible overfitting, which can be mitigated through regularization techniques.

---

## Contributions and Implications
### Contributions
- Demonstrated the effectiveness of **CNN + LSTM architectures** for image captioning.
- Established a baseline for future improvements using techniques like **attention mechanisms**, **beam search**, and **data augmentation**.

### Implications
- **Accessibility**: Assisting visually impaired individuals by describing visual content.
- **Search Engines**: Enhancing image-based search capabilities.
- **Content Creation**: Automating social media content generation.
- **Healthcare and E-commerce**: Extending image-to-text models to specific domains.

---

## Tools & Technologies Used
- **Python**: Core programming language.
- **Keras & TensorFlow**: For building and training the deep learning model.
- **DenseNet201**: Pre-trained CNN model used as the encoder.
- **Flickr8k Dataset**: For training and testing the image captioning model.
- **Jupyter Notebook**: For code implementation.

---

## Project Structure
```
|-- data
|   |-- Flickr8k_Dataset
|-- notebooks
|   |-- FinalProjectDL.ipynb
|-- presentation
|   |-- Pixels_to_Words_Presentation.pdf
|-- README.md
```

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/hemanth379/Pixels_to_Words_Image_Captioning_Using_CNN_and_LSTM.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Pixels_to_Words_Image_Captioning_Using_CNN_and_LSTM
   ```
3. Run the Jupyter notebook to train the model and generate captions:
   ```bash
   jupyter notebook FinalProjectDL.ipynb
   ```

---

## Future Enhancements
1. **Incorporate Attention Mechanism**: Improve caption accuracy by focusing on relevant parts of images.
2. **Beam Search**: Enhance the quality of generated captions by considering multiple possible outputs.
3. **Data Augmentation**: Increase the dataset's diversity to improve the model's generalization.
4. **Domain-Specific Applications**: Extend the model to healthcare, e-commerce, and other domains.

---

## Contributors
- **Hemanth Varma Pericherla**
- **Vaishnavi Karingala**
- **Venkata Naga Anirudh Chaganti**
- **Rithika Pagadala**
- **Rupesh Maadas**

---

## Acknowledgements
- **Prof. INKYU  KIM** for guidance and support throughout the project. 
