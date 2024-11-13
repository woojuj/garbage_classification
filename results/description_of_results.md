In this folder are the outputs of the model and the figures with the results. 

- `final_output_assignment2.out`: This file provides the log of the model’s training process in TALC, showing the progression of training and validation accuracy across epochs. The log reveals consistent improvement in accuracy, with early stopping applied after the model reached peak performance, saving the best model parameters. The model achieved a final evaluation accuracy of 72%, indicating effective multi-modal classification, though there is room for further optimization.

- `confusion_matrix.png`: This confusion matrix visualizes the model’s classification performance across the four categories (Black, Blue, Green, TTR). The matrix highlights areas of strong performance, particularly for the "Blue" and "Green" classes, which show high true positive rates. However, misclassifications are observed, especially between the "Black" and "TTR" categories, suggesting potential confusion between visually similar items or limitations in text description differentiation.

- `incorrect_classifications.png`: This image displays examples of misclassified items, where instances labeled as "Black" were predicted as "TTR." These examples reveal some challenges in differentiating between these two categories, potentially due to similarities in visual appearance or insufficient context from the associated text. This observation suggests that additional feature extraction or model tuning could be beneficial for clearer differentiation.

- `roc_curve.png`: The ROC curve shows the model’s performance across each class, with AUC values ranging from 0.85 to 0.96. The "Green" category achieves the highest AUC (0.96), indicating strong classification performance for this class, while the "Black" category has a slightly lower AUC (0.85), reflecting greater classification difficulty. These results suggest that, while the model performs well overall, specific improvements may be needed to enhance accuracy for more challenging classes.


Link to the .pth file (because the size it's bigger than the limit allowed by GitHub): https://uofc-my.sharepoint.com/:u:/g/personal/felipe_castanogonzal_ucalgary_ca/ESzcIHApjDNDmt3UMMFzZEEBgakADYhgW9-MLArYFl_smQ?e=sbRWPH 
