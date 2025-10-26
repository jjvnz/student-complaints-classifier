# StudentComplaintsClassifier

## Description

This project uses the **DistilBERT** model to classify university student PQRS (Petitions, Complaints, Claims, and Suggestions). The goal is to train a machine learning model capable of identifying and categorizing complaints based on their textual content.

## Project Structure

```bash
StudentComplaintsClassifier/
│
├── data/
│   ├── university_students_complaints_es.csv
│   ├── university_students_complaints_labeled.csv
│   ├── university_students_complaints_train.csv
│   └── university_students_complaints_val.csv
│
├── scripts/
│   ├── generate_data.py
│   ├── separate_dataset.py
│   ├── run_tokenizer.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── test_inference.py
│   └── gradio_interface.py
│
├── results/
│   └── evaluation_report.txt
│
├── train_dataset/
│
├── val_dataset/
│
└── fine-tuned-model/
    ├── config.json
    ├── model.safetensors
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt
```

* **`scripts/`**: Contains Python scripts for data processing and training.

  * **`label_data.py`**: Script for labeling data.
  * **`separate_dataset.py`**: Script for splitting data into training and validation sets.
  * **`run_tokenizer.py`**: Script for tokenizing data using the DistilBERT tokenizer.
  * **`train_model.py`**: Script for training the DistilBERT model with tokenized data.
* **`data/`**: Contains CSV files with complaints.

  * **`university_students_complaints_es.csv`**: Original complaints in Spanish.
  * **`university_students_complaints_labeled.csv`**: Labeled complaints file.
  * **`university_students_complaints_train.csv`**: Training data file.
  * **`university_students_complaints_val.csv`**: Validation data file.
* **`train_dataset/`**: Tokenized data for training.
* **`val_dataset/`**: Tokenized data for validation.
* **`fine-tuned-model/`**: Contains the trained model.
* **`results/`**: Contains the model evaluation results.
* **`README.md`**: This file.
* **`requirements.txt`**: Project dependencies file.

## Requirements

* **Python 3.10** or higher
* **transformers**
* **datasets**
* **pandas**
* **scikit-learn**
* **gradio**

Install dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run the Project

### 1. Prepare the Data

1. Generate data by running `scripts/generate_data.py`.
2. Split the data into training and validation sets using `scripts/separate_dataset.py`.

### 2. Tokenize the Data

Run the tokenizer script:

```bash
python scripts/run_tokenizer.py
```

### 3. Train the Model

Train the DistilBERT model:

```bash
python scripts/train_model.py
```

### 4. Save the Trained Model

The trained model will be saved in the `./fine-tuned-model` directory.

### 5. Run the Gradio Interface

Launch the Gradio app to test the model:

```bash
python scripts/gradio_interface.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

## Results

The model is evaluated during training, and the results are stored in the `./results` directory.

## Model Evaluation

After training, the model was evaluated on the validation dataset. The results were:

* **Precision**: 1.00
* **Recall**: 1.00
* **F1-Score**: 1.00
* **Accuracy**: 1.00

These results indicate perfect performance on the validation set.

## Contributions

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/new-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push your changes (`git push origin feature/new-feature`).
5. Open a Pull Request.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

## Contact

For questions or feedback, contact me at [jjvnz.dev@outlook.com](mailto:jjvnz.dev@outlook.com).
