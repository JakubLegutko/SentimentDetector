import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch

def main():
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU not found. Training will be slow.")
    parser = argparse.ArgumentParser(description="Fine-tune a DeBERTa model for sentiment detection.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="The name of the pre-trained model to use.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset to use for fine-tuning.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sentiment_model",
        help="The directory where the fine-tuned model will be saved.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="The number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="The batch size per device during training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="The batch size per device during evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="The learning rate for the optimizer.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training (faster, less memory).",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy.",
    )

    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Regression task (1 label)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

    # Load dataset
    if args.dataset_name.endswith('.json'):
        import json
        with open(args.dataset_name, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        texts = []
        labels = []
        for item in raw_data:
            if 'text' in item and 'predicted_score' in item and item['predicted_score'] is not None:
                texts.append(item['text'])
                labels.append(float(item['predicted_score']))
                
        # Split into train and eval (80/20)
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        from datasets import Dataset, DatasetDict
        train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
        val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
        dataset = DatasetDict({'train': train_dataset, 'test': val_dataset})
    else:
        dataset = load_dataset(args.dataset_name)

    # Preprocess the dataset
    def preprocess_function(examples):
        # Use simple truncation. We will handle dynamic padding using a DataCollator
        # DeBERTa-v3 base is pretrained on max 512 length. Exceeding this drastically increases memory.
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Note: transformers Trainer automatically expects the target to be named "label" or "labels" for MSELoss regression when num_labels=1

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        fp16=args.fp16,
        save_strategy="epoch",
        load_best_model_at_end=True if args.eval_strategy != "no" else False,
        metric_for_best_model="eval_loss" if args.eval_strategy != "no" else None, # Evaluate using loss for regression
        greater_is_better=False,
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
