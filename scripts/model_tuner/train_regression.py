
import argparse
import logging
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class SimpleRegressionConfig(PretrainedConfig):
    model_type = "simple_regression"

    def __init__(
        self,
        vocab_size=30522,
        embedding_dim=64,
        hidden_dim=64,
        max_position_embeddings=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_position_embeddings = max_position_embeddings


class SimpleRegressionModel(PreTrainedModel):
    config_class = SimpleRegressionConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pooler = nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        
        # Simple feed-forward network
        self.fc1 = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, 1) # Output 1 float (regression)
        
        # Initialize weights
        self.post_init()

    def init_weights(self):
        # Basic initialization, can be improved
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        
        # input_ids: [batch_size, seq_len]
        x = self.embeddings(input_ids) # [batch_size, seq_len, unique_embedding_dim]
        
        # Masking padding tokens if attention_mask is provided
        if attention_mask is not None:
            # Expand mask to match embedding dims
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
            x = x * mask_expanded
            
            # For average pooling, we need to divide by the actual length (sum of mask)
            # Use sum pooling then manual division to handle masking correctly
            x_sum = x.sum(dim=1) # [batch_size, embedding_dim]
            x_count = mask_expanded.sum(dim=1).clamp(min=1e-9) # Avoid division by zero
            pooled_output = x_sum / x_count
        else:
             # Fallback if no mask (though Trainer usually sends one)
            x_transposed = x.transpose(1, 2) # [batch_size, embedding_dim, seq_len] needed for AdaptiveAvgPool1d
            pooled_output = self.pooler(x_transposed).squeeze(-1) # [batch_size, embedding_dim]

        hidden = self.relu(self.fc1(pooled_output))
        logits = self.fc2(hidden) # [batch_size, 1]

        loss = None
        if labels is not None:
            # Ensure labels are float and correct shape
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight regression model for text objectivity.")
    
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data (JSON).")
    parser.add_argument("--val_file", type=str, help="Path to validation data (JSON).")
    parser.add_argument("--output_dir", type=str, default="./custom_regression_model", help="Directory to save model.")
    
    parser.add_argument("--tokenizer_name", type=str, default="allegro/herbert-base-cased", help="Tokenizer model name.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension.")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length.")
    
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training/eval.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Load datasets
    logger.info(f"Loading training data from {args.train_file}")
    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file
        
    dataset = load_dataset("json", data_files=data_files)
    
    # If no validation set, split train
    if "validation" not in dataset:
        logger.info("Splitting training set to create validation set (10%).")
        dataset = dataset["train"].train_test_split(test_size=0.1)
        dataset["validation"] = dataset.pop("test")

    # Verify columns
    if "text" not in dataset["train"].column_names or "score" not in dataset["train"].column_names:
        raise ValueError("Dataset must contain 'text' and 'score' columns.")

    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=args.max_len
        )
        tokenized["labels"] = [float(s) for s in examples["score"]]
        return tokenized

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    
    # Initialize Model
    logger.info("Initializing Custom Model...")
    config = SimpleRegressionConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        max_position_embeddings=args.max_len
    )
    model = SimpleRegressionModel(config)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False, # Lower MSE is better
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        report_to="none" # Disable wandb etc for simplicity
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Evaluation on validation set:")
    metrics = trainer.evaluate()
    logger.info(metrics)

if __name__ == "__main__":
    main()
