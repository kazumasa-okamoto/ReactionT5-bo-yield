"""Model utilities for ReactionT5-based yield prediction."""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, AutoConfig, PreTrainedModel
from typing import Dict, List, Union
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


class ReactionT5Yield(PreTrainedModel):
    """ReactionT5 model for yield prediction."""

    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(self.config._name_or_path)
        self.model.resize_token_embeddings(self.config.vocab_size)
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
        self.fc3 = nn.Linear(self.config.hidden_size // 2 * 2, self.config.hidden_size)
        self.fc4 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc5 = nn.Linear(self.config.hidden_size, 1)

        self._init_weights(self.fc1)
        self._init_weights(self.fc2)
        self._init_weights(self.fc3)
        self._init_weights(self.fc4)
        self._init_weights(self.fc5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids=None, attention_mask=None, inputs=None, **kwargs):
        if inputs is not None:
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
        if input_ids is None:
            raise ValueError('input_ids must be provided')

        device = input_ids.device

        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            encoder_hidden_states = encoder_outputs[0]  # (B, L, H)

            dec_input_ids = torch.full(
                (input_ids.size(0), 1),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )

            outputs = self.model.decoder(
                input_ids=dec_input_ids,
                encoder_hidden_states=encoder_hidden_states,
            )
            last_hidden_states = outputs[0]  # (B, 1, H)

        output1 = self.fc1(last_hidden_states.view(-1, self.config.hidden_size))
        output2 = self.fc2(encoder_hidden_states[:, 0, :].view(-1, self.config.hidden_size))
        output = self.fc3(torch.hstack((output1, output2)))
        output = self.fc4(output)
        output = self.fc5(output)
        return output * 100


def enable_dropout(model):
    """Enable dropout layers for MC Dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_inference(
    model,
    tokenizer,
    reaction_smiles: Union[str, List[str]],
    n_samples: int = 10,
    max_length: int = 512,
    batch_size: int = 64
) -> Dict[str, Union[float, List[float]]]:
    """
    MC Dropout-based inference for uncertainty estimation.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        reaction_smiles: Single reaction SMILES or list of SMILES
        n_samples: Number of MC samples
        max_length: Max sequence length
        batch_size: Batch size for prediction

    Returns:
        Dictionary with 'mean' and 'variance' keys
    """
    model.eval()
    enable_dropout(model)

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert single string to list
    if isinstance(reaction_smiles, str):
        reaction_smiles = [reaction_smiles]

    predictions = []

    with torch.no_grad():
        for sample_idx in range(n_samples):
            batch_predictions = []

            # Process in batches
            for i in range(0, len(reaction_smiles), batch_size):
                batch_smiles = reaction_smiles[i:i + batch_size]

                # Batch tokenization
                inputs = tokenizer(
                    batch_smiles,
                    return_tensors="pt",
                    max_length=max_length,
                    padding=True,
                    truncation=True
                ).to(device)

                # Batch prediction
                outputs = model(**inputs)
                batch_predictions.extend(outputs.cpu().numpy().flatten())

            predictions.append(np.array(batch_predictions))

    # Convert to array: (n_samples, n_reactions)
    predictions = np.array(predictions)

    # Calculate statistics
    mean_pred = np.mean(predictions, axis=0)
    var_pred = np.var(predictions, axis=0)

    if len(reaction_smiles) == 1:
        return {
            'mean': float(mean_pred[0]),
            'variance': float(var_pred[0])
        }
    else:
        return {
            'mean': mean_pred.tolist(),
            'variance': var_pred.tolist()
        }


def load_model_and_tokenizer(model_name: str = "sagawa/ReactionT5v2-yield"):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ReactionT5Yield.from_pretrained(model_name)
    return model, tokenizer
