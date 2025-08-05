import torch
import torch.nn as nn
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import AutoModel, AutoImageProcessor
from copy import deepcopy

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", 
                                                      device_map=None)
dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-small')
dino_model = AutoModel.from_pretrained('facebook/dinov2-with-registers-small')

processor.tokenizer.add_special_tokens({
    "bos_token": "<BOS>",
    "eos_token": "<EOS>"
})

processor.tokenizer.bos_token = "<BOS>"
processor.tokenizer.eos_token = "<EOS>"

# If needed, also make sure they are added to the tokenizer vocab
processor.tokenizer.add_tokens(["<BOS>", "<EOS>"])

print("BOS Token:", processor.tokenizer.bos_token)
print("EOS Token:", processor.tokenizer.eos_token)

class MultiXModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load BLIP2-OPT model
        self.blip_model = model
        self.blip_qformer = model.qformer  # Q-Former for BLIP

        # Create a second Q-Former for DINOv2 with the same config
        from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel
        qformer_config = deepcopy(self.blip_qformer.config)
        self.dino_qformer = Blip2QFormerModel(config=qformer_config)

        # Load DINOv2 model
        self.dino_model = dino_model

        # Vision dimensions
        self.blip_hidden_size = self.blip_model.vision_model.config.hidden_size  # typically 1408
        self.dino_hidden_size = self.dino_model.config.hidden_size  # e.g., 384 for small
        self.qformer_hidden_size = self.blip_qformer.config.hidden_size  # typically 768
        self.qformer_input_dim = 1408

        # Projection layers to match Q-Former input
        self.blip_proj = nn.Linear(self.blip_hidden_size, self.qformer_input_dim)
        self.dino_proj = nn.Linear(self.dino_hidden_size, self.qformer_input_dim)

        # Language model input projection (Q-Former -> decoder)
        self.opt_proj = nn.Linear(self.qformer_hidden_size, self.blip_model.language_model.config.hidden_size)

    def extract_blip_features(self, pixel_values, return_attn=False):
        with torch.no_grad():
            vision_outputs = self.blip_model.vision_model(
                pixel_values=pixel_values,
                output_attentions=return_attn,
                output_hidden_states=True
            )
        if return_attn:
            return vision_outputs.last_hidden_state, vision_outputs.attentions
        return vision_outputs.last_hidden_state  # shape: [B, T1, 1408]

    def extract_dino_features(self, pixel_values, return_attn=False):
        with torch.no_grad():
            dino_outputs = self.dino_model(pixel_values=pixel_values, output_attentions=return_attn)
        if return_attn:
            return dino_outputs.last_hidden_state, dino_outputs.attentions
        return dino_outputs.last_hidden_state  # shape: [B, T2, dino_hidden_size]

    def forward(self, pixel_values, dino_pixel_values, input_ids=None, attention_mask=None, labels=None):
        # BLIP vision features
        blip_features = self.extract_blip_features(pixel_values)
        blip_features = self.blip_proj(blip_features)

        # DINOv2 vision features
        dino_features = self.extract_dino_features(dino_pixel_values)
        dino_features = self.dino_proj(dino_features)

        # Attention masks
        blip_attn_mask = torch.ones(blip_features.shape[:2], dtype=torch.long, device=blip_features.device)
        dino_attn_mask = torch.ones(dino_features.shape[:2], dtype=torch.long, device=dino_features.device)

        # Query tokens
        query_tokens = self.blip_model.query_tokens

        # Run Q-Former on BLIP features
        blip_q_out = self.blip_qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=blip_features,
            encoder_attention_mask=blip_attn_mask,
            return_dict=True
        )
        blip_output = blip_q_out.last_hidden_state

        # Run Q-Former on DINO features
        dino_q_out = self.dino_qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=dino_features,
            encoder_attention_mask=dino_attn_mask,
            return_dict=True
        )
        dino_output = dino_q_out.last_hidden_state

        # Concatenate Q-Former outputs
        enhanced_embeddings = torch.cat([blip_output, dino_output], dim=1)

        # Project for language model
        enhanced_embeddings = self.opt_proj(enhanced_embeddings)

        # Prepare input embeddings for language model
        input_embeds = self.blip_model.language_model.model.decoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([enhanced_embeddings, input_embeds], dim=1)

        # Adjust labels and attention mask
        B, V, _ = enhanced_embeddings.shape
        padding = torch.full((B, V), -100, dtype=labels.dtype, device=labels.device)
        labels = torch.cat([padding, labels], dim=1)

        vision_attention_mask = torch.ones((B, V), dtype=attention_mask.dtype, device=attention_mask.device)
        extended_attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)

        outputs = self.blip_model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs

    @torch.no_grad()
    def get_attention_maps(self, pixel_values, dino_pixel_values):
        self.eval()
        
        # BLIP
        blip_feats, blip_attn = self.extract_blip_features(pixel_values, return_attn=True)

        # DINO
        dino_feats, dino_attn = self.extract_dino_features(dino_pixel_values, return_attn=True)

        return {
            "blip_attention": blip_attn,  # List of attention maps [num_layers x (B, num_heads, seq_len, seq_len)]
            "dino_attention": dino_attn
        }

    @torch.no_grad()
    def generate_caption(self, pixel_values, dino_pixel_values, max_length=200, min_length=100,num_beams=5, no_repeat_ngram_size=3, length_penalty=2.0, temperature=0.6, top_k=100, top_p=0.9):
        self.eval()

        device = pixel_values.device

        # Step 1: Extract visual features
        blip_features = self.extract_blip_features(pixel_values)
        blip_features = self.blip_proj(blip_features)

        dino_features = self.extract_dino_features(dino_pixel_values)
        dino_features = self.dino_proj(dino_features)

        # Step 2: Q-Former attention
        B = blip_features.shape[0]
        blip_attn_mask = torch.ones(blip_features.shape[:2], dtype=torch.long, device=device)
        dino_attn_mask = torch.ones(dino_features.shape[:2], dtype=torch.long, device=device)

        query_tokens = self.blip_model.query_tokens

        blip_q_out = self.blip_qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=blip_features,
            encoder_attention_mask=blip_attn_mask,
            return_dict=True
        )
        dino_q_out = self.dino_qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=dino_features,
            encoder_attention_mask=dino_attn_mask,
            return_dict=True
        )

        # Step 3: Fuse and project to decoder
        enhanced_embeddings = torch.cat([blip_q_out.last_hidden_state, dino_q_out.last_hidden_state], dim=1)
        enhanced_embeddings = self.opt_proj(enhanced_embeddings)

        # Step 4: Prepare for generation
        language_model = self.blip_model.language_model

        generated_ids = language_model.generate(
            inputs_embeds=enhanced_embeddings,
            max_length=max_length,
            min_length=min_length,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        return generated_ids