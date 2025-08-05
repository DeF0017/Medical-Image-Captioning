from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", 
                                                      device_map="auto")

def freeze_model_except_qformer(model):
    """Freeze all parameters except Q-Former"""
    # Freeze vision encoder
    for param in model.vision_model.parameters():
        param.requires_grad = False
    
    # Freeze language model
    for param in model.language_model.parameters():
        param.requires_grad = False
    
    # Only keep Q-Former parameters trainable
    for param in model.qformer.parameters():
        param.requires_grad = True
