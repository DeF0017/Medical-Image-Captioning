from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")