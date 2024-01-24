from . import attentionpatch
from . import transformerpatch

attention_forward = attentionpatch.default.forward
basic_transformer_forward = transformerpatch.default.forward

def reset():
   global attention_forward, basic_transformer_forward
   attention_forward = attentionpatch.default.forward
   basic_transformer_forward = transformerpatch.default.forward
