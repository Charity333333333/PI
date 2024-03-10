from fairseq.models import register_model, register_model_architecture
from fairseq.models.multilingual_transformer import MultilingualTransformerModel
from fairseq.models.transformer import base_architecture

@register_model('parameter_inheritance_model')
class ParameterInheritanceModel(MultilingualTransformerModel):
    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)
        shared_model = self.models[self.keys[0]]
        for key in self.keys[1:]:
            # share encoder
            for layer_idx in range(len(shared_model.encoder.layers)):
                self.models[key].encoder.layers[layer_idx].self_attn.k_proj = shared_model.encoder.layers[layer_idx].self_attn.k_proj
                self.models[key].encoder.layers[layer_idx].self_attn.v_proj = shared_model.encoder.layers[layer_idx].self_attn.v_proj
                self.models[key].encoder.layers[layer_idx].self_attn.q_proj = shared_model.encoder.layers[layer_idx].self_attn.q_proj
                self.models[key].encoder.layers[layer_idx].self_attn.out_proj = shared_model.encoder.layers[layer_idx].self_attn.out_proj
                self.models[key].encoder.layers[layer_idx].fc1 = shared_model.encoder.layers[layer_idx].fc1
                self.models[key].encoder.layers[layer_idx].fc2 = shared_model.encoder.layers[layer_idx].fc2
                self.models[key].encoder.layers[layer_idx].self_attn_layer_norm = shared_model.encoder.layers[layer_idx].self_attn_layer_norm
                self.models[key].encoder.layers[layer_idx].final_layer_norm = shared_model.encoder.layers[layer_idx].final_layer_norm

            # share decoder
            for layer_idx in range(len(shared_model.decoder.layers)):
                self.models[key].decoder.layers[layer_idx].self_attn.k_proj = shared_model.decoder.layers[layer_idx].self_attn.k_proj
                self.models[key].decoder.layers[layer_idx].self_attn.v_proj = shared_model.decoder.layers[layer_idx].self_attn.v_proj
                self.models[key].decoder.layers[layer_idx].self_attn.q_proj = shared_model.decoder.layers[layer_idx].self_attn.q_proj
                self.models[key].decoder.layers[layer_idx].self_attn.out_proj = shared_model.decoder.layers[layer_idx].self_attn.out_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.k_proj = shared_model.decoder.layers[layer_idx].encoder_attn.k_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.v_proj = shared_model.decoder.layers[layer_idx].encoder_attn.v_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.q_proj = shared_model.decoder.layers[layer_idx].encoder_attn.q_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.out_proj = shared_model.decoder.layers[layer_idx].encoder_attn.out_proj
                self.models[key].decoder.layers[layer_idx].fc1 = shared_model.decoder.layers[layer_idx].fc1
                self.models[key].decoder.layers[layer_idx].fc2 = shared_model.decoder.layers[layer_idx].fc2
                self.models[key].decoder.layers[layer_idx].self_attn_layer_norm = shared_model.decoder.layers[layer_idx].self_attn_layer_norm
                self.models[key].decoder.layers[layer_idx].encoder_attn_layer_norm = shared_model.decoder.layers[layer_idx].encoder_attn_layer_norm
                self.models[key].decoder.layers[layer_idx].final_layer_norm = shared_model.decoder.layers[layer_idx].final_layer_norm


    @classmethod
    def build_model(cls, args, task):
        model = super(ParameterInheritanceModel, cls).build_model(args, task)
        encoders = {key: model.models[key].encoder for key in model.keys}
        decoders = {key: model.models[key].decoder for key in model.keys}
        return cls(encoders, decoders)


@register_model_architecture("parameter_inheritance_model", "parameter_inheritance_model")
def base_parameter_inheritance_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)
