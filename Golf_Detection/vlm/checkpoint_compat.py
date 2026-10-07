"""Legacy projector names in eadx's checkpoint, mapped to native MLX-VLM names."""
PROJECTOR_ALIASES = {
    "multi_modal_projector.mlp1_0.": "multi_modal_projector.layer_norm.",
    "multi_modal_projector.mlp1_1.": "multi_modal_projector.linear_1.",
    "multi_modal_projector.mlp1_3.": "multi_modal_projector.linear_2.",
}


def install_checkpoint_aliases():
    from mlx_vlm.models.locateanything import Model
    if getattr(Model, "_eadx_aliases_installed", False):
        return
    native_sanitize = Model.sanitize

    def sanitize(self, weights):
        renamed = {}
        for key, value in weights.items():
            target = key
            for old, new in PROJECTOR_ALIASES.items():
                if key.startswith(old):
                    target = new + key[len(old):]
                    break
            if target in renamed:
                raise ValueError(f"Checkpoint alias collision: {target}")
            renamed[target] = value
        return native_sanitize(self, renamed)

    Model.sanitize = sanitize
    Model._eadx_aliases_installed = True
