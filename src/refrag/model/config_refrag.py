from transformers import PretrainedConfig


class RefragConfig(PretrainedConfig):
    model_type = "refrag"

    def __init__(
        self,
        encoder_name_or_path: str = "roberta-base",
        decoder_name_or_path: str ="ibm-granite/granite-4.0-350M",
        projector_hidden_dim: int = 768,
        compression: int = 4,
        pad_token_id: int = 0,
        training_model: str = "ped",  # letters: p=projector, e=encoder, d=decoder
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_name_or_path = encoder_name_or_path
        self.decoder_name_or_path = decoder_name_or_path
        self.projector_hidden_dim = projector_hidden_dim
        self.compression = compression
        self.pad_token_id = pad_token_id
        self.training_model = training_model.lower()

    @property
    def train_projector(self) -> bool:
        return "p" in self.training_model

    @property
    def train_encoder(self) -> bool:
        return "e" in self.training_model

    @property
    def train_decoder(self) -> bool:
        return "d" in self.training_model
