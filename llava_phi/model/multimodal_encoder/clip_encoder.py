from abc import ABC

import torch
import torch.nn as nn

from transformers import CLIPPreTrainedModel, CLIPVisionConfig, AutoTokenizer
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPTextModel
from llava_phi.model.language_model.configuration_llava_phi import LlavaPhiVisionConfig


class CLIPVisionTower(CLIPPreTrainedModel):
    config_class = LlavaPhiVisionConfig

    def __init__(self, config):
        super().__init__(config)

        self.vision_model = CLIPVisionTransformer(config)
        self.text_encoder = CLIPTextModel.from_pretrained(config.vision_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.vision_model_name_or_path)
        # Initialize weights and apply final processing
        self.post_init()
        self.create_text_modules()
        self.load_text_modules()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.config.mm_vision_select_layer]
        if self.config.mm_vision_select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.config.mm_vision_select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.config.mm_vision_select_feature}')
        return image_features
    def create_text_modules(self):

        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, self.config.hidden_size,
                                                         device=self.device)
        num_image_tokens = int(self.config.image_size / self.config.patch_size) ** 2
        num_text_tokens = 64
        fusion_parameter = torch.zeros((num_image_tokens + num_text_tokens, num_image_tokens), dtype=self.dtype,
                                       device=self.device)
        fusion_parameter[:num_image_tokens, :num_image_tokens] = torch.eye(num_image_tokens, dtype=self.dtype,
                                                                           device=self.device)
        self.image_text_infusion = nn.Linear(num_image_tokens + num_text_tokens, num_image_tokens, bias=True,
                                             device=self.device)
        self.image_text_infusion.weight = nn.Parameter(fusion_parameter.permute(1, 0), requires_grad=True)
        self.image_text_infusion.bias = nn.Parameter(torch.zeros(num_image_tokens, device=self.device),
                                                     requires_grad=True)

    def load_text_modules(self):
        if self.text_module_path:
            self.text_module_path = self.text_module_path.replace('finetune', 'pretrain')
            self.text_module_path = os.path.join(self.text_module_path, 'vision_tower.bin')
            state_dict = torch.load(self.text_module_path)
            text_projection_state_dict = dict()
            image_text_infusion_dict = dict()
            for key, value in state_dict.items():
                if key.startswith('text_projection'):
                    text_projection_state_dict[key.replace('text_projection.', '')] = value
                elif key.startswith('image_text_infusion'):
                    image_text_infusion_dict[key.replace('image_text_infusion.', '')] = value
            self.text_projection.load_state_dict(text_projection_state_dict)
            self.image_text_infusion.load_state_dict(image_text_infusion_dict)
            print('Text modules were loaded successfully!')
        else:
            print('There is no pretrained version of Text_Projection and Image_Text_Infusion')
            
    def forward(self, images, instruct):
        with torch.no_grad():
            text_features = self.text_encoder(instruct.to(device=self.device), output_hidden_states=True).hidden_states[
                self.config.mm_vision_select_layer]
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.vision_model(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                        output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
            else:
                image_forward_outs = self.vision_model(images.to(device=self.device, dtype=self.dtype),
                                                    output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
                
        text_features = nn.GELU()(self.text_projection(text_features).to(self.dtype))
        infused_image_features = torch.cat((image_features, text_features), dim=1)
        infused_image_features = self.image_text_infusion(infused_image_features.permute(0, 2, 1)).permute(0, 2, 1)
        return infused_image_features

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return list(self.vision_model.parameters())[0].dtype

    @property
    def device(self):
        return list(self.vision_model.parameters())[0].device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


if __name__ == '__main__':
    clip_config = CLIPVisionConfig.from_pretrained(
        "/data/private/zhumj/GPTcode/mm-phi/openai/clip-vit-large-patch14-336"
    )
    print("################ clip_config ##############")
    print(clip_config)
    phi_vis_config = LlavaPhiVisionConfig(**clip_config.to_dict())
    print("################ phi_vis_config ##############")
    print(phi_vis_config)

    model = CLIPVisionTower(clip_config)
    # print(list(model.vision_model.parameters())[0].dtype)
