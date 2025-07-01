import cv2, lpips, csv, os
import numpy as np
import torch, datetime, pytz

from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, Tuple, List, Callable, Dict

from tqdm.notebook import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
# from IPython.display import display

from run import ALPHA as alpha



def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
):
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, params=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    seoul_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.datetime.now(seoul_tz).strftime("%Y-%m-%dT%H-%M-%S")
    
    if params is not None:
        # file_name = f"./result/{current_time}_down_alpha{params['alpha']}@@@_cross{params['cross_int']}_self{params['self_int']}_eq{params['eq_int']}.png"
        file_name = f"./result/{current_time}_alpha{params['alpha']}_cross{params['cross_int']}_self{params['self_int']}_eq{params['eq_int']}.png"
    else:
        file_name = f"./result/{current_time}.png"
        
    
    pil_img.save(file_name)
    print(f"Image saved as {file_name}") 
    print(pil_img)
    # display(pil_img)

def save_first_image(images, params=None):
    if len(images) == 2:
        first_image = images[1].astype(np.uint8)
        pil_img = Image.fromarray(first_image)
        seoul_tz = pytz.timezone("Asia/Seoul")
        current_time = datetime.datetime.now(seoul_tz).strftime("%Y-%m-%dT%H-%M-%S")
        
        if params is not None:
            file_name = f"./result/{current_time}_alpha{params['alpha']}_cross{params['cross_int']}_self{params['self_int']}_eq{params['eq_int']}.png"
        else:
            file_name = f"./result/{current_time}.png"
        
        pil_img.save(file_name)
        print(f"First image saved as {file_name}") 
        print(pil_img)
        # display(pil_img)
    else:
        print("The function is designed to save the first image only if exactly two images are provided.")


def save_individual_images(images, directory="./result"):
    if not isinstance(images, list):
        images = [images]

    if len(images) < 3:
        raise ValueError("At least three images are required to compare index 0 and 2.")

    # Saving images
    pil_img = Image.fromarray(images[2].astype(np.uint8))
    pil_img2 = Image.fromarray(images[1].astype(np.uint8))

    seoul_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.datetime.now(seoul_tz).strftime("%Y-%m-%dT%H-%M-%S")

    percept = lpips.LPIPS(net="vgg").cuda()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Comparing images at index 0 and 2
    images = np.array(images)
    image0 = images[0].astype(np.uint8)
    image1 = images[1].astype(np.uint8)
    image2 = images[2].astype(np.uint8)

    imageA_t = transform(image0).unsqueeze(0).cuda()
    imageB_t = transform(image2).unsqueeze(0).cuda()

    imageC_t = transform(image1).unsqueeze(0).cuda()

    psnr_value = psnr(image0, image2)
    ssim_value, _ = ssim(image0, image2, full=True, channel_axis=2, win_size=7)
    lpips_value = percept(imageA_t, imageB_t).item()

    psnr_value_1 = psnr(image0, image1)
    ssim_value_1, _ = ssim(image0, image1, full=True, channel_axis=2, win_size=7)
    lpips_value_1 = percept(imageA_t, imageC_t).item()

    print("---- DDIM inv ----")
    print(f"⛴️ PSNR original vs new: {psnr_value_1:.3f}")
    print(f"⛴️ SSIM original vs new: {ssim_value_1:.3f}")
    print(f"⛴️ LPIPS original vs new: {lpips_value_1:.3f}")
    print("---- Null text optim ----")
    print(f"⭐️ PSNR original vs new: {psnr_value:.3f}")
    print(f"⭐️ SSIM original vs new: {ssim_value:.3f}")
    print(f"⭐️ LPIPS original vs new: {lpips_value:.3f}")

    file_path = f"{directory}/{current_time}_{psnr_value:.3f}_new.png"
    file_path_2 = f"{directory}/{current_time}_ori.png"
    pil_img2.save(file_path_2)
    pil_img.save(file_path)
    print(f"Image saved as {file_path}")


def make_dataset(images, directory="./new_dataset", image_path=None):
    if not isinstance(images, list):
        images = [images]

    if len(images) < 3:
        raise ValueError("At least three images are required to compare index 0 and 2.")

    # Saving images
    pil_img = Image.fromarray(images[2].astype(np.uint8))

    file_path = f"{directory}/new_{image_path}"
    pil_img.save(file_path)
    print(f"Image saved as {file_path}")

    percept = lpips.LPIPS(net="vgg").cuda()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Comparing images at index 0 and 2
    image0 = images[0].astype(np.uint8)
    image2 = images[2].astype(np.uint8)

    imageA_t = transform(image0).unsqueeze(0).cuda()
    imageB_t = transform(image2).unsqueeze(0).cuda()

    psnr_value = psnr(image0, image2)
    ssim_value, _ = ssim(image0, image2, full=True, channel_axis=2, win_size=7)
    lpips_value = percept(imageA_t, imageB_t).item()

    print(f"⭐️ PSNR original vs new: {psnr_value:.3f}")
    print(f"⭐️ SSIM original vs new: {ssim_value:.3f}")
    print(f"⭐️ LPIPS original vs new: {lpips_value:.3f}")

    # print(f" LPIPS took a long time so I excluded it. Check it out later in results.txt! ")


def diffusion_step(
    model,
    controller,
    latents,
    context,
    context_p,
    add_time_ids,
    t,
    guidance_scale,
    low_resource=False,
):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])[
            "sample"
        ]
        noise_prediction_text = model.unet(
            latents, t, encoder_hidden_states=context[1]
        )["sample"]
    else:
        latents_input = torch.cat([latents] * 2)

        added_cond_kwargs = {"text_embeds": context_p, "time_ids": add_time_ids}
        noise_pred = model.unet(
            latents_input,
            t,
            encoder_hidden_states=context,
            added_cond_kwargs=added_cond_kwargs,
        )["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    t = t.to(noise_pred.device)
    latents = latents.to(noise_pred.device)

    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):

    latents = 1 / 0.13025 * latents
    image = vae.decode(latents)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        print("init_latent-- latent is None:")
        latent = torch.randn(
            (1, model.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
    if batch_size == 2:
        if latent.size()[-1] == 128:

            latents = latent.repeat(2,1,1,1).to(model.device)
        elif latent.size()[-1] == 32:

            latents = latent.repeat(2,1,1,1).to(model.device)

        else:

            latents = latent.repeat(2,1,1,1).to(model.device)
    else:
        latents = latent

    latents = latents * model.scheduler.init_noise_sigma

    return latent, latents

def get_attention_scores_custom(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        # to_out = self.to_out
        # if type(to_out) is torch.nn.modules.container.ModuleList:
        #     to_out = self.to_out[0]
        # else:
        #     to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,scale=1.0):
            if self.residual_connection:
                print("🌸"*40)


            is_cross = encoder_hidden_states is not None    
            #residual save            
            residual = hidden_states
            query = self.to_q(hidden_states) * 1.0

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
            
            if self.group_norm is not None:
                
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
                
            # encoder_hidden_states, torch.Size([1, 77, 2048])
            # hidden_states, torch.Size([1, 1024, 640])
            # torch.Size([1, 256, 1280])
            # torch.Size([1, 77, 2048])


            key = self.to_k(encoder_hidden_states) * 1.0
            value = self.to_v(encoder_hidden_states) * 1.0

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            baddbmm_input = None
            # baddbmm_input = torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device).requires_grad_(True)

            attention_probs = get_attention_scores_custom(self, query, key, attention_mask=baddbmm_input)
            # <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f0c108019f0>
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.matmul(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)                
            hidden_states = self.to_out[0](hidden_states) * 1.0
            hidden_states = self.to_out[1](hidden_states)


            
            
            if alpha is not None:
                mid_scale, down_scale = alpha, alpha
            
                if self.to_k.in_features != self.to_q.in_features:
                    pass
 
                #------------------------------------cross attention------------------------------------------------------
                else:
                #-------------------------------------self attention ----------------------------------------------------        
                    if place_in_unet == "down": 
                        pass
                        # hidden_states = (1-down_scale)*hidden_states + residual*(down_scale)

                    #-----------------------------------------------------------------------------------------------
                    elif place_in_unet == "mid": 
                        # hidden_states = (1- mid_scale)*hidden_states + residual*(mid_scale)
                        pass
                            
                    #-----------------------------------------------------------------------------------------------
                    elif place_in_unet == "up":
                        hidden_states = (1-down_scale)*hidden_states + residual*(down_scale)

                    
            
            # return to_out(hidden_states)
            return hidden_states

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        print("Dummy Controller Declaration because there is no Controller")
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def get_word_inds(text: str, word_place: int, tokenizer):


    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [
            tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)
        ][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(
    alpha,
    bounds: Union[float, Tuple[float, float]],
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts,
    num_steps,
    cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
    tokenizer,
    max_num_words=77,
):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)

    num_steps = 49
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(
            alpha_time_words, cross_replace_steps["default_"], i
        )
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [
                get_word_inds(prompts[i], key, tokenizer)
                for i in range(1, len(prompts))
            ]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(
                        alpha_time_words, item, i, ind
                    )
    alpha_time_words = alpha_time_words.reshape(
        num_steps + 1, len(prompts) - 1, 1, 1, max_num_words
    )
    return alpha_time_words