import gc, pickle, torch, argparse, os, sys
import ptp_utils, seq_aligner
import numpy as np

from diffusers import  DiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, DDPMScheduler
from null import NullInversion
from local import AttentionStore, show_cross_attention, run_and_display, make_controller


ALPHA = 0.1



def main(args):
    
    if args.bigger:
        file_paths = [
            './pickle/x_t_p_1024.pkl',
            './pickle/uncond_embeddings_p_p_1024.pkl',
            './pickle/uncond_embeddings_p_1024.pkl']
        
    else:
        file_paths = [
            './pickle/x_t_p.pkl',
            './pickle/uncond_embeddings_p_p.pkl',
            './pickle/uncond_embeddings_p.pkl']
    
    
    all_files_exist = all(os.path.exists(path) for path in file_paths)

    if all_files_exist:
        content_is_none = False
        for path in file_paths:
            with open(path, 'rb') as file:
                content = pickle.load(file)
                if content is None:
                    content_is_none = True
                    break

        if content_is_none:
            print("One or more files are empty. Switching to verbose mode.")
            verbose = True
        else:
            print("All files exist and are non-empty")
            verbose = False
    else:
        print("All files are missing. Let's start the creation process.")
        verbose = True
    
    chprompt = args.chprompt
    neg_prompt = args.neg_prompt
    
    

    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    
    
    
    beta_end = args.beta_end 
    scheduler = DDIMScheduler(beta_start=0.0001, beta_end=beta_end, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = "stabilityai/stable-diffusion-xl-base-1.0"
    SDXL = DiffusionPipeline.from_pretrained(
            model,
            scheduler=scheduler,
            torch_dtype=torch.float32,
        ).to(device)


    def get_pickle_path(filename, bigger):
        return f'./pickle/{filename}_1024.pkl' if bigger else f'./pickle/{filename}.pkl'

    def save_to_pickle(data, filename, bigger):
        with open(get_pickle_path(filename, bigger), 'wb') as f:
            pickle.dump(data, f)

    def load_from_pickle(filename, bigger):
        with open(get_pickle_path(filename, bigger), 'rb') as f:
            return pickle.load(f)
        
    
    prompt = "a cat sitting next to a mirror"
    image_path = "./img/cat.jpg"
    offsets = (0, 0, 0, 0) 
    
    # prompt = "A smiling girl"
    # image_path = "./img/000000000038.jpg"
    # offsets = (0, 0, 0, 0) 

    
    # prompt = "black shirt man house"
    # image_path = "./img/000000000014.jpg"
    # offsets = (0, 0, 0, 0) 

    # prompt = "a white flower in spring"
    # image_path = "./img/000000000028.jpg"
    # offsets = (0, 0, 0, 0) 

    
    
    prompts = [prompt, prompt]
    neg_prompts =  [neg_prompt, neg_prompt]
    if verbose:
        null_inversion = NullInversion(SDXL)
        (image_gt, image_enc), x_t, uncond_embeddings, uncond_embeddings_p = null_inversion.invert(image_path, 
                                                                                                   prompt, 
                                                                                                   offsets = offsets,
                                                                                                   num_inner_steps=10, 
                                                                                                   early_stop_epsilon=1e-5, 
                                                                                                   verbose=verbose, 
                                                                                                   do_1024=args.bigger)
        torch.cuda.empty_cache()
        gc.collect()
        save_to_pickle(x_t, 'x_t_p', args.bigger)
        save_to_pickle(uncond_embeddings, 'uncond_embeddings_p', args.bigger)
        save_to_pickle(uncond_embeddings_p, 'uncond_embeddings_p_p', args.bigger)
        controller = AttentionStore()
        image_inv, x_t = run_and_display(SDXL,neg_prompts,prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, uncond_embeddings_p=uncond_embeddings_p,verbose=False)
        
        # ptp_utils.view_images([image_gt, image_enc, image_inv[0]])
        ptp_utils.view_images([image_gt, image_enc, image_inv[0]], num_rows=1, offset_ratio=0.02, params=None)
        ptp_utils.save_individual_images([image_gt, image_enc, image_inv[0]])
        show_cross_attention(SDXL,prompts,controller, 32, ["up"])
    else:
        x_t = load_from_pickle('x_t_p', args.bigger)
        uncond_embeddings = load_from_pickle('uncond_embeddings_p', args.bigger)
        uncond_embeddings_p = load_from_pickle('uncond_embeddings_p_p', args.bigger)

    
    neg_prompts = [neg_prompt, neg_prompt]     
    # prompts = ["A smiling girl",
    #         "A angry girl"]
    prompts = ["a cat sitting next to a mirror",
            "a tiger sitting next to a mirror"]
    
    # prompts = ["black shirt man house",
    #         "blue shirt man house"]
    
    for cross_value in [0.8]: 
        for self_value in [0.4]:
            for eq in [2]:
                params = {
                    "cross_int": cross_value,
                    "self_int": self_value,
                    "eq_int": eq,
                    "alpha":ALPHA,
                }
                cross_int = params["cross_int"]
                self_int = params["self_int"]
                eq_int = params["eq_int"]
                
                cross_replace_steps = {'default_': cross_int}
                self_replace_steps = self_int
                blend_word = ((('cat',), ("tiger",)))
                eq_params = {"words": ("tiger",), "values": (eq_int,)}
                # blend_word = ((('black',), ("blue",))) 
                # eq_params = {"words": ("blue",), "values": (eq_int,)}
                
                
                torch.cuda.empty_cache()
                gc.collect()
                controller = make_controller(SDXL, prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
                images, _ = run_and_display(SDXL, neg_prompts, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, uncond_embeddings_p=uncond_embeddings_p, steps=50)
                ptp_utils.view_images(images, num_rows=1, offset_ratio=0.02, params=params)
                ptp_utils.save_first_image(images, params=params)




if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, default="./img/gnochi_mirror.jpeg", help="Image Path")
    p.add_argument("--prompt", type=str, default="a cat sitting next to a mirror", help="Positive Prompt") 
    p.add_argument("--chprompt", type=str, default="a cat sitting next to a mirror", help="Positive Prompt") 
    p.add_argument("--neg_prompt", type=str, default="", help="Negative Prompt")  
    p.add_argument("--beta_end", type=float, default=0.012, help="Negative Prompt")  
    p.add_argument("--bigger", action='store_true', help="If you want to create an image 1024")
  
    args = p.parse_args()

    
    main(args)
  


