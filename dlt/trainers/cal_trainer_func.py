import torch
import warnings
warnings.filterwarnings("ignore")

def sample_from_model(batch, model, device, diffusion, geometry_scale, diffusion_mode, image_pred_ox):
    shape = batch['geometry'].shape
    shape2 = batch['image_features'].shape
    model.eval()
    # generate initial noise
    noisy_batch = {
        'geometry': torch.randn(*shape, dtype=torch.float32, device=device)*geometry_scale.view(1, 1, 6).to(device)* batch['padding_mask'],
        "image_features": batch['image_features']
    }

    # sample x_0 = q(x_0|x_t)
    for i in range(diffusion.num_cont_steps)[::-1]:
        t = torch.tensor([i] * shape[0], device=device)
        
        if image_pred_ox:
            img_feature_noise = torch.randn(batch['image_features'].shape).to(device)
            noisy_image_features = diffusion.add_noise_Geometry(batch['image_features'], t, img_feature_noise) * batch['padding_mask_img']
            noisy_batch["image_features"] = noisy_image_features
            
        with torch.no_grad():
            # denoise for step t.
            if diffusion_mode == "sample":
                x0_pred = model(batch, noisy_batch, timesteps=t)
                geometry_pred = diffusion.inference_step(x0_pred[:,:,:6],
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['geometry'])
            elif diffusion_mode == "epsilon":
                epsilon_pred = model(batch, noisy_batch, timesteps=t)
                geometry_pred = diffusion.inference_step(epsilon_pred[:,:,:6],
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['geometry'])                
            
            noisy_batch['geometry'] = geometry_pred.prev_sample * batch['padding_mask']
        
        if i%200 == 0:
            print("############################################################################################################")
            print("step: ", i)
            print("batch: ", noisy_batch["geometry"][:,:,0].mean(), noisy_batch["geometry"][:,:,1].mean(), noisy_batch["geometry"][:,:,2].mean(), noisy_batch["geometry"][:,:,3].mean())
            print("############################################################################################################")
        
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(geometry_pred.pred_original_sample[:,:,0].mean(), geometry_pred.pred_original_sample[:,:,1].mean(), geometry_pred.pred_original_sample[:,:,2].mean(), geometry_pred.pred_original_sample[:,:,3].mean())
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    return geometry_pred.pred_original_sample



def refinement_from_model(batch, model, device, diffusion, geometry_scale, diffusion_mode, image_pred_ox):
    shape = batch['geometry'].shape
    shape2 = batch['image_features'].shape
    model.eval()
    # generate initial noise
    noise = torch.randn(*shape, dtype=torch.float32, device=device)*geometry_scale.view(1, 1, 6).to(device)* batch['padding_mask']
    t = torch.tensor([199] * shape[0], device=device)
    noisy_batch = {
        'geometry': diffusion.add_noise_Geometry(batch['geometry'], t, noise),
        "image_features": torch.randn(*shape2, dtype=torch.float32, device=device) #이미지의 가우시안 노이즈 
    }

    # sample x_0 = q(x_0|x_t)
    for i in range(200)[::-1]:
        t = torch.tensor([i] * shape[0], device=device)
        if image_pred_ox:
            img_feature_noise = torch.randn(batch['image_features'].shape).to(device)
            noisy_image_features = diffusion.add_noise_Geometry(batch['image_features'], t, img_feature_noise) * batch['padding_mask_img']
            noisy_batch["image_features"] = noisy_image_features
        with torch.no_grad():
            # denoise for step t.
            if diffusion_mode == "sample":
                x0_pred = model(batch, noisy_batch, timesteps=t)
                geometry_pred = diffusion.inference_step(x0_pred[:,:,:6],
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['geometry'])
            elif diffusion_mode == "epsilon":
                epsilon_pred = model(batch, noisy_batch, timesteps=t)
                geometry_pred = diffusion.inference_step(epsilon_pred[:,:,:6],
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['geometry'])                
            
            noisy_batch['geometry'] = geometry_pred.prev_sample * batch['padding_mask']
            
            
    return geometry_pred.pred_original_sample


def sample2dev(sample, device): # sample to device
    for k, v in sample.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                sample[k][k1] = v1.to(device)
        else:
            sample[k] = v.to(device)



def sample_timesteps(num_steps, batch_size, alpha, device):
        weight_explicit = torch.tensor([num_steps + alpha - t for t in range(num_steps)])
        weights = weight_explicit / weight_explicit.sum()
        # Sample from this distribution
        timesteps = torch.multinomial(weights, batch_size, replacement=True)

        return timesteps.to(device)