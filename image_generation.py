import replicate

output = replicate.run(
    "ai-forever/kandinsky-2:3c6374e7a9a17e01afe306a5218cc67de55b19ea536466d6ea2602cfecea40a9",
    input={
        "width": 256,
        "height": 256,
        "prompt": "a happy human face image, realistic and with a slight shy smile and the shoulders are shown to the top of the head and open nice eyes",
        "scheduler": "p_sampler",
        "batch_size": 4,
        "prior_steps": "5",
        "output_format": "jpg",
        "guidance_scale": 4,
        "output_quality": 80,
        "prior_cf_scale": 4,
        "num_inference_steps": 100
    }
)
print(output)