import gradio as gr
import numpy as np
import random
import torch
import spaces

from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from optimization import optimize_pipeline_
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

import math
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from PIL import Image
import os
import gradio as gr
from gradio_client import Client, handle_file
import tempfile
from typing import Optional, Tuple, Any


# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "linoyts/Qwen-Image-Edit-Rapid-AIO",
        subfolder='transformer',
        torch_dtype=dtype,
        device_map='cuda'
    ),
    torch_dtype=dtype
).to(device)

pipe.load_lora_weights(
    "dx8152/Qwen-Edit-2509-Multiple-angles",
    weight_name="镜头转换.safetensors",
    adapter_name="angles"
)

# pipe.load_lora_weights(
#         "lovis93/next-scene-qwen-image-lora-2509", 
#         weight_name="next-scene_lora-v2-3000.safetensors", adapter_name="next-scene"
#     )
pipe.set_adapters(["angles"], adapter_weights=[1.])
pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
# pipe.fuse_lora(adapter_names=["next-scene"], lora_scale=1.)
pipe.unload_lora_weights()

pipe.transformer.__class__ = QwenImageTransformer2DModel
pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

optimize_pipeline_(
    pipe,
    image=[Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))],
    prompt="prompt"
)

MAX_SEED = np.iinfo(np.int32).max


def _generate_video_segment(
    input_image_path: str,
    output_image_path: str,
    prompt: str,
    request: gr.Request
) -> str:
    """
    Generate a single video segment between two frames by calling an external
    Wan 2.2 image-to-video service hosted on Hugging Face Spaces.

    This helper function is used internally when the user asks to create
    a video between the input and output images.

    Args:
        input_image_path (str):
            Path to the starting frame image on disk.
        output_image_path (str):
            Path to the ending frame image on disk.
        prompt (str):
            Text prompt describing the camera movement / transition.
        request (gr.Request):
            Gradio request object, used here to forward the `x-ip-token`
            header to the downstream Space for authentication/rate limiting.

    Returns:
        str:
            A string returned by the external service, usually a URL or path
            to the generated video.
    """
    x_ip_token = request.headers['x-ip-token']
    video_client = Client(
        "multimodalart/wan-2-2-first-last-frame",
        headers={"x-ip-token": x_ip_token}
    )
    result = video_client.predict(
        start_image_pil=handle_file(input_image_path),
        end_image_pil=handle_file(output_image_path),
        prompt=prompt,
        api_name="/generate_video",
    )
    return result[0]["video"]


def build_camera_prompt(
    rotate_deg: float = 0.0,
    move_forward: float = 0.0,
    vertical_tilt: float = 0.0,
    wideangle: bool = False
) -> str:
    """
    Build a camera movement prompt based on the chosen controls.

    This converts the provided control values into a prompt instruction with the corresponding trigger words for the multiple-angles LoRA.

    Args:
        rotate_deg (float, optional):
            Horizontal rotation in degrees. Positive values rotate left,
            negative values rotate right. Defaults to 0.0.
        move_forward (float, optional):
            Forward movement / zoom factor. Larger values imply moving the
            camera closer or into a close-up. Defaults to 0.0.
        vertical_tilt (float, optional):
            Vertical angle of the camera:
            - Negative ≈ bird's-eye view
            - Positive ≈ worm's-eye view
            Defaults to 0.0.
        wideangle (bool, optional):
            Whether to switch to a wide-angle lens style. Defaults to False.

    Returns:
        str:
            A text prompt describing the camera motion. If no controls are
            active, returns `"no camera movement"`.
    """
    prompt_parts = []

    # Rotation
    if rotate_deg != 0:
        direction = "left" if rotate_deg > 0 else "right"
        if direction == "left":
            prompt_parts.append(
                f"将镜头向左旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the left."
            )
        else:
            prompt_parts.append(
                f"将镜头向右旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the right."
            )

    # Move forward / close-up
    if move_forward > 5:
        prompt_parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
    elif move_forward >= 1:
        prompt_parts.append("将镜头向前移动 Move the camera forward.")

    # Vertical tilt
    if vertical_tilt <= -1:
        prompt_parts.append("将相机转向鸟瞰视角 Turn the camera to a bird's-eye view.")
    elif vertical_tilt >= 1:
        prompt_parts.append("将相机切换到仰视视角 Turn the camera to a worm's-eye view.")

    # Lens option
    if wideangle:
        prompt_parts.append(" 将镜头转为广角镜头 Turn the camera to a wide-angle lens.")

    final_prompt = " ".join(prompt_parts).strip()
    return final_prompt if final_prompt else "no camera movement"


@spaces.GPU
def infer_camera_edit(
    image: Optional[Image.Image] = None,
    rotate_deg: float = 0.0,
    move_forward: float = 0.0,
    vertical_tilt: float = 0.0,
    wideangle: bool = False,
    seed: int = 0,
    randomize_seed: bool = True,
    true_guidance_scale: float = 1.0,
    num_inference_steps: int = 4,
    height: int = 1024,
    width: int = 1024,
    prev_output: Optional[Image.Image] = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True)
) -> Tuple[Image.Image, int, str]:
    """
    Edit the camera angles/view of an image with Qwen Image Edit 2509 and dx8152's Qwen-Edit-2509-Multiple-angles LoRA.

    Applies a camera-style transformation (rotation, zoom, tilt, lens)
    to an input image.

    Args:
        image (PIL.Image.Image | None, optional):
            Input image to edit. If `None`, the function will instead try to
            use `prev_output`. At least one of `image` or `prev_output` must
            be available. Defaults to None.
        rotate_deg (float, optional):
            Horizontal rotation in degrees (-90, -45, 0, 45, 90). Positive values rotate
            to the left, negative to the right. Defaults to 0.0.
        move_forward (float, optional):
            Forward movement / zoom factor (0, 5, 10). Higher values move the
            camera closer; values >5 switch to a close-up style. Defaults to 0.0.
        vertical_tilt (float, optional):
            Vertical tilt (-1 to 1). -1 ≈ bird's-eye view, +1 ≈ worm's-eye view.
            Defaults to 0.0.
        wideangle (bool, optional):
            Whether to use a wide-angle lens style. Defaults to False.
        seed (int, optional):
            Random seed for the generation. Ignored if `randomize_seed=True`.
            Defaults to 0.
        randomize_seed (bool, optional):
            If True, a random seed (0..MAX_SEED) is chosen per call.
            Defaults to True.
        true_guidance_scale (float, optional):
            CFG / guidance scale controlling prompt adherence.
            Defaults to 1.0 since the demo is using a distilled transformer for faster inference.
        num_inference_steps (int, optional):
            Number of inference steps. Defaults to 4.
        height (int, optional):
            Output image height. Must typically be a multiple of 8.
            If set to 0, the model will infer a size. Defaults to 1024 if none is provided.
        width (int, optional):
            Output image width. Must typically be a multiple of 8.
            If set to 0, the model will infer a size. Defaults to 1024 if none is provided.
        prev_output (PIL.Image.Image | None, optional):
            Previous output image to use as input when no new image is uploaded.
            Defaults to None.
        progress (gr.Progress, optional):
            Gradio progress tracker, automatically provided by Gradio in the UI.
            Defaults to a progress tracker with tqdm support.

    Returns:
        Tuple[PIL.Image.Image, int, str]:
            - The edited output image.
            - The actual seed used for generation.
            - The constructed camera prompt string.
    """
    prompt = build_camera_prompt(rotate_deg, move_forward, vertical_tilt, wideangle)
    print(f"Generated Prompt: {prompt}")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Choose input image (prefer uploaded, else last output)
    pil_images = []
    if image is not None:
        if isinstance(image, Image.Image):
            pil_images.append(image.convert("RGB"))
        elif hasattr(image, "name"):
            pil_images.append(Image.open(image.name).convert("RGB"))
    elif prev_output:
        pil_images.append(prev_output.convert("RGB"))

    if len(pil_images) == 0:
        raise gr.Error("Please upload an image first.")

    if prompt == "no camera movement":
        return image, seed, prompt

    result = pipe(
        image=pil_images,
        prompt=prompt,
        height=height if height != 0 else None,
        width=width if width != 0 else None,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    return result, seed, prompt


def create_video_between_images(
    input_image: Optional[Image.Image],
    output_image: Optional[np.ndarray],
    prompt: str,
    request: gr.Request
) -> str:
    """
    Create a short transition video between the input and output images via the 
    Wan 2.2 first-last-frame Space.

    Args:
        input_image (PIL.Image.Image | None):
            Starting frame image (the original / previous view).
        output_image (numpy.ndarray | None):
            Ending frame image - the output image with the the edited camera angles.
        prompt (str):
            The camera movement prompt used to describe the transition.
        request (gr.Request):
            Gradio request object, used to forward the `x-ip-token` header
            to the video generation app.

    Returns:
        str:
            a path pointing to the generated video.

    Raises:
        gr.Error:
            If either image is missing or if the video generation fails.
    """
    if input_image is None or output_image is None:
        raise gr.Error("Both input and output images are required to create a video.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            input_image.save(tmp.name)
            input_image_path = tmp.name

        output_pil = Image.fromarray(output_image.astype('uint8'))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            output_pil.save(tmp.name)
            output_image_path = tmp.name

        video_path = _generate_video_segment(
            input_image_path,
            output_image_path,
            prompt if prompt else "Camera movement transformation",
            request
        )
        return video_path
    except Exception as e:
        raise gr.Error(f"Video generation failed: {e}")


# --- UI ---
css = '''#col-container { max-width: 800px; margin: 0 auto; }
.dark .progress-text{color: white !important}
#examples{max-width: 800px; margin: 0 auto; }'''


def reset_all() -> list:
    """
    Reset all camera control knobs and flags to their default values.

    This is used by the "Reset" button to set:
    - rotate_deg = 0
    - move_forward = 0
    - vertical_tilt = 0
    - wideangle = False
    - is_reset = True

    Returns:
        list:
            A list of values matching the order of the reset outputs:
            [rotate_deg, move_forward, vertical_tilt, wideangle, is_reset, True]
    """
    return [0, 0, 0, 0, False, True]


def end_reset() -> bool:
    """
    Mark the end of a reset cycle.

    This helper is chained after `reset_all` to set the internal
    `is_reset` flag back to False, so that live inference can resume.

    Returns:
        bool:
            Always returns False.
    """
    return False


def update_dimensions_on_upload(
    image: Optional[Image.Image]
) -> Tuple[int, int]:
    """
    Compute recommended (width, height) for the output resolution when an
    image is uploaded while preserveing the aspect ratio.

    Args:
        image (PIL.Image.Image | None):
            The uploaded image. If `None`, defaults to (1024, 1024).

    Returns:
        Tuple[int, int]:
            The new (width, height).
    """
    if image is None:
        return 1024, 1024

    original_width, original_height = image.size

    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)

    # Ensure dimensions are multiples of 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    return new_width, new_height


with gr.Blocks(theme=gr.themes.Citrus(), css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("## 🎬 Qwen Image Edit — Camera Angle Control")
        gr.Markdown("""
            Qwen Image Edit 2509 for Camera Control ✨ 
            Using [dx8152's Qwen-Edit-2509-Multiple-angles LoRA](https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles) and [Phr00t/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/tree/main) for 4-step inference 💨
            """
        )

        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input Image", type="pil")
                prev_output = gr.Image(value=None, visible=False)
                is_reset = gr.Checkbox(value=False, visible=False)

                with gr.Tab("Camera Controls"):
                    rotate_deg = gr.Slider(
                        label="Rotate Right-Left (degrees °)",
                        minimum=-90,
                        maximum=90,
                        step=45,
                        value=0
                    )
                    move_forward = gr.Slider(
                        label="Move Forward → Close-Up",
                        minimum=0,
                        maximum=10,
                        step=5,
                        value=0
                    )
                    vertical_tilt = gr.Slider(
                        label="Vertical Angle (Bird ↔ Worm)",
                        minimum=-1,
                        maximum=1,
                        step=1,
                        value=0
                    )
                    wideangle = gr.Checkbox(label="Wide-Angle Lens", value=False)
                with gr.Row():
                    reset_btn = gr.Button("Reset")
                    run_btn = gr.Button("Generate", variant="primary")

                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0
                    )
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed",
                        value=True
                    )
                    true_guidance_scale = gr.Slider(
                        label="True Guidance Scale",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        value=1.0
                    )
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=1,
                        maximum=40,
                        step=1,
                        value=4
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=2048,
                        step=8,
                        value=1024
                    )
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=2048,
                        step=8,
                        value=1024
                    )

            with gr.Column():
                result = gr.Image(label="Output Image", interactive=False)
                prompt_preview = gr.Textbox(label="Processed Prompt", interactive=False)
                create_video_button = gr.Button(
                    "🎥 Create Video Between Images",
                    variant="secondary",
                    visible=False
                )
                with gr.Group(visible=False) as video_group:
                    video_output = gr.Video(
                        label="Generated Video",
                        show_download_button=True,
                        autoplay=True
                    )

    inputs = [
        image, rotate_deg, move_forward,
        vertical_tilt, wideangle,
        seed, randomize_seed, true_guidance_scale, num_inference_steps, height, width, prev_output
    ]
    outputs = [result, seed, prompt_preview]

    # Reset behavior
    reset_btn.click(
        fn=reset_all,
        inputs=None,
        outputs=[rotate_deg, move_forward, vertical_tilt, wideangle, is_reset],
        queue=False
    ).then(fn=end_reset, inputs=None, outputs=[is_reset], queue=False)

    # Manual generation with video button visibility control
    def infer_and_show_video_button(*args: Any):
        """
        Wrapper around `infer_camera_edit` that also controls the visibility
        of the 'Create Video Between Images' button.

        The first argument in `args` is expected to be the input image; if both
        input and output images are present, the video button is shown.

        Args:
            *args:
                Positional arguments forwarded directly to `infer_camera_edit`.

        Returns:
            tuple:
                (output_image, seed, prompt, video_button_visibility_update)
        """
        result_img, result_seed, result_prompt = infer_camera_edit(*args)
        # Show video button if we have both input and output images
        show_button = args[0] is not None and result_img is not None
        return result_img, result_seed, result_prompt, gr.update(visible=show_button)

    run_event = run_btn.click(
        fn=infer_and_show_video_button,
        inputs=inputs,
        outputs=outputs + [create_video_button]
    )

    # Video creation
    create_video_button.click(
        fn=lambda: gr.update(visible=True),
        outputs=[video_group],
        api_name=False
    ).then(
        fn=create_video_between_images,
        inputs=[image, result, prompt_preview],
        outputs=[video_output],
        api_name=False
    )

    # Examples
    gr.Examples(
        examples=[
            ["tool_of_the_sea.png", 90, 0, 0, False, 0, True, 1.0, 4, 568, 1024],
            ["monkey.jpg", -90, 0, 0, False, 0, True, 1.0, 4, 704, 1024],
            ["metropolis.jpg", 0, 0, -1, False, 0, True, 1.0, 4, 816, 1024],
            ["disaster_girl.jpg", -45, 0, 1, False, 0, True, 1.0, 4, 768, 1024],
            ["grumpy.png", 90, 0, 1, False, 0, True, 1.0, 4, 576, 1024]
        ],
        inputs=[
            image, rotate_deg, move_forward,
            vertical_tilt, wideangle,
            seed, randomize_seed, true_guidance_scale, num_inference_steps, height, width
        ],
        outputs=outputs,
        fn=infer_camera_edit,
        cache_examples="lazy",
        elem_id="examples"
    )

    # Image upload triggers dimension update and control reset
    image.upload(
        fn=update_dimensions_on_upload,
        inputs=[image],
        outputs=[width, height]
    ).then(
        fn=reset_all,
        inputs=None,
        outputs=[rotate_deg, move_forward, vertical_tilt, wideangle, is_reset],
        queue=False
    ).then(
        fn=end_reset,
        inputs=None,
        outputs=[is_reset],
        queue=False
    )

    # Live updates
    def maybe_infer(
        is_reset: bool,
        progress: gr.Progress = gr.Progress(track_tqdm=True),
        *args: Any
    ):
        if is_reset:
            return gr.update(), gr.update(), gr.update(), gr.update()
        else:
            result_img, result_seed, result_prompt = infer_camera_edit(*args)
            # Show video button if we have both input and output
            show_button = args[0] is not None and result_img is not None
            return result_img, result_seed, result_prompt, gr.update(visible=show_button)

    control_inputs = [
        image, rotate_deg, move_forward,
        vertical_tilt, wideangle,
        seed, randomize_seed, true_guidance_scale, num_inference_steps, height, width, prev_output
    ]
    control_inputs_with_flag = [is_reset] + control_inputs

    for control in [rotate_deg, move_forward, vertical_tilt]:
        control.release(
            fn=maybe_infer,
            inputs=control_inputs_with_flag,
            outputs=outputs + [create_video_button]
        )

    wideangle.input(
        fn=maybe_infer,
        inputs=control_inputs_with_flag,
        outputs=outputs + [create_video_button]
    )

    run_event.then(lambda img, *_: img, inputs=[result], outputs=[prev_output])

demo.launch(mcp_server=True)
