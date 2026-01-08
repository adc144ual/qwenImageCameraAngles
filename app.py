import gradio as gr
import numpy as np
import random
import torch
import spaces

from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

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

pipe.set_adapters(["angles"], adapter_weights=[1.])
pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
pipe.unload_lora_weights()

spaces.aoti_blocks_load(pipe.transformer, "zerogpu-aoti/Qwen-Image", variant="fa3")

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
    """
    x_ip_token = request.headers['x-ip-token']
    video_client = Client(
        "multimodalart/wan-2-2-first-last-frame",
        headers={"x-ip-token": x_ip_token}
    )
    result = video_client.predict(
        start_image_pil=handle_file(input_image_path),
        end_image_pil=handle_file(output_image_path),
        prompt=prompt if prompt else "Camera movement transformation",
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
        prompt_parts.append("将镜头转为广角镜头 Turn the camera to a wide-angle lens.")

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
    height: Optional[int] = None,
    width: Optional[int] = None,
    prev_output: Optional[Image.Image] = None,
) -> Tuple[Image.Image, int, str]:
    """
    Edit the camera angles/view of an image with Qwen Image Edit 2509.
    """
    progress = gr.Progress(track_tqdm=True)
    
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
    Create a short transition video between the input and output images.
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


def update_dimensions_on_upload(image: Optional[Image.Image]) -> Tuple[int, int]:
    """Compute recommended dimensions preserving aspect ratio."""
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

    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    return new_width, new_height


def reset_all() -> list:
    """Reset all camera control knobs and flags to their default values."""
    return [0, 0, 0, False, True]


def end_reset() -> bool:
    """Mark the end of a reset cycle."""
    return False


# --- UI ---
css = '''
#col-container { max-width: 1200px; margin: 0 auto; }
.dark .progress-text { color: white !important; }
#camera-3d-control { min-height: 450px; }
#examples { max-width: 1200px; margin: 0 auto; }

/* Custom styling for the 3D control */
.camera-3d-wrapper {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.control-legend {
    display: flex;
    gap: 16px;
    justify-content: center;
    padding: 12px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    margin-top: 8px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: #e0e0e0;
}

.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
}

.legend-dot.rotation { background: #00ff88; }
.legend-dot.tilt { background: #ff69b4; }
.legend-dot.zoom { background: #ffa500; }
'''

# Three.js script to load
three_js_head = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'

# 3D Camera Control HTML Component
camera_3d_html = """
<div id="camera-control-wrapper" style="width: 100%; height: 400px; position: relative; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; overflow: hidden;">
    <div id="prompt-overlay" style="position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.85); padding: 10px 20px; border-radius: 10px; font-family: 'SF Mono', 'Consolas', monospace; font-size: 11px; color: #00ff88; white-space: nowrap; z-index: 10; border: 1px solid rgba(0, 255, 136, 0.3); max-width: 90%; overflow: hidden; text-overflow: ellipsis;"></div>
    <div id="instructions-overlay" style="position: absolute; top: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.7); padding: 8px 16px; border-radius: 8px; font-size: 12px; color: #aaa; z-index: 10;">Drag handles to control camera</div>
</div>
<div style="display: flex; gap: 20px; justify-content: center; padding: 12px; background: rgba(0, 0, 0, 0.2); border-radius: 0 0 12px 12px;">
    <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #e0e0e0;">
        <div style="width: 14px; height: 14px; border-radius: 50%; background: #00ff88; box-shadow: 0 0 8px #00ff88;"></div>
        <span>Rotation</span>
    </div>
    <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #e0e0e0;">
        <div style="width: 14px; height: 14px; border-radius: 50%; background: #ff69b4; box-shadow: 0 0 8px #ff69b4;"></div>
        <span>Vertical Tilt</span>
    </div>
    <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #e0e0e0;">
        <div style="width: 14px; height: 14px; border-radius: 50%; background: #ffa500; box-shadow: 0 0 8px #ffa500;"></div>
        <span>Zoom/Distance</span>
    </div>
</div>
"""

camera_3d_js = """
() => {
    const wrapper = element.querySelector('#camera-control-wrapper');
    const promptOverlay = element.querySelector('#prompt-overlay');
    const instructionsOverlay = element.querySelector('#instructions-overlay');
    
    // Hide instructions after 3 seconds
    setTimeout(() => {
        if (instructionsOverlay) {
            instructionsOverlay.style.opacity = '0';
            instructionsOverlay.style.transition = 'opacity 0.5s';
            setTimeout(() => instructionsOverlay.remove(), 500);
        }
    }, 3000);
    
    const initScene = () => {
        if (typeof THREE === 'undefined') {
            setTimeout(initScene, 100);
            return;
        }
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        const camera = new THREE.PerspectiveCamera(50, wrapper.clientWidth / wrapper.clientHeight, 0.1, 1000);
        camera.position.set(4.5, 3, 4.5);
        camera.lookAt(0, 0.75, 0);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        wrapper.insertBefore(renderer.domElement, promptOverlay);
        
        // Lighting
        scene.add(new THREE.AmbientLight(0xffffff, 0.5));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
        dirLight.position.set(5, 10, 5);
        scene.add(dirLight);
        
        // Grid with custom colors
        const grid = new THREE.GridHelper(8, 16, 0x2a2a4a, 0x1f1f3a);
        scene.add(grid);
        
        // Constants
        const CENTER = new THREE.Vector3(0, 0.75, 0);
        const BASE_DISTANCE = 1.8;
        const ROTATION_RADIUS = 2.4;
        const TILT_RADIUS = 1.8;
        
        // State - mapped to 2509 control values
        let rotationAngle = 0;      // -90 to 90 degrees (maps to rotate_deg)
        let verticalTilt = 0;       // -1 to 1 (maps to vertical_tilt)
        let zoomLevel = 0;          // 0, 5, 10 (maps to move_forward)
        
        // Snap values for 2509 LoRA
        const rotationSteps = [-90, -45, 0, 45, 90];
        const tiltSteps = [-1, 0, 1];
        const zoomSteps = [0, 5, 10];
        
        function snapToNearest(value, steps) {
            return steps.reduce((prev, curr) => Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev);
        }
        
        // Create placeholder texture
        function createPlaceholderTexture() {
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 256;
            const ctx = canvas.getContext('2d');
            
            // Background gradient
            const gradient = ctx.createLinearGradient(0, 0, 256, 256);
            gradient.addColorStop(0, '#3a3a5a');
            gradient.addColorStop(1, '#2a2a4a');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, 256, 256);
            
            // Simple face placeholder
            ctx.fillStyle = '#ffcc99';
            ctx.beginPath();
            ctx.arc(128, 128, 70, 0, Math.PI * 2);
            ctx.fill();
            
            // Eyes
            ctx.fillStyle = '#333';
            ctx.beginPath();
            ctx.arc(105, 115, 8, 0, Math.PI * 2);
            ctx.arc(151, 115, 8, 0, Math.PI * 2);
            ctx.fill();
            
            // Smile
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(128, 125, 30, 0.2, Math.PI - 0.2);
            ctx.stroke();
            
            // Border
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 4;
            ctx.strokeRect(2, 2, 252, 252);
            
            return new THREE.CanvasTexture(canvas);
        }
        
        // Target image plane
        let currentTexture = createPlaceholderTexture();
        const planeMaterial = new THREE.MeshBasicMaterial({ map: currentTexture, side: THREE.DoubleSide });
        let targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 1.3), planeMaterial);
        targetPlane.position.copy(CENTER);
        scene.add(targetPlane);
        
        // Function to update texture from image URL
        function updateTextureFromUrl(url) {
            if (!url) {
                planeMaterial.map = createPlaceholderTexture();
                planeMaterial.needsUpdate = true;
                scene.remove(targetPlane);
                targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 1.3), planeMaterial);
                targetPlane.position.copy(CENTER);
                scene.add(targetPlane);
                return;
            }
            
            const loader = new THREE.TextureLoader();
            loader.crossOrigin = 'anonymous';
            loader.load(url, (texture) => {
                texture.minFilter = THREE.LinearFilter;
                texture.magFilter = THREE.LinearFilter;
                planeMaterial.map = texture;
                planeMaterial.needsUpdate = true;
                
                const img = texture.image;
                if (img && img.width && img.height) {
                    const aspect = img.width / img.height;
                    const maxSize = 1.5;
                    let planeWidth, planeHeight;
                    if (aspect > 1) {
                        planeWidth = maxSize;
                        planeHeight = maxSize / aspect;
                    } else {
                        planeHeight = maxSize;
                        planeWidth = maxSize * aspect;
                    }
                    scene.remove(targetPlane);
                    targetPlane = new THREE.Mesh(
                        new THREE.PlaneGeometry(planeWidth, planeHeight),
                        planeMaterial
                    );
                    targetPlane.position.copy(CENTER);
                    scene.add(targetPlane);
                }
            });
        }
        
        // Check for initial imageUrl
        if (props.imageUrl) {
            updateTextureFromUrl(props.imageUrl);
        }
        
        // Camera model (stylized)
        const cameraGroup = new THREE.Group();
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0x5588bb, metalness: 0.6, roughness: 0.3 });
        const body = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.2, 0.35), bodyMat);
        cameraGroup.add(body);
        
        const lens = new THREE.Mesh(
            new THREE.CylinderGeometry(0.08, 0.1, 0.16, 16),
            new THREE.MeshStandardMaterial({ color: 0x334455, metalness: 0.7, roughness: 0.2 })
        );
        lens.rotation.x = Math.PI / 2;
        lens.position.z = 0.24;
        cameraGroup.add(lens);
        
        // Lens glass
        const lensGlass = new THREE.Mesh(
            new THREE.CircleGeometry(0.06, 16),
            new THREE.MeshStandardMaterial({ color: 0x88aaff, metalness: 0.9, roughness: 0.1 })
        );
        lensGlass.position.z = 0.33;
        cameraGroup.add(lensGlass);
        
        scene.add(cameraGroup);
        
        // GREEN: Rotation ring (horizontal)
        const rotationRing = new THREE.Mesh(
            new THREE.TorusGeometry(ROTATION_RADIUS, 0.035, 16, 64),
            new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.4 })
        );
        rotationRing.rotation.x = Math.PI / 2;
        rotationRing.position.y = 0.05;
        scene.add(rotationRing);
        
        const rotationHandle = new THREE.Mesh(
            new THREE.SphereGeometry(0.16, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.6 })
        );
        rotationHandle.userData.type = 'rotation';
        scene.add(rotationHandle);
        
        // PINK: Vertical tilt arc
        const arcPoints = [];
        for (let i = 0; i <= 32; i++) {
            const angle = THREE.MathUtils.degToRad(-45 + (90 * i / 32));
            arcPoints.push(new THREE.Vector3(-0.8, TILT_RADIUS * Math.sin(angle) + CENTER.y, TILT_RADIUS * Math.cos(angle)));
        }
        const arcCurve = new THREE.CatmullRomCurve3(arcPoints);
        const tiltArc = new THREE.Mesh(
            new THREE.TubeGeometry(arcCurve, 32, 0.035, 8, false),
            new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.4 })
        );
        scene.add(tiltArc);
        
        const tiltHandle = new THREE.Mesh(
            new THREE.SphereGeometry(0.16, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.6 })
        );
        tiltHandle.userData.type = 'tilt';
        scene.add(tiltHandle);
        
        // ORANGE: Zoom/distance line & handle
        const zoomLineGeo = new THREE.BufferGeometry();
        const zoomLine = new THREE.Line(zoomLineGeo, new THREE.LineBasicMaterial({ color: 0xffa500, linewidth: 2 }));
        scene.add(zoomLine);
        
        const zoomHandle = new THREE.Mesh(
            new THREE.SphereGeometry(0.16, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0xffa500, emissive: 0xffa500, emissiveIntensity: 0.6 })
        );
        zoomHandle.userData.type = 'zoom';
        scene.add(zoomHandle);
        
        function buildPromptText(rot, tilt, zoom) {
            const parts = [];
            
            if (rot !== 0) {
                const dir = rot > 0 ? 'left' : 'right';
                parts.push(`Rotate ${Math.abs(rot)}° ${dir}`);
            }
            
            if (zoom > 5) {
                parts.push('Close-up');
            } else if (zoom >= 1) {
                parts.push('Move forward');
            }
            
            if (tilt <= -1) {
                parts.push("Bird's-eye view");
            } else if (tilt >= 1) {
                parts.push("Worm's-eye view");
            }
            
            return parts.length > 0 ? parts.join(' • ') : 'No camera movement';
        }
        
        function updatePositions() {
            // Map rotation angle to camera position (inverted for visual consistency)
            const rotRad = THREE.MathUtils.degToRad(-rotationAngle);
            
            // Map zoom to distance (higher zoom = closer)
            const zoomFactor = 1 - (zoomLevel / 15);
            const distance = BASE_DISTANCE * zoomFactor;
            
            // Map vertical tilt (-1 to 1) to elevation angle
            const elevationAngle = verticalTilt * 30; // -30 to 30 degrees
            const elRad = THREE.MathUtils.degToRad(elevationAngle);
            
            const camX = distance * Math.sin(rotRad) * Math.cos(elRad);
            const camY = distance * Math.sin(elRad) + CENTER.y;
            const camZ = distance * Math.cos(rotRad) * Math.cos(elRad);
            
            cameraGroup.position.set(camX, camY, camZ);
            cameraGroup.lookAt(CENTER);
            
            // Update rotation handle position
            const handleRotRad = THREE.MathUtils.degToRad(-rotationAngle);
            rotationHandle.position.set(
                ROTATION_RADIUS * Math.sin(handleRotRad),
                0.05,
                ROTATION_RADIUS * Math.cos(handleRotRad)
            );
            
            // Update tilt handle position
            const tiltAngle = verticalTilt * 30;
            const tiltRad = THREE.MathUtils.degToRad(tiltAngle);
            tiltHandle.position.set(
                -0.8,
                TILT_RADIUS * Math.sin(tiltRad) + CENTER.y,
                TILT_RADIUS * Math.cos(tiltRad)
            );
            
            // Update zoom handle position (along camera-to-center line)
            const zoomDist = distance - 0.4;
            zoomHandle.position.set(
                zoomDist * Math.sin(rotRad) * Math.cos(elRad),
                zoomDist * Math.sin(elRad) + CENTER.y,
                zoomDist * Math.cos(rotRad) * Math.cos(elRad)
            );
            zoomLineGeo.setFromPoints([cameraGroup.position.clone(), CENTER.clone()]);
            
            // Update prompt overlay
            const rotSnap = snapToNearest(rotationAngle, rotationSteps);
            const tiltSnap = snapToNearest(verticalTilt, tiltSteps);
            const zoomSnap = snapToNearest(zoomLevel, zoomSteps);
            promptOverlay.textContent = buildPromptText(rotSnap, tiltSnap, zoomSnap);
        }
        
        function updatePropsAndTrigger() {
            const rotSnap = snapToNearest(rotationAngle, rotationSteps);
            const tiltSnap = snapToNearest(verticalTilt, tiltSteps);
            const zoomSnap = snapToNearest(zoomLevel, zoomSteps);
            
            props.value = { rotate_deg: rotSnap, vertical_tilt: tiltSnap, move_forward: zoomSnap };
            trigger('change', props.value);
        }
        
        // Raycasting for interaction
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        let isDragging = false;
        let dragTarget = null;
        let dragStartMouse = new THREE.Vector2();
        let dragStartZoom = 0;
        const intersection = new THREE.Vector3();
        
        const canvas = renderer.domElement;
        
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects([rotationHandle, tiltHandle, zoomHandle]);
            
            if (intersects.length > 0) {
                isDragging = true;
                dragTarget = intersects[0].object;
                dragTarget.material.emissiveIntensity = 1.0;
                dragTarget.scale.setScalar(1.3);
                dragStartMouse.copy(mouse);
                dragStartZoom = zoomLevel;
                canvas.style.cursor = 'grabbing';
            }
        });
        
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            
            if (isDragging && dragTarget) {
                raycaster.setFromCamera(mouse, camera);
                
                if (dragTarget.userData.type === 'rotation') {
                    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        let angle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                        // Clamp to -90 to 90 range
                        rotationAngle = THREE.MathUtils.clamp(-angle, -90, 90);
                    }
                } else if (dragTarget.userData.type === 'tilt') {
                    const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0.8);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        const relY = intersection.y - CENTER.y;
                        const relZ = intersection.z;
                        const angle = THREE.MathUtils.radToDeg(Math.atan2(relY, relZ));
                        // Map -30 to 30 degrees to -1 to 1
                        verticalTilt = THREE.MathUtils.clamp(angle / 30, -1, 1);
                    }
                } else if (dragTarget.userData.type === 'zoom') {
                    const deltaY = mouse.y - dragStartMouse.y;
                    // Map drag to 0-10 range
                    zoomLevel = THREE.MathUtils.clamp(dragStartZoom - deltaY * 12, 0, 10);
                }
                updatePositions();
            } else {
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects([rotationHandle, tiltHandle, zoomHandle]);
                [rotationHandle, tiltHandle, zoomHandle].forEach(h => {
                    h.material.emissiveIntensity = 0.6;
                    h.scale.setScalar(1);
                });
                if (intersects.length > 0) {
                    intersects[0].object.material.emissiveIntensity = 0.9;
                    intersects[0].object.scale.setScalar(1.15);
                    canvas.style.cursor = 'grab';
                } else {
                    canvas.style.cursor = 'default';
                }
            }
        });
        
        const onMouseUp = () => {
            if (dragTarget) {
                dragTarget.material.emissiveIntensity = 0.6;
                dragTarget.scale.setScalar(1);
                
                // Snap and animate
                const targetRot = snapToNearest(rotationAngle, rotationSteps);
                const targetTilt = snapToNearest(verticalTilt, tiltSteps);
                const targetZoom = snapToNearest(zoomLevel, zoomSteps);
                
                const startRot = rotationAngle, startTilt = verticalTilt, startZoom = zoomLevel;
                const startTime = Date.now();
                
                function animateSnap() {
                    const t = Math.min((Date.now() - startTime) / 200, 1);
                    const ease = 1 - Math.pow(1 - t, 3);
                    
                    rotationAngle = startRot + (targetRot - startRot) * ease;
                    verticalTilt = startTilt + (targetTilt - startTilt) * ease;
                    zoomLevel = startZoom + (targetZoom - startZoom) * ease;
                    
                    updatePositions();
                    if (t < 1) requestAnimationFrame(animateSnap);
                    else updatePropsAndTrigger();
                }
                animateSnap();
            }
            isDragging = false;
            dragTarget = null;
            canvas.style.cursor = 'default';
        };
        
        canvas.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('mouseleave', onMouseUp);
        
        // Touch support
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects([rotationHandle, tiltHandle, zoomHandle]);
            
            if (intersects.length > 0) {
                isDragging = true;
                dragTarget = intersects[0].object;
                dragTarget.material.emissiveIntensity = 1.0;
                dragTarget.scale.setScalar(1.3);
                dragStartMouse.copy(mouse);
                dragStartZoom = zoomLevel;
            }
        }, { passive: false });
        
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
            
            if (isDragging && dragTarget) {
                raycaster.setFromCamera(mouse, camera);
                
                if (dragTarget.userData.type === 'rotation') {
                    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        let angle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                        rotationAngle = THREE.MathUtils.clamp(-angle, -90, 90);
                    }
                } else if (dragTarget.userData.type === 'tilt') {
                    const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0.8);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        const relY = intersection.y - CENTER.y;
                        const relZ = intersection.z;
                        const angle = THREE.MathUtils.radToDeg(Math.atan2(relY, relZ));
                        verticalTilt = THREE.MathUtils.clamp(angle / 30, -1, 1);
                    }
                } else if (dragTarget.userData.type === 'zoom') {
                    const deltaY = mouse.y - dragStartMouse.y;
                    zoomLevel = THREE.MathUtils.clamp(dragStartZoom - deltaY * 12, 0, 10);
                }
                updatePositions();
            }
        }, { passive: false });
        
        canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            onMouseUp();
        }, { passive: false });
        
        canvas.addEventListener('touchcancel', (e) => {
            e.preventDefault();
            onMouseUp();
        }, { passive: false });
        
        // Initial update
        updatePositions();
        
        // Render loop
        function render() {
            requestAnimationFrame(render);
            renderer.render(scene, camera);
        }
        render();
        
        // Handle resize
        new ResizeObserver(() => {
            camera.aspect = wrapper.clientWidth / wrapper.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
        }).observe(wrapper);
        
        // Store update functions
        wrapper._updateFromProps = (newVal) => {
            if (newVal && typeof newVal === 'object') {
                rotationAngle = newVal.rotate_deg ?? rotationAngle;
                verticalTilt = newVal.vertical_tilt ?? verticalTilt;
                zoomLevel = newVal.move_forward ?? zoomLevel;
                updatePositions();
            }
        };
        
        wrapper._updateTexture = updateTextureFromUrl;
        
        // Watch for prop changes
        let lastImageUrl = props.imageUrl;
        let lastValue = JSON.stringify(props.value);
        setInterval(() => {
            if (props.imageUrl !== lastImageUrl) {
                lastImageUrl = props.imageUrl;
                updateTextureFromUrl(props.imageUrl);
            }
            const currentValue = JSON.stringify(props.value);
            if (currentValue !== lastValue) {
                lastValue = currentValue;
                if (props.value && typeof props.value === 'object') {
                    rotationAngle = props.value.rotate_deg ?? rotationAngle;
                    verticalTilt = props.value.vertical_tilt ?? verticalTilt;
                    zoomLevel = props.value.move_forward ?? zoomLevel;
                    updatePositions();
                }
            }
        }, 100);
    };
    
    initScene();
}
"""


with gr.Blocks() as demo:
    gr.Markdown("""
    # 🎬 Qwen Image Edit 2509 — 3D Camera Control
    
    Control camera angles using the **3D viewport** or **sliders**.  
    Using [dx8152's Qwen-Edit-2509-Multiple-angles LoRA](https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles) and [Phr00t/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/tree/main) for 4-step inference 💨
    """)
    
    with gr.Row():
        # Left column: Input and controls
        with gr.Column(scale=1):
            image = gr.Image(label="Input Image", type="pil", height=280)
            prev_output = gr.Image(value=None, visible=False)
            is_reset = gr.Checkbox(value=False, visible=False)
            
            gr.Markdown("### 🎮 3D Camera Control")
            gr.Markdown("*Drag the colored handles: 🟢 Rotation, 🩷 Vertical Tilt, 🟠 Zoom*")
            
            camera_3d = gr.HTML(
                value={"rotate_deg": 0, "vertical_tilt": 0, "move_forward": 0},
                elem_id="camera-3d-control",
                html_template=camera_3d_html,
                js_on_load=camera_3d_js
            )
            
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset", variant="secondary")
                run_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
            
            gr.Markdown("### 🎚️ Slider Controls")
            
            rotate_deg = gr.Slider(
                label="Rotate Left-Right (degrees °)",
                minimum=-90,
                maximum=90,
                step=45,
                value=0,
                info="Positive = left, Negative = right"
            )
            
            move_forward = gr.Slider(
                label="Move Forward → Close-Up",
                minimum=0,
                maximum=10,
                step=5,
                value=0,
                info="0 = normal, 5 = forward, 10 = close-up"
            )
            
            vertical_tilt = gr.Slider(
                label="Vertical Angle (Bird ↔ Worm)",
                minimum=-1,
                maximum=1,
                step=1,
                value=0,
                info="-1 = bird's eye, 0 = eye level, 1 = worm's eye"
            )
            
            wideangle = gr.Checkbox(label="🔲 Wide-Angle Lens", value=False)
            
            prompt_preview = gr.Textbox(
                label="Generated Prompt",
                value="no camera movement",
                interactive=False
            )
        
        # Right column: Output
        with gr.Column(scale=1):
            result = gr.Image(label="Output Image", height=500, interactive=False)
            
            create_video_button = gr.Button(
                "🎥 Create Video Between Images",
                variant="secondary",
                visible=False
            )
            
            with gr.Group(visible=False) as video_group:
                video_output = gr.Video(
                    label="Generated Video",
                    autoplay=True
                )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                true_guidance_scale = gr.Slider(label="True Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=40, step=1, value=4)
                height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=1024)
                width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=1024)
    
    # --- Event Handlers ---
    
    def update_prompt_from_sliders(rotate, forward, tilt, wide):
        """Update prompt preview when sliders change."""
        return build_camera_prompt(rotate, forward, tilt, wide)
    
    def sync_3d_to_sliders(camera_value):
        """Sync 3D control changes to sliders."""
        if camera_value and isinstance(camera_value, dict):
            rot = camera_value.get('rotate_deg', 0)
            tilt = camera_value.get('vertical_tilt', 0)
            zoom = camera_value.get('move_forward', 0)
            prompt = build_camera_prompt(rot, zoom, tilt, False)
            return rot, zoom, tilt, prompt
        return gr.update(), gr.update(), gr.update(), gr.update()
    
    def sync_sliders_to_3d(rotate, forward, tilt):
        """Sync slider changes to 3D control."""
        return {"rotate_deg": rotate, "vertical_tilt": tilt, "move_forward": forward}
    
    def update_3d_image(image):
        """Update the 3D component with the uploaded image."""
        if image is None:
            return gr.update(imageUrl=None)
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        return gr.update(imageUrl=data_url)
    
    # Slider -> Prompt preview
    for slider in [rotate_deg, move_forward, vertical_tilt]:
        slider.change(
            fn=update_prompt_from_sliders,
            inputs=[rotate_deg, move_forward, vertical_tilt, wideangle],
            outputs=[prompt_preview]
        )
    
    wideangle.change(
        fn=update_prompt_from_sliders,
        inputs=[rotate_deg, move_forward, vertical_tilt, wideangle],
        outputs=[prompt_preview]
    )
    
    # 3D control -> Sliders + Prompt
    camera_3d.change(
        fn=sync_3d_to_sliders,
        inputs=[camera_3d],
        outputs=[rotate_deg, move_forward, vertical_tilt, prompt_preview]
    )
    
    # Sliders -> 3D control
    for slider in [rotate_deg, move_forward, vertical_tilt]:
        slider.release(
            fn=sync_sliders_to_3d,
            inputs=[rotate_deg, move_forward, vertical_tilt],
            outputs=[camera_3d]
        )
    
    # Reset behavior
    reset_btn.click(
        fn=reset_all,
        inputs=None,
        outputs=[rotate_deg, move_forward, vertical_tilt, wideangle, is_reset],
        queue=False
    ).then(
        fn=end_reset,
        inputs=None,
        outputs=[is_reset],
        queue=False
    ).then(
        fn=sync_sliders_to_3d,
        inputs=[rotate_deg, move_forward, vertical_tilt],
        outputs=[camera_3d]
    )
    
    # Manual generation with video button visibility
    def infer_and_show_video_button(*args):
        result_img, result_seed, result_prompt = infer_camera_edit(*args)
        show_button = args[0] is not None and result_img is not None
        return result_img, result_seed, result_prompt, gr.update(visible=show_button)
    
    inputs = [
        image, rotate_deg, move_forward,
        vertical_tilt, wideangle,
        seed, randomize_seed, true_guidance_scale, num_inference_steps, height, width, prev_output
    ]
    outputs = [result, seed, prompt_preview]
    
    run_event = run_btn.click(
        fn=infer_and_show_video_button,
        inputs=inputs,
        outputs=outputs + [create_video_button]
    )
    
    # Video creation
    create_video_button.click(
        fn=lambda: gr.update(visible=True),
        outputs=[video_group],
        api_visibility="private"
    ).then(
        fn=create_video_between_images,
        inputs=[image, result, prompt_preview],
        outputs=[video_output],
        api_visibility="private"
    )
    
    # Image upload -> update dimensions AND update 3D preview
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
    ).then(
        fn=update_3d_image,
        inputs=[image],
        outputs=[camera_3d]
    )
    
    # Handle image clear
    image.clear(
        fn=lambda: gr.update(imageUrl=None),
        outputs=[camera_3d]
    )
    
    # Live updates
    def maybe_infer(is_reset_flag, *args):
        if is_reset_flag:
            return gr.update(), gr.update(), gr.update(), gr.update()
        else:
            result_img, result_seed, result_prompt = infer_camera_edit(*args)
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
        cache_examples=True,
        cache_mode="lazy",
        elem_id="examples"
    )
    
    gr.api(infer_camera_edit, api_name="infer_edit_camera_angles")
    gr.api(create_video_between_images, api_name="create_video_between_images")

if __name__ == "__main__":
    demo.launch(
        head=three_js_head,
        mcp_server=True,
        theme=gr.themes.Citrus(),
        css=css,
        footer_links=["api", "gradio", "settings"]
    )